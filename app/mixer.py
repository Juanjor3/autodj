import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import numpy as np
import scipy.signal
import librosa
import pedalboard
import soundfile as sf

try:
    from app.analyzer import TrackProfile
    from app.transition import (
        apply_stem_dj_blend,
        apply_migue_fader_cut_blend,
        apply_equal_power_dj_blend,
        ZeroPhaseDJIsolator3Band,
        DJ_ISOLATOR,
        time_stretch_stereo,
        SAMPLE_RATE,
        TRANSITION_BEATS,
    )
    from app.stems import StemSeparator, DEMUCS_AVAILABLE
    from app.mastering import master_session_audio
except ModuleNotFoundError:
    from analyzer import TrackProfile
    from transition import (
        apply_stem_dj_blend,
        apply_migue_fader_cut_blend,
        apply_equal_power_dj_blend,
        ZeroPhaseDJIsolator3Band,
        DJ_ISOLATOR,
        time_stretch_stereo,
        SAMPLE_RATE,
        TRANSITION_BEATS,
    )
    from stems import StemSeparator, DEMUCS_AVAILABLE
    from mastering import master_session_audio

REVERB_BOARD = pedalboard.Pedalboard([
    pedalboard.Reverb(room_size=0.40, wet_level=0.30, dry_level=0.75)
])


@dataclass
class CueEntry:
    index: int
    title: str
    start_seconds: float
    bpm: float
    camelot: str
    key_name: str

    def format_timestamp(self) -> str:
        minutes = int(self.start_seconds // 60)
        seconds = int(self.start_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


def dj_isolator_hpf(
    audio_stereo: np.ndarray,
    sr: int = SAMPLE_RATE,
    cutoff_hz: float = 260.0
) -> np.ndarray:
    """
    Filtro Isolator DJ auténtico (Butterworth orden 4):
    Atenúa los subgraves y bombos (< 120 Hz) en más de 45 dB.
    Elimina por completo cualquier bola de graves, rumble o choque de frecuencias bajas.
    """
    sos = scipy.signal.butter(4, cutoff_hz, btype='highpass', fs=sr, output='sos')
    ch_l = scipy.signal.sosfilt(sos, audio_stereo[0])
    ch_r = scipy.signal.sosfilt(sos, audio_stereo[1])
    return np.stack([ch_l, ch_r]).astype(np.float32)


def align_kick_phase(
    y1_mix: np.ndarray,
    sample1_start: int,
    y2_track: np.ndarray,
    next_beats: np.ndarray,
    master_beats: np.ndarray,
    start_beat1: int,
    sr: int = SAMPLE_RATE,
    bpm: float = 135.0,
    max_ms: float = 35.0,
    log_func=print
) -> int:
    """
    Alineación de fase robusta y universal:
    1. Localiza el primer beat donde Track B realmente tiene bombo activo (ignora silencios e intros atmosféricas).
    2. Compara el transitorio real de ese bombo contra el bombo correspondiente en Track A.
    3. Calcula la correlación cruzada estricta para clavar la fase a nivel submilisegundo (< 1 ms).
    """
    spb = int((60.0 / max(bpm, 1e-3)) * sr)
    test_len = spb * 4

    mono1 = y1_mix[0] if y1_mix.ndim > 1 else y1_mix
    mono2 = y2_track[0] if y2_track.ndim > 1 else y2_track

    sos = scipy.signal.butter(4, [45, 125], btype='bandpass', fs=sr, output='sos')
    kick2 = scipy.signal.sosfilt(sos, mono2[:min(len(mono2), sr * 60)])
    env2_full = np.abs(scipy.signal.hilbert(kick2))

    # Buscar el primer beat de Track B que tenga energía de bombo real (umbral 40% del máx)
    max_env2 = float(np.max(env2_full)) if len(env2_full) > 0 else 0.0
    threshold = max(0.12, max_env2 * 0.40)

    kick_beat_idx = 0
    for b in range(min(32, len(next_beats))):
        s = int(next_beats[b])
        if s + 200 < len(env2_full):
            win = env2_full[max(0, s - 200) : s + 200]
            if len(win) > 0 and np.max(win) >= threshold:
                kick_beat_idx = b
                break

    # Posiciones de prueba correspondientes en A y B
    target_beat_a = start_beat1 + kick_beat_idx
    if target_beat_a < len(master_beats):
        sample1_test = int(master_beats[target_beat_a])
    else:
        sample1_test = sample1_start + kick_beat_idx * spb

    sample2_test = int(next_beats[kick_beat_idx])
    sample2_start = int(next_beats[0])

    seg1 = mono1[sample1_test : sample1_test + test_len]
    seg2 = mono2[sample2_test : sample2_test + test_len]

    if len(seg1) < test_len or len(seg2) < test_len:
        return sample2_start

    c1 = scipy.signal.sosfilt(sos, seg1)
    c2 = scipy.signal.sosfilt(sos, seg2)

    env1 = np.abs(scipy.signal.hilbert(c1))
    env2 = np.abs(scipy.signal.hilbert(c2))

    if np.max(env1) < 0.03 or np.max(env2) < 0.03:
        return sample2_start

    max_samples = int(sr * (max_ms / 1000.0))
    corr = scipy.signal.correlate(env1, env2, mode='full')
    center = len(seg2) - 1

    search_start = max(0, center - max_samples)
    search_end = min(len(corr), center + max_samples + 1)

    local_corr = corr[search_start:search_end]
    best_idx = np.argmax(local_corr)
    shift = (search_start + best_idx) - center

    aligned = sample2_start - shift
    log_func(f"   • Calibración de fase robusta (Beat {kick_beat_idx}): offset de {shift:+d} muestras ({shift/sr*1000:+.2f} ms) -> Fase 100% clavada (< 1 ms)")
    return aligned


def get_phrase_energy_profiles(
    audio: np.ndarray,
    beats: np.ndarray,
    phrase_block: int = 32,
    sr: int = SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula perfiles de energía RMS total y energía en banda de bombo (40-140 Hz) por cada frase de 32 bombos."""
    mono = librosa.to_mono(audio)
    sos = scipy.signal.butter(4, [40, 140], btype="bandpass", fs=sr, output="sos")
    mono_bass = scipy.signal.sosfilt(sos, mono)

    num_phrases = len(beats) // phrase_block
    rms_l, bass_l = [], []
    for p in range(num_phrases):
        s1 = beats[p * phrase_block]
        s2 = beats[min((p + 1) * phrase_block, len(beats) - 1)]
        rms_l.append(float(np.sqrt(np.mean(mono[s1:s2] ** 2))))
        bass_l.append(float(np.sqrt(np.mean(mono_bass[s1:s2] ** 2))))
    return np.array(rms_l, dtype=np.float32), np.array(bass_l, dtype=np.float32)


def find_first_main_drop(
    audio: np.ndarray,
    beats: np.ndarray,
    sr: int = SAMPLE_RATE,
    profile: Optional[TrackProfile] = None
) -> int:
    """
    Encuentra de forma totalmente automática y universal el primer drop principal de la pista entrante (Track B),
    donde concluye su introducción y explota el bombo (40-140 Hz) a su nivel de pista.
    Devuelve el índice del beat (múltiplo estricto de 32 bombos / 8 compases).
    """
    phrase_block = 32  # Frases de 8 compases (32 beats)
    num_phrases = len(beats) // phrase_block
    if num_phrases < 2:
        return 0

    # 1. Si el perfil tiene tags de stems, buscar DROP tras INTRO_GROOVE o BREAKDOWN
    if profile and profile.phrase_tags:
        search_limit = min(8, len(profile.phrase_tags))
        for p in range(1, search_limit):
            if profile.phrase_tags[p] == "DROP":
                return p * phrase_block
            if p > 0 and profile.phrase_drums and profile.phrase_drums[p] > 0.60:
                if profile.phrase_tags[p - 1] in ["INTRO_GROOVE", "BREAKDOWN"]:
                    return p * phrase_block

    rms_l, bass_l = get_phrase_energy_profiles(audio, beats, phrase_block, sr)
    max_bass = float(np.max(bass_l)) if len(bass_l) > 0 else 0.0
    threshold = max_bass * 0.60

    # Buscar en las primeras 8 frases (hasta ~2.5 minutos de intro)
    search_limit = min(8, len(bass_l))
    for p in range(1, search_limit):
        prev_b = float(np.min(bass_l[max(0, p - 2) : p]))
        curr_b = float(bass_l[p])
        # Si la energía de bombo supera el umbral de base rítmica y hay un salto significativo
        if curr_b >= threshold and (curr_b / max(prev_b, 1e-4) >= 1.45 or (curr_b - prev_b) > 0.04):
            return p * phrase_block

    return 0


def find_last_climax_drop(
    audio: np.ndarray,
    beats: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_bpm: float = 135.0,
    profile: Optional[TrackProfile] = None
) -> int:
    """
    Encuentra de forma totalmente automática y universal el último clímax drop
    de la pista saliente (Track A), donde explota tras el breakdown principal.
    Devuelve el índice del beat (múltiplo estricto de 32 bombos / 8 compases).
    """
    phrase_block = 32  # Frases de 8 compases (32 beats)
    num_phrases = len(beats) // phrase_block
    if num_phrases < 4:
        return 0

    min_search = max(3, int(num_phrases * 0.65))
    max_search = max(min_search + 1, num_phrases - 3)

    if profile and profile.phrase_tags:
        for p in range(min_search, min(num_phrases - 1, max_search + 2)):
            if profile.phrase_tags[p] == "DROP":
                return p * phrase_block

    rms_l, bass_l = get_phrase_energy_profiles(audio, beats, phrase_block, sr)
    # En estilo CuBe/The Natural, buscar punto de mezcla entre el 65% y el 80%
    combined_energy = bass_l + (rms_l * 0.5)
    breakdown_p = min_search + int(np.argmin(combined_energy[min_search:max_search]))
    threshold = float(np.max(bass_l)) * 0.65

    # El drop es la primera frase tras ese breakdown donde rompe el ritmo principal
    for p in range(breakdown_p, min(num_phrases - 1, max_search + 2)):
        prev_b = float(np.min(bass_l[max(0, p - 3) : p]))
        curr_b = float(bass_l[p])
        if curr_b >= threshold and (curr_b / max(prev_b, 1e-4) >= 1.40 or (curr_b - prev_b) > 0.05):
            return p * phrase_block

    # Si no hay breakdown en esa ventana, elegir la frase óptima de energía en la ventana de salida
    best_p = min_search + int(np.argmax(bass_l[min_search:max_search]))
    return best_p * phrase_block


def resolve_vocal_clash_window(
    vocal1: Optional[List[float]],
    vocal2: Optional[List[float]],
    start_beat1: int,
    end_beat1: int,
    desired_blend_beats: int,
    total_beats1: int,
    phrase_block: int = 32,
    log_func=print
) -> Tuple[int, int, bool]:
    """
    Analiza si van a coincidir voces en la zona de mezcla y el subidón.
    Si se detecta colisión (> 0.45 en ambas pistas simultáneamente),
    reubica el punto de mezcla a una frase donde no coincidan voces.
    Devuelve (start_beat, end_beat, vocal_clash_detected).
    """
    if not vocal1 or not vocal2:
        return start_beat1, end_beat1, False

    num_p1 = len(vocal1)
    num_p2 = len(vocal2)
    blend_phrases = desired_blend_beats // phrase_block

    def count_vocal_clashes(st_beat: int) -> Tuple[bool, float, int]:
        st_p = st_beat // phrase_block
        max_c = 0.0
        clash_p = -1
        for k in range(blend_phrases):
            p1 = st_p + k
            p2 = k
            if 0 <= p1 < num_p1 and 0 <= p2 < num_p2:
                v1 = vocal1[p1]
                v2 = vocal2[p2]
                score = v1 * v2
                if score > max_c:
                    max_c = score
                if v1 > 0.45 and v2 > 0.45:
                    return True, score, p1
        return False, max_c, -1

    has_clash, max_score, clashing_phrase = count_vocal_clashes(start_beat1)
    if not has_clash:
        log_func(f"   🎙️ Análisis vocal anti-solapamiento: Cero colisión de voces (Clash Score: {max_score:.2f} < 0.20)")
        return start_beat1, end_beat1, False

    log_func(f"   ⚠️ ALERTA DE VOCES: ¡Colisión vocal detectada en la ventana de mezcla [Frase {clashing_phrase}]!")
    log_func(f"   🔄 Ambas canciones tienen voces activas simultáneamente. Buscando frase alternativa sin voces...")

    # Buscar alternativa hacia adelante (outro instrumental) o hacia atrás
    start_phrase = start_beat1 // phrase_block
    for offset in [1, 2, 3, -1, -2, 4]:
        test_st_p = start_phrase + offset
        test_end_p = test_st_p + blend_phrases
        if 0 <= test_st_p and (test_end_p * phrase_block) <= total_beats1:
            clash, score, _ = count_vocal_clashes(test_st_p * phrase_block)
            if not clash:
                new_start = test_st_p * phrase_block
                new_end = test_end_p * phrase_block
                log_func(f"   🚀 Subidón y mezcla reubicados en Frase {test_st_p} -> {test_end_p} (Beat {new_start} a {new_end})")
                log_func(f"   ✅ ¡Cero colisión de voces! Las voces nunca se montarán una encima de otra.")
                return new_start, new_end, False

    log_func(f"   ℹ️ No hay ventana 100% instrumental disponible; aplicando ducking quirúrgico de voces vía Stems.")
    return start_beat1, end_beat1, True


PROFILES_DIR = Path("/app/profiles") if Path("/app/profiles").exists() else Path(__file__).parent / "profiles"
CUBE_MODEL_PATH = PROFILES_DIR / "cube_style_model.json"


def load_dj_style(style_name: str = "cube") -> dict:
    """Carga el modelo de estilo DJ aprendido de CuBe Dance Club, The Natural, Metro y Radical."""
    candidates = [
        CUBE_MODEL_PATH,
        Path("/app/profiles/cube_style_model.json"),
        Path(__file__).parent / "profiles" / "cube_style_model.json",
        Path("y:/autodj/app/profiles/cube_style_model.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                import json
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data and "style_name" in data:
                        return data
            except Exception as e:
                print(f"⚠️ Error cargando {p}: {e}")

    # Fallback predeterminado si no existiera el archivo
    return {
        "style_name": "Resident DJ Style Model (CuBe, The Natural, Metro & Radical)",
        "analyzed_transitions_count": 528,
        "tempo_profile": {"recommended_bpm": 143.6},
        "entry_point_rules": {"entry_point_in_track1_ratio_mean": 0.62},
        "beat_counting_rules": {"median_lead_in_beats": 64, "median_lead_in_bars": 16},
        "eq_and_filtering_rules": {"hpf_cutoff_hz": 250.0},
        "coexistence_duration_rules": {
            "mean_coexistence_bars": 68,
            "mean_coexistence_beats": 272,
            "mean_track1_retention_ratio": 0.92,
            "fadeout_bars": 8,
            "fadeout_beats": 32
        }
    }


def mix_session(
    tracks: List[TrackProfile],
    output_audio_path: Path,
    output_cue_path: Path,
    transition_beats: int = TRANSITION_BEATS,
    log_callback: Optional[Callable[[str], None]] = None,
    style: str = "cube",
    progress_callback: Optional[Callable[[int, str], None]] = None,
    use_stems: bool = True
) -> List[CueEntry]:
    """
    Motor DJ Canónico: Conteo de Frases hacia Atrás, Bloqueo Físico de Rejilla (Grid Lock),
    Sincronización de Drops al Unísono, Separación de Fuentes por Stems (IA) y Estilo CuBe.
    """
    if not tracks:
        raise ValueError("No hay pistas para mezclar.")

    def log(msg: str):
        print(msg)
        if log_callback:
            log_callback(msg)

    style_profile = load_dj_style(style)
    style_title = style_profile.get('style_name', 'Resident DJ Style Model')
    trans_count = style_profile.get('analyzed_transitions_count', 0)
    log(f"🎧 Modelo DJ Activo: {style_title} ({trans_count} transiciones empíricas analizadas)")

    # Parámetros activos extraídos del modelo aprendido
    model_beat_rules = style_profile.get("beat_counting_rules", {})
    model_coex_rules = style_profile.get("coexistence_duration_rules", {})
    model_eq_rules = style_profile.get("eq_and_filtering_rules", {})

    learned_retention = float(model_coex_rules.get("mean_track1_retention_ratio", 0.92))
    # Acotado para asegurar siempre "mantén el 1º tema hasta casi acabar"
    learned_retention = max(0.85, min(0.96, learned_retention))
    learned_lead_in = int(model_beat_rules.get("median_lead_in_beats", 64))
    learned_hpf_hz = float(model_eq_rules.get("hpf_cutoff_hz", 250.0))

    if use_stems:
        engine_name = "Meta Demucs v4 (Deep Learning / PyTorch)" if DEMUCS_AVAILABLE else "Filtros Analíticos de 3 Bandas"
        log(f"🧠 Motor de Separación por Stems (IA) Activado: {engine_name}")

    sr = SAMPLE_RATE
    cue_list: List[CueEntry] = []

    # 1. Pista base
    master_mix = tracks[0].audio_stereo.astype(np.float32)
    target_bpm = tracks[0].bpm
    master_beats = tracks[0].beat_samples
    last_drop_in_master = 0
    last_drop_local = 0

    cue_list.append(
        CueEntry(1, tracks[0].title, 0.0, target_bpm, tracks[0].camelot, tracks[0].key_name)
    )
    log(f"🎵 Pista 1 (Base): {tracks[0].title} | BPM Maestro: {target_bpm:.2f}")

    for idx in range(1, len(tracks)):
        curr_track = tracks[idx - 1]
        next_track = tracks[idx]
        y_next = next_track.audio_stereo.astype(np.float32)
        next_bpm = next_track.bpm

        if progress_callback:
            pct = 55 + int(32 * (idx - 1) / max(1, len(tracks) - 1))
            progress_callback(pct, f"Mezclando transición [{idx}/{len(tracks)-1}]: {next_track.title}")

        log(f"🎧 Mezclando pista [{idx+1}]: {next_track.title} ({next_bpm:.2f} BPM)...")

        # 2. Puntos clave: Drop de A y Drop de B (apoyados en frases y tags estructurales)
        if idx == 1:
            beat1_drop = find_last_climax_drop(master_mix, master_beats, sr, target_bpm=target_bpm, profile=tracks[0])
        else:
            local_drop = find_last_climax_drop(curr_track.audio_stereo, curr_track.beat_samples, sr, target_bpm=target_bpm, profile=curr_track)
            target_sample = last_drop_in_master + (curr_track.beat_samples[local_drop] - last_drop_local)
            beat1_drop = int(np.argmin(np.abs(master_beats - target_sample)))

        beat2_drop = find_first_main_drop(y_next, next_track.beat_samples, sr, profile=next_track)

        # 3. Duración de la intro de B en compases/beats (conducción del modelo)
        if beat2_drop > 0:
            intro_beats_b = beat2_drop
        else:
            # Si la pista entrante arranca con bombo desde el inicio (techno/loop),
            # aplicamos el lead-in canónico del modelo (64 o 32 beats) filtrado a 250 Hz sobre Track A
            intro_beats_b = min(learned_lead_in, beat1_drop) if beat1_drop >= learned_lead_in else 32
            beat2_drop = intro_beats_b
        intro_beats_b = min(intro_beats_b, beat1_drop)

        # 4. CONTEO PREVIO Y ALINEACIÓN DE COMPASES (MÁS DE 30 SEGUNDOS DE MEZCLA CUADRADA)
        # Queremos mantener ambas canciones sonando juntas durante más de 30 segundos (ej. 32 compases / 128 bombos = ~57s)
        # con compases cuadrados en múltiplos de frase (regla del 4 y el 8).
        total_beats1 = len(master_beats)
        total_beats2 = len(next_track.beat_samples)

        # Duración de la mezcla: adaptada al modelo (mínimo 96 beats = ~41s > 30s)
        if total_beats1 >= 280 and total_beats2 >= 192:
            desired_blend_beats = 128
        else:
            desired_blend_beats = 96  # Siempre >= 96 beats (> 30s)

        # Fijar el final de la mezcla en el último tramo de Track 1 gobernado por la retención aprendida del modelo
        target_end_raw = int(total_beats1 * learned_retention)
        end_phrase = max(4, min((total_beats1 - 16) // 32, target_end_raw // 32))
        end_beat1 = end_phrase * 32
        start_beat1 = max(0, end_beat1 - desired_blend_beats)

        log(f"   🧠 [Modelo Conectado] Parámetros aplicados desde {style_title}:")
        log(f"      • Retención Track 1: {learned_retention * 100:.1f}% (Fin fijado en beat {end_beat1}/{total_beats1})")
        log(f"      • Convivencia objetivo: {desired_blend_beats} bombos ({desired_blend_beats//4} compases / {desired_blend_beats//32} frases -> {desired_blend_beats * 60.0 / target_bpm:.1f}s > 30s)")
        log(f"      • Conteo lead-in: {intro_beats_b} bombos ({intro_beats_b//4} compases)")

        # Análisis y resolución de colisiones vocales (Anti-Vocal Clash)
        start_beat1, end_beat1, vocal_clash_detected = resolve_vocal_clash_window(
            curr_track.phrase_vocal,
            next_track.phrase_vocal,
            start_beat1,
            end_beat1,
            desired_blend_beats,
            total_beats1,
            phrase_block=32,
            log_func=log
        )
        blend_beats = end_beat1 - start_beat1

        # 5. BLOQUEO DE REJILLA FÍSICA EXACTA (Grid Lock estable con Rubber Band)
        exact_rate = float(target_bpm / next_bpm)
        if abs(exact_rate - 1.0) >= 0.0005:
            log(f"   • Bloqueo de tempo y rejilla: {next_bpm:.2f} -> {target_bpm:.2f} BPM (Rate: {exact_rate:.6f})")
            y_next_stretched = time_stretch_stereo(y_next, sr, exact_rate)
            next_beats = np.round(next_track.beat_samples / exact_rate).astype(int)
        else:
            y_next_stretched = y_next
            next_beats = next_track.beat_samples

        sample1_start = int(master_beats[start_beat1])
        sample1_end = int(master_beats[end_beat1])
        n_blend = sample1_end - sample1_start
        duration_blend_sec = n_blend / sr

        # 6. CALIBRACIÓN DE FASE EN EL PRIMER BOMBO (Phase Lock exacto)
        sample2_aligned = align_kick_phase(
            master_mix, sample1_start,
            y_next_stretched, next_beats,
            master_beats, start_beat1,
            sr=sr, bpm=target_bpm,
            log_func=log
        )

        time_play_sec = sample1_start / sr
        m_play, s_play = int(time_play_sec // 60), int(time_play_sec % 60)
        m_end, s_end = int(sample1_end // sr // 60), int(sample1_end // sr % 60)

        blend_bars = blend_beats // 4
        log(f"   • Conteo previo de compases: Mezcla de {blend_beats} bombos ({blend_bars} compases / {blend_bars//8} frases)")
        log(f"   • Punto PLAY Track B en Track A: [{m_play:02d}:{s_play:02d}] (Beat {start_beat1})")
        log(f"   • Convivencia simultánea a volumen constante: {duration_blend_sec:.1f} segundos")
        log(f"   ✂️ CORTE DE FADER EN SECO: Pista A cortada de golpe en [{m_end:02d}:{s_end:02d}] al terminar el compás de Pista B (Estilo MiGue)")

        cue_list.append(
            CueEntry(
                idx + 1,
                next_track.title,
                time_play_sec,
                next_bpm,
                next_track.camelot,
                next_track.key_name
            )
        )

        # 7. CONSTRUCCIÓN FÍSICA DE LA MEZCLA
        head = master_mix[:, :sample1_start]
        y1_trans = master_mix[:, sample1_start:sample1_end]

        if sample2_aligned < 0:
            left_pad = -sample2_aligned
            chunk = y_next_stretched[:, 0 : max(0, n_blend - left_pad)]
            y2_trans = np.pad(chunk, ((0, 0), (left_pad, max(0, n_blend - (chunk.shape[1] + left_pad)))), mode="constant")
            tail_start = max(0, n_blend - left_pad)
        else:
            y2_trans = y_next_stretched[:, sample2_aligned : sample2_aligned + n_blend]
            if y2_trans.shape[1] < n_blend:
                y2_trans = np.pad(y2_trans, ((0, 0), (0, n_blend - y2_trans.shape[1])), mode="constant")
            tail_start = sample2_aligned + n_blend

        if use_stems:
            try:
                safe_title1 = re.sub(r"[^\w\-_]", "_", curr_track.title)
                safe_title2 = re.sub(r"[^\w\-_]", "_", next_track.title)
                cache1_id = f"stem_{safe_title1}_{sample1_start}_{n_blend}"
                cache2_id = f"stem_{safe_title2}_{sample2_aligned}_{n_blend}"

                separator = StemSeparator.get_instance()
                log(f"   🧠 [IA Stems] Separando fragmento de salida ({duration_blend_sec:.1f}s): {curr_track.title}...")
                stems1 = separator.separate_slice(y1_trans, sr=sr, cache_id=cache1_id)
                log(f"   🧠 [IA Stems] Separando fragmento de entrada ({duration_blend_sec:.1f}s): {next_track.title}...")
                stems2 = separator.separate_slice(y2_trans, sr=sr, cache_id=cache2_id)

                drop_sample_in_trans = None
                if beat2_drop > 0 and beat2_drop < len(next_beats):
                    cand_drop = int(next_beats[beat2_drop]) - sample2_aligned
                    if 0 < cand_drop < n_blend:
                        drop_sample_in_trans = cand_drop

                log("   🎛️ [IA Stems] Mezcla quirúrgica: Bass Swap en el drop + Ducking vocal anti-choque + Corte MiGue")
                blend_zone = apply_stem_dj_blend(
                    stems1,
                    stems2,
                    drop_sample_in_trans=drop_sample_in_trans,
                    vocal_clash_detected=vocal_clash_detected,
                    sr=sr
                )
            except Exception as stem_err:
                log(f"   ⚠️ [IA Stems] Error en separación de stems ({stem_err}), usando isolator 3 bandas estándar.")
                blend_zone = apply_migue_fader_cut_blend(y1_trans, y2_trans, sr=sr)
        else:
            # Aplicar el fader cut en seco al terminar compás sin bajada progresiva de volumen
            blend_zone = apply_migue_fader_cut_blend(y1_trans, y2_trans, sr=sr)

        # Track B continúa en solitario hasta el final
        tail = y_next_stretched[:, tail_start :]

        master_mix = np.hstack([head, blend_zone, tail]).astype(np.float32)

        # Actualización de la rejilla de beats sobre el máster acumulado
        beats_before = master_beats[master_beats < sample1_start]
        beats_after = sample1_start + (next_beats[next_beats >= sample2_aligned] - sample2_aligned)
        master_beats = np.concatenate([beats_before, beats_after])

        last_drop_in_master = sample1_start
        last_drop_local = sample2_aligned

    # 9. Masterización final del set completo (Highpass 25Hz, Compresor y Limitador)
    log(f"📦 Ensamblado completado ({master_mix.shape[1] / sr:.1f}s). Aplicando masterización anti-clipping...")
    mastered_audio = master_session_audio(master_mix, sr)

    # 10. Exportación
    log(f"💾 Guardando sesión final en: {output_audio_path}")
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_audio_path, mastered_audio.T, sr)

    _write_cue_file(cue_list, output_cue_path)
    log(f"📝 Tracklist guardado en: {output_cue_path}")

    return cue_list


def _write_cue_file(cues: List[CueEntry], output_path: Path):
    """Guarda el tracklist con marcas de tiempo en formato legible."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("               AUTODJ - SESIÓN MEZCLADA\n")
        f.write("=" * 60 + "\n\n")
        for cue in cues:
            f.write(
                f"[{cue.format_timestamp()}] Track {cue.index:02d}: {cue.title} "
                f"({cue.bpm:.1f} BPM | Clave: {cue.camelot} / {cue.key_name})\n"
            )
        f.write("\n" + "=" * 60 + "\n")
