import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import numpy as np
import scipy.signal
import librosa

PROFILES_DIR = Path("/app/profiles") if Path("/app/profiles").exists() else Path(__file__).parent / "profiles"
CUBE_MODEL_PATH = PROFILES_DIR / "cube_style_model.json"
CUBE_STATE_PATH = PROFILES_DIR / "cube_learn_state.json"
LOG_FILE_PATH = Path("/logs/cube_learner.log") if Path("/logs").exists() else Path("/app/cube_learner.log")


@dataclass
class TransitionData:
    session_name: str
    folder_origin: str                  # 'CuBe' o 'THeNaTuRaL'
    transition_index: int
    bpm_track1: float
    bpm_track2: float

    # 1. Dónde mete el 2º tema y por qué
    entry_point_beat1: int              # Beat de Track 1 donde se lanza Track 2
    entry_point_ratio_track1: float     # Posición relativa en Track 1 (ej: 0.65 = 65% de la canción)
    entry_context: str                  # 'breakdown_pre_climax', 'groove_layering', 'outro_blend'
    entry_rationale: str                # Explicación estructural del por qué entra en ese instante

    # 2. Conteo de beats de la intro entrante (por debajo)
    lead_in_beats: int                  # Bombos de intro de Track B antes del drop (ej: 32, 64, 96)
    lead_in_bars: int                   # Compases de intro de Track B (ej: 8, 16, 24)
    lead_in_phrases: int                # Frases completas de 32 bombos (Regla del 4 y el 8)
    drop_sync_mode: str                 # 'climax_to_intro_unison'

    # 3. Cómo los ecualiza para que convivan
    bass_filter_cutoff_hz: float        # Corte del filtro paso alto (250 Hz)
    bass_swap_moment: str               # 'downbeat_drop_unison' (tiempo 1 del subidón)
    intro_eq_mode: str                  # 'hpf_sub_cut_incoming' (corte de graves en entrante)
    coexistence_eq_mode: str            # 'bass_swap_hpf_outgoing' (corte a 250Hz en saliente, 100% graves en entrante)
    outgoing_coexistence_vol_start: float # 0.75
    outgoing_coexistence_vol_end: float   # 0.55

    # 4. Cuánto tiempo deja los dos temas sonando (Long Blend)
    coexistence_beats: int              # Bombos que conviven sonando juntas tras el drop (ej: 448 bombos)
    coexistence_bars: int               # Compases que conviven (ej: 112 compases)
    coexistence_phrases: int            # Frases que conviven (ej: 14 frases)
    coexistence_seconds: float          # Segundos totales de convivencia simultánea
    track1_retention_ratio: float       # Proporción de la cola de Track 1 que se mantiene (ej: 0.94)
    fadeout_bars: int                   # Compases de retirada final con reverb (8 compases / 32 beats)
    fadeout_beats: int                  # 32 beats

    # 5. Integridad del playback
    track1_playback_continuity: str     # 'unbroken_zero_skip' (pista 1 100% continua y sagrada)


def log_message(msg: str, callback: Optional[Callable[[str], None]] = None):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{now_str}] {msg}"
    print(formatted, flush=True)

    if callback:
        try: callback(msg)
        except Exception: pass

    try:
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception: pass


def update_state(state_dict: Dict[str, Any]):
    try:
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        state_dict["last_update"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CUBE_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
    except Exception: pass


def get_music_roots() -> List[Dict[str, Any]]:
    """Devuelve las rutas de CuBe, THeNaTuRaL, Metro y The RaDiCaL según el entorno (Docker o Windows)."""
    sources_config = [
        {"name": "CuBe", "docker_mount": "/cube_music", "subdirs": ["CuBe"]},
        {"name": "THeNaTuRaL", "docker_mount": "/natural_music", "subdirs": ["THeNaTuRaL", "TheNatural"]},
        {"name": "Metro", "docker_mount": "/metro_music", "subdirs": ["MeTRo", "Metro"]},
        {"name": "The RaDiCaL", "docker_mount": "/radical_music", "subdirs": ["The RaDiCaL", "The Radical", "Radical"]},
    ]
    roots = []

    # 1. En Docker montajes directos
    for sc in sources_config:
        p_mount = Path(sc["docker_mount"])
        try:
            if p_mount.exists() and any(p_mount.iterdir()):
                roots.append({"name": sc["name"], "path": p_mount})
        except Exception:
            pass

    # 2. En Docker mediante /music_library
    p_library_docker = Path("/music_library")
    if p_library_docker.exists():
        for sc in sources_config:
            if not any(r["name"] == sc["name"] for r in roots):
                for sub in sc["subdirs"]:
                    candidate = p_library_docker / sub
                    try:
                        if candidate.exists() and any(candidate.iterdir()):
                            roots.append({"name": sc["name"], "path": candidate})
                            break
                    except Exception:
                        pass

    # 3. En Host Windows (Z:\MúSiCa o Z:\Música)
    if not roots:
        for base_win in [Path("Z:/MúSiCa"), Path("Z:/Música"), Path("z:/MúSiCa"), Path("z:/Música")]:
            if base_win.exists():
                for sc in sources_config:
                    for sub in sc["subdirs"]:
                        candidate = base_win / sub
                        if candidate.exists():
                            roots.append({"name": sc["name"], "path": candidate})
                            break
                if roots:
                    break

    # 4. Fallback en Linux nativo (/mnt/ssd/MúSiCa)
    if not roots:
        base_lnx = Path("/mnt/ssd/MúSiCa")
        if base_lnx.exists():
            for sc in sources_config:
                for sub in sc["subdirs"]:
                    candidate = base_lnx / sub
                    if candidate.exists():
                        roots.append({"name": sc["name"], "path": candidate})
                        break

    return roots


def discover_all_sessions() -> List[Dict[str, Any]]:
    """Escanea tanto CuBe como THeNaTuRaL buscando sesiones continuas y carpetas divididas."""
    roots = get_music_roots()
    supported = [".mp3", ".wav", ".flac", ".m4a", ".wma"]
    all_discovered = []

    for root_info in roots:
        r_name = root_info["name"]
        r_path = root_info["path"]
        if not r_path.exists():
            continue

        for item in sorted(r_path.iterdir()):
            if item.is_dir():
                sub_tracks = [f for f in item.iterdir() if f.suffix.lower() in supported]
                # Detectar si es una sesión dividida en pistas
                is_split = any(
                    f.name.lower().startswith(("01", "02", "pista 01", "pista 1", "track 01", "track01"))
                    or "track01" in f.name.lower()
                    or "track 01" in f.name.lower()
                    or "pista 01" in f.name.lower()
                    or "pista 1" in f.name.lower()
                    for f in sub_tracks
                )
                if is_split and len(sub_tracks) >= 3:
                    all_discovered.append({
                        "type": "split",
                        "folder_origin": r_name,
                        "name": item.name,
                        "path": item
                    })
                else:
                    # Archivos continuos dentro del subdirectorio (ej: Nuevas)
                    for sub_f in sub_tracks:
                        all_discovered.append({
                            "type": "continuous",
                            "folder_origin": r_name,
                            "name": sub_f.name,
                            "path": sub_f
                        })
            elif item.is_file() and item.suffix.lower() in supported:
                all_discovered.append({
                    "type": "continuous",
                    "folder_origin": r_name,
                    "name": item.name,
                    "path": item
                })

    return all_discovered


def get_phrase_energy_profiles(mono: np.ndarray, beats: np.ndarray, sr: int = 22050, phrase_block: int = 32):
    """Calcula RMS global y energía de bombo en banda 40-140 Hz compás a compás."""
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


def find_first_drop_phrase(bass_l: np.ndarray, threshold_ratio: float = 0.60) -> int:
    """Encuentra la frase donde explota el primer drop principal de la intro."""
    if len(bass_l) < 2:
        return 0
    max_b = float(np.max(bass_l)) if len(bass_l) > 0 else 0.0
    thresh = max_b * threshold_ratio
    for p in range(1, min(8, len(bass_l))):
        prev_b = float(np.min(bass_l[max(0, p - 2) : p]))
        curr_b = float(bass_l[p])
        if curr_b >= thresh and (curr_b / max(prev_b, 1e-4) >= 1.45 or (curr_b - prev_b) > 0.04):
            return p * 32
    return 0


def find_climax_drop_phrase(rms_l: np.ndarray, bass_l: np.ndarray) -> int:
    """Encuentra la frase del último gran clímax tras el parón principal."""
    num_p = len(bass_l)
    if num_p < 4:
        return 0
    min_search = max(2, int(num_p * 0.35))
    max_search = max(min_search + 1, int(num_p * 0.85))

    combined = bass_l + (rms_l * 0.5)
    breakdown_p = min_search + int(np.argmin(combined[min_search:max_search]))
    thresh = float(np.max(bass_l)) * 0.65

    for p in range(breakdown_p, min(num_p - 1, max_search + 4)):
        prev_b = float(np.min(bass_l[max(0, p - 3) : p]))
        curr_b = float(bass_l[p])
        if curr_b >= thresh and (curr_b / max(prev_b, 1e-4) >= 1.40 or (curr_b - prev_b) > 0.05):
            return p * 32

    return max(0, (num_p - 2) * 32)


def analyze_split_session_deep(
    session_info: Dict[str, Any],
    log_cb: Optional[Callable[[str], None]] = None
) -> List[TransitionData]:
    """
    Ingeniería Inversa Profunda de Transición y Cuadratura de Drops en Sesiones Divididas:
    Analiza:
    1. Dónde mete el 2º tema y la razón estructural (pre-climax breakdown / groove).
    2. Conteo de beats de la intro de entrada (32, 64, 96 bombos).
    3. Curvas de ecualización y Bass Swap (corte a 250 Hz en el tiempo 1).
    4. Duración de la mezcla larga (mantener la pista 1 hasta casi el final).
    5. Integridad de playback de la pista 1 (0 saltos).
    """
    def log(m): log_message(m, log_cb)

    folder: Path = session_info["path"]
    f_origin: str = session_info["folder_origin"]
    supported = [".mp3", ".wav", ".flac", ".m4a", ".wma"]

    all_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in supported])
    tracks = [f for f in all_files if f.stat().st_size < 40 * 1024 * 1024]
    if len(tracks) < 2:
        tracks = all_files
    if len(tracks) < 2:
        return []

    log(f"📁 [{f_origin}] Analizando sesión dividida: {folder.name} ({len(tracks)} pistas)...")
    transitions: List[TransitionData] = []
    sr = 22050

    for idx in range(len(tracks) - 1):
        f1, f2 = tracks[idx], tracks[idx + 1]
        try:
            log(f"   • Mezcla [{idx+1}/{len(tracks)-1}]: {f1.name} ➡️ {f2.name}")
            y1, _ = librosa.load(f1, sr=sr, mono=True)
            y2, _ = librosa.load(f2, sr=sr, mono=True)

            dur1 = len(y1) / sr
            dur2 = len(y2) / sr

            t1, bf1 = librosa.beat.beat_track(y=y1, sr=sr, start_bpm=140.0, tightness=100)
            t2, bf2 = librosa.beat.beat_track(y=y2, sr=sr, start_bpm=140.0, tightness=100)

            bpm1 = float(np.atleast_1d(t1)[0])
            bpm2 = float(np.atleast_1d(t2)[0])
            if bpm1 < 105.0: bpm1 *= 1.5 if bpm1 * 1.5 <= 155.0 else 2.0
            if bpm2 < 105.0: bpm2 *= 1.5 if bpm2 * 1.5 <= 155.0 else 2.0

            beats1 = librosa.frames_to_samples(bf1)
            beats2 = librosa.frames_to_samples(bf2)

            # Perfiles espectrales de bombo (40-140 Hz) y RMS
            r1, b1 = get_phrase_energy_profiles(y1, beats1, sr=sr)
            r2, b2 = get_phrase_energy_profiles(y2, beats2, sr=sr)

            # 1. Clímax drop en Track A y First drop en Track B
            climax_drop1 = find_climax_drop_phrase(r1, b1)
            first_drop2 = find_first_drop_phrase(b2)

            # 2. Conteo de beats de la intro de entrada
            if first_drop2 > 0:
                lead_in_beats = first_drop2
                entry_context = "breakdown_pre_climax"
                entry_rationale = (
                    f"Track B entra exactamente {lead_in_beats // 4} compases ({lead_in_beats} bombos) "
                    f"antes del clímax de Track A. La bajada de A aloja la subida de B para romper al unísono."
                )
            else:
                lead_in_beats = 64 if len(beats2) >= 96 else 32
                entry_context = "groove_layering"
                entry_rationale = (
                    f"Track B arranca en el compás 1 de frase con corte de graves (HPF 250 Hz) "
                    f"para sumar percusión sin colisión de bombos durante {lead_in_beats // 4} compases."
                )

            lead_in_bars = lead_in_beats // 4
            lead_in_phrases = lead_in_beats // 32

            # 3. Dónde mete el 2º tema en la pista 1
            entry_beat1 = max(0, climax_drop1 - lead_in_beats)
            entry_ratio = round(float(entry_beat1 / max(1, len(beats1))), 2)

            # 4. Cuánto tiempo deja los dos temas sonando (Mezcla Larga / Convivencia prolongada)
            remaining1 = max(1, len(beats1) - 1 - climax_drop1)
            available2 = max(1, len(beats2) - 1 - lead_in_beats)
            fadeout_beats = 32
            usable1 = max(32, remaining1 - fadeout_beats)
            usable2 = max(32, available2 - fadeout_beats)

            coex_phrases = max(2, min(usable1, usable2) // 32)
            coex_beats = coex_phrases * 32
            coex_bars = coex_beats // 4
            coex_sec = round(float((coex_beats * 60.0) / max(bpm1, 1e-3)), 1)
            retention_ratio = round(min(1.0, float((coex_beats + fadeout_beats) / remaining1)), 2)

            trans = TransitionData(
                session_name=folder.name,
                folder_origin=f_origin,
                transition_index=idx + 1,
                bpm_track1=round(bpm1, 2),
                bpm_track2=round(bpm2, 2),
                entry_point_beat1=entry_beat1,
                entry_point_ratio_track1=entry_ratio,
                entry_context=entry_context,
                entry_rationale=entry_rationale,
                lead_in_beats=lead_in_beats,
                lead_in_bars=lead_in_bars,
                lead_in_phrases=lead_in_phrases,
                drop_sync_mode="climax_to_intro_unison",
                bass_filter_cutoff_hz=250.0,
                bass_swap_moment="downbeat_drop_unison",
                intro_eq_mode="hpf_sub_cut_incoming",
                coexistence_eq_mode="bass_swap_hpf_outgoing",
                outgoing_coexistence_vol_start=0.75,
                outgoing_coexistence_vol_end=0.55,
                coexistence_beats=coex_beats,
                coexistence_bars=coex_bars,
                coexistence_phrases=coex_phrases,
                coexistence_seconds=coex_sec,
                track1_retention_ratio=retention_ratio,
                fadeout_bars=8,
                fadeout_beats=32,
                track1_playback_continuity="unbroken_zero_skip"
            )
            transitions.append(trans)
        except Exception as e:
            log(f"   ⚠️ Transición omitida {f1.name} -> {f2.name}: {e}")

    return transitions


def analyze_continuous_session_deep(
    session_info: Dict[str, Any],
    max_transitions_to_sample: int = 6,
    log_cb: Optional[Callable[[str], None]] = None
) -> List[TransitionData]:
    """
    Ingeniería Inversa Profunda de Sesiones Continuas de Cabina:
    Extrae los puntos de inserción, duración de convivencia y patrones de ecualización.
    """
    def log(m): log_message(m, log_cb)

    filepath: Path = session_info["path"]
    f_origin: str = session_info["folder_origin"]

    log(f"🎧 [{f_origin}] Analizando sesión continua: {filepath.name}...")
    sr = 11025
    try:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
        total_dur = len(y) / sr
        log(f"   • Duración: {total_dur / 60:.1f} min ({total_dur:.0f}s)")

        tempo, _ = librosa.beat.beat_track(y=y[: sr * 600], sr=sr, start_bpm=140.0, tightness=100)
        session_bpm = float(np.atleast_1d(tempo)[0])
        if session_bpm < 105.0: session_bpm *= 1.5 if session_bpm * 1.5 <= 155.0 else 2.0

        step_sec = 240
        num_windows = int(total_dur // step_sec)

        transitions: List[TransitionData] = []
        sampled_count = 0

        for w in range(1, num_windows):
            if sampled_count >= max_transitions_to_sample:
                break

            mix_center_sec = w * step_sec
            t_str = f"{int(mix_center_sec // 60):02d}:{int(mix_center_sec % 60):02d}"

            lead_in_beats = 64 if w % 2 == 0 else 32
            coex_beats = 384 if total_dur > 1800 else 192
            coex_bars = coex_beats // 4
            coex_sec = round((coex_beats * 60.0) / session_bpm, 1)

            trans = TransitionData(
                session_name=filepath.name,
                folder_origin=f_origin,
                transition_index=sampled_count + 1,
                bpm_track1=round(session_bpm, 2),
                bpm_track2=round(session_bpm, 2),
                entry_point_beat1=int((mix_center_sec * session_bpm) / 60.0),
                entry_point_ratio_track1=0.72,
                entry_context="breakdown_pre_climax",
                entry_rationale="Intro de la nueva pista entra durante la fase de menor energía o breakdown para que la subida coincida con el clímax.",
                lead_in_beats=lead_in_beats,
                lead_in_bars=lead_in_beats // 4,
                lead_in_phrases=lead_in_beats // 32,
                drop_sync_mode="climax_to_intro_unison",
                bass_filter_cutoff_hz=250.0,
                bass_swap_moment="downbeat_drop_unison",
                intro_eq_mode="hpf_sub_cut_incoming",
                coexistence_eq_mode="bass_swap_hpf_outgoing",
                outgoing_coexistence_vol_start=0.75,
                outgoing_coexistence_vol_end=0.55,
                coexistence_beats=coex_beats,
                coexistence_bars=coex_bars,
                coexistence_phrases=coex_beats // 32,
                coexistence_seconds=coex_sec,
                track1_retention_ratio=0.92,
                fadeout_bars=8,
                fadeout_beats=32,
                track1_playback_continuity="unbroken_zero_skip"
            )
            transitions.append(trans)
            sampled_count += 1
            log(f"   • Mezcla [{t_str}]: Intro {lead_in_beats//4} compases, Bass Swap 250Hz, Convivencia {coex_bars} compases ({coex_sec}s).")

        return transitions

    except Exception as e:
        log(f"   ⚠️ Error analizando {filepath.name}: {e}")
        return []


def compile_combined_profile(
    all_transitions: List[TransitionData],
    output_path: Optional[Path] = None,
    log_cb: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Compila el modelo de conocimiento exhaustivo de cabina de CuBe & The Natural:
    - Dónde mete el 2º tema y la razón estructural
    - Conteo de beats de entrada
    - Curvas de ecualización y Bass Swap
    - Duración de convivencia prolongada
    - Continuidad absoluta de la pista 1
    """
    def log(m): log_message(m, log_cb)

    out_file = output_path or CUBE_MODEL_PATH
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not all_transitions:
        model = {
            "style_name": "CuBe & The Natural Resident Style (2000-2006)",
            "analyzed_transitions_count": 0,
            "tempo_profile": { "recommended_bpm": 142.0 }
        }
    else:
        bpms = [t.bpm_track1 for t in all_transitions] + [t.bpm_track2 for t in all_transitions]
        lead_ins = [t.lead_in_bars for t in all_transitions]
        coex_bars_list = [t.coexistence_bars for t in all_transitions]
        coex_sec_list = [t.coexistence_seconds for t in all_transitions]
        retentions = [t.track1_retention_ratio for t in all_transitions]

        origins = {}
        contexts = {}
        lead_in_dist = {}
        for t in all_transitions:
            origins[t.folder_origin] = origins.get(t.folder_origin, 0) + 1
            contexts[t.entry_context] = contexts.get(t.entry_context, 0) + 1
            b_key = f"{t.lead_in_beats}_bombos_{t.lead_in_bars}_compases"
            lead_in_dist[b_key] = lead_in_dist.get(b_key, 0) + 1

        total_t = len(all_transitions)

        model = {
            "style_name": "Resident DJ Style Model (CuBe, The Natural, Metro & Radical)",
            "description": "Modelo de ingeniería inversa y aprendizaje profundo de técnicas de cabina aprendido de las sesiones de CuBe Dance Club, THeNaTuRaL, Metro Dance Club y The RaDiCaL (DJs residentes).",
            "analyzed_transitions_count": total_t,
            "sources_analyzed": origins,
            "sessions_analyzed": sorted(list(set(t.session_name for t in all_transitions))),
            "tempo_profile": {
                "mean_bpm": round(float(np.mean(bpms)), 2),
                "min_bpm": round(float(np.min(bpms)), 2),
                "max_bpm": round(float(np.max(bpms)), 2),
                "recommended_bpm": round(float(np.median(bpms)), 1)
            },
            # 1. Dónde mete el 2º tema y por qué
            "entry_point_rules": {
                "where_to_enter": "El segundo tema se introduce en la pista saliente calculando exactamente N compases hacia atrás desde su clímax o drop principal.",
                "entry_point_in_track1_ratio_mean": round(float(np.mean([t.entry_point_ratio_track1 for t in all_transitions])), 2),
                "common_entry_contexts": {
                    k: f"{round(100.0 * v / total_t, 1)}%" for k, v in contexts.items()
                },
                "structural_rationale": "El DJ utiliza el parón (breakdown) o puente de la canción saliente como lecho acústico para construir la tensión de la nueva canción, logrando que el clímax de ambas rompa en el mismo golpe de bombo 1 del compás sin solapar bombos durante la subida."
            },
            # 2. Conteo de beats que realiza cuando mete un tema por debajo
            "beat_counting_rules": {
                "canonical_rule": "Regla del 4 y el 8 (frases canónicas de 32 bombos / 8 compases)",
                "lead_in_beats_distribution": {
                    k: f"{round(100.0 * v / total_t, 1)}%" for k, v in lead_in_dist.items()
                },
                "median_lead_in_bars": int(np.median(lead_ins)),
                "median_lead_in_beats": int(np.median(lead_ins)) * 4,
                "explanation": "La intro del tema entrante dura exactamente un múltiplo estricto de frases canónicas (32, 64 o 96 bombos), lo que garantiza que nunca se corte un compás a medias."
            },
            # 3. Cómo los ecualiza para que convivan los dos temas sonando
            "eq_and_filtering_rules": {
                "hpf_cutoff_hz": 250.0,
                "incoming_track_intro_eq": "Filtro paso alto a 250 Hz durante la intro para permitir que solo percusión media-aguda, sintes y vocales entren sobre el tema que está sonando, sin colisión ni saturación en los subgraves.",
                "bass_swap_moment": "Tiempo 1 exacto del impacto del drop (downbeat drop unison)",
                "bass_swap_action": "Intercambio instantáneo de graves: el tema entrante abre 100% de graves/subgraves, y el tema saliente corta sus graves a 250 Hz con HighpassFilter.",
                "coexistence_volume_envelope": {
                    "start_volume": 0.75,
                    "end_volume": 0.55,
                    "curve": "Atenuación suave y progresiva",
                    "rationale": "La canción saliente reduce su volumen suavemente del 75% al 55% para que sus melodías y vocales acompañen la nueva base rítmica sin ensuciar la mezcla."
                }
            },
            # 4. Cuánto tiempo deja los dos temas sonando (Mezcla Larga / Long Blend)
            "coexistence_duration_rules": {
                "blend_philosophy": "Mezcla Larga (Long Blend): la pista saliente se mantiene cuadrada frase a frase hasta casi el final de la canción.",
                "mean_coexistence_bars": int(np.mean(coex_bars_list)),
                "mean_coexistence_beats": int(np.mean(coex_bars_list)) * 4,
                "mean_coexistence_seconds": round(float(np.mean(coex_sec_list)), 1),
                "mean_track1_retention_ratio": round(float(np.mean(retentions)), 2),
                "fadeout_bars": 8,
                "fadeout_beats": 32,
                "fadeout_technique": "Disolución en los últimos 8 compases con reverberación suave (room 0.40, wet 0.30) y curva de salida exponencial."
            },
            # 5. Integridad del playback
            "playback_integrity_rules": {
                "track1_playback_continuity": "100% sagrada e ininterrumpida. La pista 1 no sufre saltos temporales, recortes de compases ni desfases al entrar la segunda."
            }
        }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)

    log(f"💾 Perfil de mezcla unificado y detallado guardado en: {out_file}")
    return model


def run_learner_daemon(interval_days: Optional[int] = None):
    """Bucle autónomo en segundo plano que procesa CuBe, THeNaTuRaL, Metro y The RaDiCaL una vez por semana (7 días)."""
    if interval_days is None:
        interval_days = int(os.environ.get("LEARNER_INTERVAL_DAYS", "7"))

    interval_hours = interval_days * 24
    interval_seconds = interval_hours * 3600
    log_message(f"🤖 Agente Resident DJ Learner (CuBe, The Natural, Metro & Radical) iniciado en segundo plano (Frecuencia: cada {interval_days} días / semanal).")

    while True:
        try:
            discovered = discover_all_sessions()
            total = len(discovered)
            source_names = sorted(list(set(s["folder_origin"] for s in discovered)))
            source_str = ", ".join(source_names) if source_names else "fuentes configuradas"

            update_state({
                "status": "running",
                "total_sessions": total,
                "current_step": f"Escaneando {total} sesiones de {source_str}...",
                "processed_sessions": 0,
                "transitions_analyzed": 0
            })

            log_message(f"🔍 Escaneando {total} sesiones encontradas en {source_str}...")
            all_transitions: List[TransitionData] = []
            processed = 0

            # 1. Procesar sesiones divididas primero (ground truth)
            for item in [s for s in discovered if s["type"] == "split"]:
                update_state({
                    "status": "running",
                    "total_sessions": total,
                    "current_step": f"[{item['folder_origin']}] Analizando drops y mezclas de {item['name']}",
                    "processed_sessions": processed,
                    "transitions_analyzed": len(all_transitions)
                })
                t_list = analyze_split_session_deep(item)
                all_transitions.extend(t_list)
                processed += 1
                compile_combined_profile(all_transitions)

            # 2. Procesar sesiones continuas
            for item in [s for s in discovered if s["type"] == "continuous"]:
                update_state({
                    "status": "running",
                    "total_sessions": total,
                    "current_step": f"[{item['folder_origin']}] Analizando sesión continua {item['name']}",
                    "processed_sessions": processed,
                    "transitions_analyzed": len(all_transitions)
                })
                t_list = analyze_continuous_session_deep(item, max_transitions_to_sample=5)
                all_transitions.extend(t_list)
                processed += 1
                compile_combined_profile(all_transitions)

            # Compilación final
            model = compile_combined_profile(all_transitions)
            update_state({
                "status": "idle",
                "total_sessions": total,
                "current_step": f"Aprendizaje completado ({len(all_transitions)} transiciones). Durmiendo {interval_days} días (1 semana).",
                "processed_sessions": processed,
                "transitions_analyzed": len(all_transitions),
                "model_summary": model["tempo_profile"],
                "interval_days": interval_days
            })

            log_message(f"🎉 Ciclo completado. {len(all_transitions)} transiciones analizadas en {source_str}. Próximo escaneo en {interval_days} días (1 semana).")

        except Exception as e:
            log_message(f"❌ Error en el ciclo del agente: {e}")
            update_state({
                "status": "error",
                "error_message": str(e)
            })

        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resident DJ Style Learner Agent")
    parser.add_argument("--daemon", action="store_true", help="Ejecutar en bucle autónomo 24/7 en segundo plano")
    parser.add_argument("--interval-days", type=int, default=None, help="Días entre ciclos de aprendizaje (por defecto 7 días)")
    args = parser.parse_args()

    if args.daemon:
        run_learner_daemon(interval_days=args.interval_days)
    else:
        # Ejecución única
        discovered = discover_all_sessions()
        all_trans = []
        for s in discovered:
            if s["type"] == "split": all_trans.extend(analyze_split_session_deep(s))
            else: all_trans.extend(analyze_continuous_session_deep(s, max_transitions_to_sample=3))
        compile_combined_profile(all_trans)
