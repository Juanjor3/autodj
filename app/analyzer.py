import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import scipy.signal
import librosa
import soundfile as sf


# Perfiles de Krumhansl-Schmuckler para detección de tonalidad mayor y menor
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Mapeo a notación Camelot Wheel
CAMELOT_MAJOR = {
    "B": "1B", "F#": "2B", "C#": "3B", "G#": "4B",
    "D#": "5B", "A#": "6B", "F": "7B", "C": "8B",
    "G": "9B", "D": "10B", "A": "11B", "E": "12B"
}
CAMELOT_MINOR = {
    "G#": "1A", "D#": "2A", "A#": "3A", "F": "4A",
    "C": "5A", "G": "6A", "D": "7A", "A": "8A",
    "E": "9A", "B": "10A", "F#": "11A", "C#": "12A"
}

PROFILES_DIR = Path("/app/profiles") if Path("/app/profiles").exists() else Path(__file__).parent / "profiles"
CACHE_FILE = PROFILES_DIR / "track_analysis_cache.json"


@dataclass
class TrackProfile:
    path: Path
    title: str
    sr: int
    duration: float
    bpm: float
    beat_samples: np.ndarray  # Índices de muestra de cada beat
    key_name: str             # ej: "A min"
    camelot: str              # ej: "8A"
    phrase_vocal: Optional[List[float]] = None   # Presencia vocal por frase de 32 beats (0.0 a 1.0)
    phrase_drums: Optional[List[float]] = None   # Presencia de percusión/bombo por frase
    phrase_bass: Optional[List[float]] = None    # Presencia de bajo/subgrave por frase
    phrase_melody: Optional[List[float]] = None  # Presencia melódica/armónica por frase
    phrase_tags: Optional[List[str]] = None      # Tags: INTRO_GROOVE, BREAKDOWN, DROP, OUTRO, etc.
    audio_stereo: Optional[np.ndarray] = None    # Shape: (2, num_samples), float32


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def estimate_key(chroma_mean: np.ndarray) -> Tuple[str, str]:
    """Estima la clave musical y el valor Camelot usando correlación espectral."""
    best_corr = -float("inf")
    best_key = "C maj"
    best_camelot = "8B"

    chroma_norm = (chroma_mean - np.mean(chroma_mean)) / (np.std(chroma_mean) + 1e-8)
    maj_norm = (MAJOR_PROFILE - np.mean(MAJOR_PROFILE)) / np.std(MAJOR_PROFILE)
    min_norm = (MINOR_PROFILE - np.mean(MINOR_PROFILE)) / np.std(MINOR_PROFILE)

    for i in range(12):
        rotated_chroma = np.roll(chroma_norm, -i)
        corr_maj = np.corrcoef(rotated_chroma, maj_norm)[0, 1]
        corr_min = np.corrcoef(rotated_chroma, min_norm)[0, 1]

        pitch = PITCH_NAMES[i]
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = f"{pitch} maj"
            best_camelot = CAMELOT_MAJOR.get(pitch, "8B")

        if corr_min > best_corr:
            best_corr = corr_min
            best_key = f"{pitch} min"
            best_camelot = CAMELOT_MINOR.get(pitch, "8A")

    return best_key, best_camelot


def load_stereo_audio(path: Path, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """Carga audio estéreo en float32 de forma ultra rápida."""
    try:
        data, sr = sf.read(path, dtype="float32")
        if sr != target_sr:
            data = librosa.resample(data.T, orig_sr=sr, target_sr=target_sr).T
            sr = target_sr

        if data.ndim == 1:
            audio_stereo = np.vstack([data, data]).astype(np.float32)
        else:
            audio_stereo = data.T[:2, :].astype(np.float32)
        return audio_stereo, sr
    except Exception:
        y, sr = librosa.load(path, sr=target_sr, mono=False)
        if y.ndim == 1:
            audio_stereo = np.vstack([y, y]).astype(np.float32)
        else:
            audio_stereo = y[:2, :].astype(np.float32)
        return audio_stereo, sr


def compute_phrase_vocal_activity(
    y_mono: np.ndarray,
    sr: int,
    beat_samples: np.ndarray,
    phrase_beats: int = 32
) -> List[float]:
    """
    Calcula la presencia vocal media por frase de 32 beats (0.0 a 1.0) de forma ultra rápida y de bajo consumo de RAM:
    1. Diezmado ligero a ~22 kHz sin duplicar memoria en disco
    2. Filtrado pasa-banda en formantes vocales (300 a 3400 Hz)
    3. Densidad espectral de energía y contraste armónico en formantes vocales (800 a 3000 Hz)
    """
    if len(beat_samples) < phrase_beats:
        return []

    # Diezmado por 2 mediante strided slicing (sin allocs gigantes)
    if sr >= 32000:
        y_v = y_mono[::2]
        sr_v = sr // 2
    else:
        y_v = y_mono
        sr_v = sr

    sos_voice = scipy.signal.butter(4, [300, 3400], btype="bandpass", fs=sr_v, output="sos")
    voice_band = scipy.signal.sosfilt(sos_voice, y_v)

    hop_length = 1024
    D = librosa.stft(voice_band, n_fft=1024, hop_length=hop_length)
    mag = np.abs(D)

    # Energía en la banda de formantes vocales clave (800 a 3000 Hz)
    freqs = librosa.fft_frequencies(sr=sr_v, n_fft=1024)
    vocal_mask = (freqs >= 800) & (freqs <= 3000)
    vocal_energy = np.mean(mag[vocal_mask, :], axis=0) if np.any(vocal_mask) else np.zeros(mag.shape[1])

    # Contraste espectral en bandas intermedias
    contrast = librosa.feature.spectral_contrast(S=mag, sr=sr_v, fmin=300, n_bands=4)
    vocal_contrast = contrast[2]

    h_p10, h_p90 = np.percentile(vocal_energy, 10), np.percentile(vocal_energy, 90)
    h_norm = (vocal_energy - h_p10) / max(1e-6, h_p90 - h_p10)
    h_norm = np.clip(h_norm, 0, 1)

    c_p10, c_p90 = np.percentile(vocal_contrast, 10), np.percentile(vocal_contrast, 90)
    c_norm = (vocal_contrast - c_p10) / max(1e-6, c_p90 - c_p10)
    c_norm = np.clip(c_norm, 0, 1)

    vocal_score = 0.65 * h_norm + 0.35 * c_norm
    times = librosa.frames_to_time(np.arange(len(vocal_score)), sr=sr_v, hop_length=hop_length)

    num_phrases = len(beat_samples) // phrase_beats
    phrase_vocal = []
    for p in range(num_phrases):
        t_start = beat_samples[p * phrase_beats] / sr
        t_end = beat_samples[(p + 1) * phrase_beats - 1] / sr
        mask = (times >= t_start) & (times <= t_end)
        mean_v = float(np.mean(vocal_score[mask])) if np.any(mask) else 0.0
        phrase_vocal.append(round(mean_v, 3))

    return phrase_vocal


def compute_phrase_stem_profiles(
    y_mono: np.ndarray,
    sr: int,
    beat_samples: np.ndarray,
    phrase_beats: int = 32
) -> Tuple[List[float], List[float], List[float], List[float], List[str]]:
    """
    Calcula de forma ultra rápida (en ~1s) la densidad energética de los 4 stems por frase de 32 beats:
    - Vocals: Formantes vocales y contraste espectral (0.0 a 1.0)
    - Drums: Pegada de bombo y ritmo en 50-130 Hz (0.0 a 1.0)
    - Bass: Frecuencias subgraves en < 140 Hz (0.0 a 1.0)
    - Melody: Contenido armónico en 400-3500 Hz (0.0 a 1.0)
    - Tags: INTRO_GROOVE, BREAKDOWN, DROP, OUTRO_GROOVE, VOCAL_MAIN, MAIN_GROOVE
    """
    phrase_vocal = compute_phrase_vocal_activity(y_mono, sr, beat_samples, phrase_beats=phrase_beats)
    num_phrases = len(beat_samples) // phrase_beats
    if num_phrases == 0:
        return [], [], [], [], []

    # 1. Banda de ritmo / bombo (50 a 130 Hz)
    sos_kick = scipy.signal.butter(4, [50, 130], btype="bandpass", fs=sr, output="sos")
    kick = scipy.signal.sosfilt(sos_kick, y_mono)

    # 2. Banda de bajo / subgrave (< 140 Hz)
    sos_bass = scipy.signal.butter(4, 140, btype="lowpass", fs=sr, output="sos")
    bass = scipy.signal.sosfilt(sos_bass, y_mono)

    # 3. Banda de melodía / sintetizadores (400 a 3500 Hz)
    sos_melody = scipy.signal.butter(4, [400, 3500], btype="bandpass", fs=sr, output="sos")
    melody = scipy.signal.sosfilt(sos_melody, y_mono)

    kick_rms, bass_rms, mel_rms = [], [], []
    for p in range(num_phrases):
        s0 = beat_samples[p * phrase_beats]
        s1 = beat_samples[(p + 1) * phrase_beats - 1]
        kick_rms.append(float(np.sqrt(np.mean(kick[s0:s1]**2))))
        bass_rms.append(float(np.sqrt(np.mean(bass[s0:s1]**2))))
        mel_rms.append(float(np.sqrt(np.mean(melody[s0:s1]**2))))

    def _norm(lst):
        if not lst:
            return []
        p10, p90 = np.percentile(lst, 10), np.percentile(lst, 90)
        denom = max(1e-6, p90 - p10)
        return [round(float(np.clip((x - p10) / denom, 0, 1)), 3) for x in lst]

    phrase_drums = _norm(kick_rms)
    phrase_bass = _norm(bass_rms)
    phrase_melody = _norm(mel_rms)

    # Generar etiquetas estructurales automáticas
    phrase_tags = []
    for p in range(num_phrases):
        d = phrase_drums[p]
        b = phrase_bass[p]
        m = phrase_melody[p]
        v = phrase_vocal[p] if p < len(phrase_vocal) else 0.0

        if p < 3 and d > 0.45 and v < 0.35:
            tag = "INTRO_GROOVE"
        elif d < 0.25 and (m > 0.35 or v > 0.35):
            tag = "BREAKDOWN"
        elif d > 0.60 and b > 0.60 and p > 0 and phrase_tags[-1] in ["BREAKDOWN", "INTRO_GROOVE"]:
            tag = "DROP"
        elif p >= num_phrases - 3 and v < 0.35 and d > 0.40:
            tag = "OUTRO_GROOVE"
        elif v > 0.45 and d > 0.45:
            tag = "VOCAL_MAIN"
        else:
            tag = "MAIN_GROOVE"
        phrase_tags.append(tag)

    return phrase_vocal, phrase_drums, phrase_bass, phrase_melody, phrase_tags


def analyze_audio_file(
    file_path: Path,
    target_sr: int = 44100,
    load_audio: bool = True
) -> TrackProfile:
    """
    Carga y analiza una pista de audio con caché persistente en disco.
    Optimizado para mantener el consumo de RAM por debajo de 700 MB incluso en archivos FLAC extensos.
    """
    path = Path(file_path)
    stat = path.stat()
    cache_key = f"{path.name}_{stat.st_size}_{stat.st_mtime}_v4"
    cache = _load_cache()

    if cache_key in cache:
        item = cache[cache_key]
        duration = float(item["duration"])
        bpm = float(item["bpm"])
        key_name = item["key_name"]
        camelot = item["camelot"]
        beat_samples = np.array(item["beat_samples"], dtype=int)

        phrase_vocal = item.get("phrase_vocal", None)
        phrase_drums = item.get("phrase_drums", None)
        phrase_bass = item.get("phrase_bass", None)
        phrase_melody = item.get("phrase_melody", None)
        phrase_tags = item.get("phrase_tags", None)

        audio_stereo = None
        if load_audio:
            audio_stereo, _ = load_stereo_audio(path, target_sr=target_sr)

        return TrackProfile(
            path=path,
            title=path.stem,
            sr=target_sr,
            duration=duration,
            bpm=bpm,
            beat_samples=beat_samples,
            key_name=key_name,
            camelot=camelot,
            phrase_vocal=phrase_vocal,
            phrase_drums=phrase_drums,
            phrase_bass=phrase_bass,
            phrase_melody=phrase_melody,
            phrase_tags=phrase_tags,
            audio_stereo=audio_stereo
        )

    # Si no está en caché, analizar de forma optimizada
    audio_stereo, sr = load_stereo_audio(path, target_sr=target_sr)
    y_mono = librosa.to_mono(audio_stereo)
    duration = float(audio_stereo.shape[1] / sr)

    # 1. Detección de tempo y beatgrid:
    # Autocorrelación sobre el lecho rítmico y detección de fase de bombo por interferencia constructiva (Folded Envelope)
    n_total = len(y_mono)
    search_start = min(sr * 15, int(n_total * 0.15))
    search_end = min(n_total, search_start + sr * 60)
    groove_mono = y_mono[search_start:search_end]
    if len(groove_mono) < sr * 10:
        groove_mono = y_mono
        search_start = 0

    sos_kick = scipy.signal.butter(4, [45, 125], btype='bandpass', fs=sr, output='sos')
    kick_filt = scipy.signal.sosfilt(sos_kick, groove_mono)
    groove_env = np.abs(scipy.signal.hilbert(kick_filt))

    sample_len = min(len(groove_env), sr * 30)
    acorr = scipy.signal.correlate(groove_env[:sample_len], groove_env[:sample_len], mode='full')
    center = sample_len - 1
    lags = acorr[center:]

    min_lag = int(sr * (4.0 * 60.0 / 165.0))
    max_lag = int(sr * (4.0 * 60.0 / 105.0))
    p_bar = np.argmax(lags[min_lag:max_lag]) + min_lag

    # Refinamiento a 16 bombos (4 compases) para una precisión de tempo perfecta (< 0.01 BPM)
    p_4bar_center = p_bar * 4
    w_search = sr // 10
    s_start = max(0, p_4bar_center - w_search)
    s_end = min(len(lags), p_4bar_center + w_search)
    p_4bar = np.argmax(lags[s_start:s_end]) + s_start

    spb = float(p_4bar) / 16.0
    bpm = float((60.0 * sr) / spb)

    # Detección de fase de bombo por plegado de envolvente (Folded Envelope Phase Locking):
    # La interferencia constructiva de todos los bombos en el groove localiza la fase de bombo
    # a nivel de muestra sin ser engañada por silencios iniciales o intros lentas.
    nbins = int(round(spb))
    phase_acc = np.zeros(nbins, dtype=np.float64)
    num_cycles = len(groove_env) // nbins
    for c in range(num_cycles):
        phase_acc += groove_env[c * nbins : (c + 1) * nbins]

    best_phase_in_groove = int(np.argmax(phase_acc))
    phase_0 = (search_start + best_phase_in_groove) % spb

    num_beats = int((len(y_mono) - phase_0) // spb)
    beat_samples = np.round(phase_0 + np.arange(num_beats) * spb).astype(int)

    # 2. Clave Camelot: extraída del cuerpo armónico (ventana de 90s)
    dur_sec = len(y_mono) / sr
    if dur_sec > 100:
        start_sec = max(30.0, dur_sec * 0.20)
        end_sec = min(dur_sec - 10.0, start_sec + 90.0)
        y_key = y_mono[int(start_sec * sr):int(end_sec * sr)]
    else:
        y_key = y_mono

    try:
        chroma = librosa.feature.chroma_stft(y=y_key, sr=sr, n_fft=4096, hop_length=1024)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=y_key, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key_name, camelot = estimate_key(chroma_mean)

    # 3. Detección multi-stem y frases estructurales (voces, bombos, bajos, melodías y tags)
    phrase_vocal, phrase_drums, phrase_bass, phrase_melody, phrase_tags = compute_phrase_stem_profiles(
        y_mono, sr, beat_samples, phrase_beats=32
    )

    # Guardar en caché persistente en disco
    cache[cache_key] = {
        "duration": duration,
        "bpm": bpm,
        "key_name": key_name,
        "camelot": camelot,
        "phrase_vocal": phrase_vocal,
        "phrase_drums": phrase_drums,
        "phrase_bass": phrase_bass,
        "phrase_melody": phrase_melody,
        "phrase_tags": phrase_tags,
        "beat_samples": beat_samples.tolist()
    }
    _save_cache(cache)

    return TrackProfile(
        path=path,
        title=path.stem,
        sr=sr,
        duration=duration,
        bpm=bpm,
        beat_samples=beat_samples,
        key_name=key_name,
        camelot=camelot,
        phrase_vocal=phrase_vocal,
        phrase_drums=phrase_drums,
        phrase_bass=phrase_bass,
        phrase_melody=phrase_melody,
        phrase_tags=phrase_tags,
        audio_stereo=audio_stereo if load_audio else None
    )
