import numpy as np
import soundfile as sf
import scipy.signal
import librosa
from pathlib import Path
from analyzer import load_stereo_audio

p1 = Path('/app/input/136 8a Icehouse - Hey Little Girl (X-Tended Edit).flac')
p2 = Path('/app/input/Harry_choo-choo_romero-i_want_out_(i_cant_believe).flac')

sr = 22050  # 22 kHz es ideal y super rápido para análisis de voces y MFCCs
y1, _ = load_stereo_audio(p1, sr)
y2, _ = load_stereo_audio(p2, sr)

m1 = 0.5 * (y1[0] + y1[1])
m2 = 0.5 * (y2[0] + y2[1])

def detect_vocal_activity_profile(mono, sr, hop_length=512):
    """
    Detector espectral de presencia vocal basado en:
    1. Separación armónica (HPSS) en rango vocal (300 - 3200 Hz)
    2. Coeficientes MFCC (formantes vocales humanos F1/F2)
    3. Contraste espectral en bandas vocales (1000 - 2500 Hz)
    """
    # Filtro paso-banda vocal (300 a 3400 Hz)
    sos_voice = scipy.signal.butter(4, [300, 3400], btype='bandpass', fs=sr, output='sos')
    voice_band = scipy.signal.sosfilt(sos_voice, mono)
    
    # Separación Armónica-Percusiva (HPSS)
    # Las voces son armónicas y dinámicas
    D = librosa.stft(voice_band, n_fft=1024, hop_length=hop_length)
    H, P = librosa.decompose.hpss(D, margin=1.5)
    harm_energy = np.sum(np.abs(H)**2, axis=0)
    
    # Contraste espectral
    contrast = librosa.feature.spectral_contrast(S=np.abs(D), sr=sr, fmin=300, n_bands=4)
    vocal_contrast = contrast[2]  # Banda alrededor de 1200 - 2500 Hz (formantes de voz)
    
    # Normalizar métricas
    h_norm = (harm_energy - np.percentile(harm_energy, 10)) / max(1e-6, np.percentile(harm_energy, 90) - np.percentile(harm_energy, 10))
    h_norm = np.clip(h_norm, 0, 1)
    
    c_norm = (vocal_contrast - np.percentile(vocal_contrast, 10)) / max(1e-6, np.percentile(vocal_contrast, 90) - np.percentile(vocal_contrast, 10))
    c_norm = np.clip(c_norm, 0, 1)
    
    # Puntuación vocal frame a frame
    vocal_score = 0.65 * h_norm + 0.35 * c_norm
    times = librosa.frames_to_time(np.arange(len(vocal_score)), sr=sr, hop_length=hop_length)
    
    return times, vocal_score

times1, v_score1 = detect_vocal_activity_profile(m1, sr)
times2, v_score2 = detect_vocal_activity_profile(m2, sr)

def summarize_vocal_by_phrase(times, scores, track_duration, phrase_len_sec):
    num_phrases = int(track_duration // phrase_len_sec)
    res = []
    for ph in range(num_phrases):
        t_start = ph * phrase_len_sec
        t_end = (ph + 1) * phrase_len_sec
        mask = (times >= t_start) & (times < t_end)
        mean_score = float(np.mean(scores[mask])) if np.any(mask) else 0.0
        # Consideramos presencia vocal significativa si mean_score > 0.45
        has_vocal = mean_score > 0.45
        m, s = int(t_start // 60), int(t_start % 60)
        res.append({
            "phrase": ph,
            "time_str": f"{m:02d}:{s:02d}",
            "time_sec": t_start,
            "vocal_score": mean_score,
            "has_vocal": has_vocal
        })
    return res

# Cada frase de 32 beats a ~136 BPM dura unos 14.1 segundos
ph_len1 = (60.0 / 135.80) * 32
ph_len2 = (60.0 / 139.17) * 32

voc1 = summarize_vocal_by_phrase(times1, v_score1, len(m1)/sr, ph_len1)
voc2 = summarize_vocal_by_phrase(times2, v_score2, len(m2)/sr, ph_len2)

print("=== MAPA VOCAL DE PISTA 1: Icehouse ===")
for v in voc1:
    tag = "🎤 [VOCAL]" if v["has_vocal"] else "   [INSTRUMENTAL]"
    print(f"Frase {v['phrase']:02d} [{v['time_str']}]: Score={v['vocal_score']:.2f} {tag}")

print("\n=== MAPA VOCAL DE PISTA 2: Harry Choo Choo Romero ===")
for v in voc2:
    tag = "🎤 [VOCAL]" if v["has_vocal"] else "   [INSTRUMENTAL]"
    print(f"Frase {v['phrase']:02d} [{v['time_str']}]: Score={v['vocal_score']:.2f} {tag}")
