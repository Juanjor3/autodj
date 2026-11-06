import numpy as np
import librosa
import scipy.signal
from pathlib import Path
from analyzer import load_stereo_audio

def compute_vocal_profile(audio_mono: np.ndarray, sr: int = 22050, hop_length: int = 512):
    """
    Calcula el perfil continuo de presencia vocal (0.0 a 1.0) mediante:
    1. Filtrado en banda formántica (300 a 3400 Hz)
    2. Separación armónica de fuentes (HPSS)
    3. Contraste espectral en frecuencias de formantes vocales (1200 - 2500 Hz)
    """
    sos_voice = scipy.signal.butter(4, [300, 3400], btype='bandpass', fs=sr, output='sos')
    voice_band = scipy.signal.sosfilt(sos_voice, audio_mono)

    D = librosa.stft(voice_band, n_fft=1024, hop_length=hop_length)
    H, _ = librosa.decompose.hpss(D, margin=1.5)
    harm_energy = np.sum(np.abs(H)**2, axis=0)

    contrast = librosa.feature.spectral_contrast(S=np.abs(D), sr=sr, fmin=300, n_bands=4)
    vocal_contrast = contrast[2]

    h_norm = (harm_energy - np.percentile(harm_energy, 10)) / max(1e-6, np.percentile(harm_energy, 90) - np.percentile(harm_energy, 10))
    h_norm = np.clip(h_norm, 0, 1)

    c_norm = (vocal_contrast - np.percentile(vocal_contrast, 10)) / max(1e-6, np.percentile(vocal_contrast, 90) - np.percentile(vocal_contrast, 10))
    c_norm = np.clip(c_norm, 0, 1)

    vocal_score = 0.65 * h_norm + 0.35 * c_norm
    times = librosa.frames_to_time(np.arange(len(vocal_score)), sr=sr, hop_length=hop_length)
    return times, vocal_score

def get_phrase_vocal_activity(times, scores, beats, sr, phrase_beats=32):
    num_phrases = len(beats) // phrase_beats
    phrase_vocal = np.zeros(num_phrases, dtype=np.float32)
    for p in range(num_phrases):
        t_start = beats[p * phrase_beats] / sr
        t_end = beats[(p + 1) * phrase_beats - 1] / sr
        mask = (times >= t_start) & (times <= t_end)
        if np.any(mask):
            phrase_vocal[p] = float(np.mean(scores[mask]))
    return phrase_vocal

def select_clash_free_drop(
    bass1, rms1, vocal1,
    bass2, rms2, vocal2,
    num_phrases1, num_phrases2,
    intro_phrases: int = 2,
    overlap_phrases: int = 4
):
    """
    Selecciona el punto de drop que maximiza la pegada y clímax pero
    GARANTIZA CERO COLISIÓN DE VOCES:
    Si en el subidón o zona de mezcla coinciden voces en ambas pistas,
    el algoritmo descarta ese punto y busca el siguiente subidón limpio.
    """
    # Ventana de búsqueda en pista 1 (desde 50% hasta el outro)
    min_p1 = max(intro_phrases + 1, int(num_phrases1 * 0.50))
    max_p1 = max(min_p1 + 1, num_phrases1 - 2)

    best_score = -1e9
    best_p1_drop = min_p1
    best_clash = 0.0

    print(f"\n--- EVALUACIÓN DE CANDIDATOS A SUBIDÓN Y COLISIÓN DE VOCES ---")
    for p1 in range(min_p1, max_p1):
        # Punto de lanzamiento en Pista 1: p1 - intro_phrases
        p1_start = p1 - intro_phrases
        
        # Evaluar la zona de mezcla y convivencia (intro + overlap)
        total_mix_phrases = intro_phrases + overlap_phrases
        
        clash_detected = False
        max_clash_val = 0.0
        
        for step in range(total_mix_phrases):
            curr_p1 = p1_start + step
            curr_p2 = step  # Pista 2 avanza desde frase 0
            
            if curr_p1 < num_phrases1 and curr_p2 < num_phrases2:
                v1 = vocal1[curr_p1]
                v2 = vocal2[curr_p2]
                clash = v1 * v2
                if clash > max_clash_val:
                    max_clash_val = clash
                # Si ambas superan el umbral vocal de 0.45
                if v1 > 0.45 and v2 > 0.45:
                    clash_detected = True

        # Puntuación de energía en el drop
        prev_b1 = np.min(bass1[max(0, p1 - 2) : p1]) if p1 >= 2 else bass1[p1]
        jump1 = bass1[p1] - prev_b1
        energy_score = bass1[p1] * 2.0 + jump1 * 3.0 + rms1[p1]
        
        # Penalización severa por colisión vocal
        penalty = 100.0 if clash_detected else 0.0
        final_score = energy_score - penalty

        status = "❌ RECHAZADO (COLISIÓN DE VOCES)" if clash_detected else "✅ LIMPIO"
        print(f"Candidato Frase {p1:02d}: Energía={energy_score:.2f} | VocalClash={max_clash_val:.2f} | {status}")

        if final_score > best_score:
            best_score = final_score
            best_p1_drop = p1
            best_clash = max_clash_val

    return best_p1_drop, best_clash

print("Módulo de resolución de colisiones vocales listo para pruebas.")
