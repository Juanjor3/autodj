import numpy as np
import soundfile as sf
import scipy.signal
from pathlib import Path
from analyzer import load_stereo_audio

input_files = sorted([f for f in Path('/app/input').iterdir() if f.suffix.lower() in ['.flac', '.wav', '.mp3']])
print(f"Archivos en /app/input encontrados ({len(input_files)}):")
for f in input_files:
    print(f"  - {f.name}")

sr = 44100

def analyze_track_subidones(file_path):
    y, _ = load_stereo_audio(file_path, sr)
    mono = 0.5 * (y[0] + y[1])
    
    # Filtro de bombo para detectar tempo y rejilla
    sos_kick = scipy.signal.butter(4, [45, 120], btype='bandpass', fs=sr, output='sos')
    kick_filt = scipy.signal.sosfilt(sos_kick, mono)
    env = np.abs(scipy.signal.hilbert(kick_filt))
    
    sample_len = min(len(env), sr * 45)
    acorr = scipy.signal.correlate(env[:sample_len], env[:sample_len], mode='full')
    center = sample_len - 1
    lags = acorr[center:]
    p_bar = np.argmax(lags[int(sr * 1.60) : int(sr * 1.95)]) + int(sr * 1.60)
    spb = float(p_bar) / 4.0
    bpm = (60.0 * sr) / spb
    
    peaks, _ = scipy.signal.find_peaks(env[:sr*3], height=np.max(env[:sr*3])*0.25, distance=int(spb*0.80))
    first_kick = int(peaks[0]) if len(peaks) > 0 else 0
    num_beats = int((len(mono) - first_kick) // spb)
    grid = (first_kick + np.arange(num_beats) * spb).astype(int)
    
    # Análisis espectral multibanda
    sos_l = scipy.signal.butter(4, [40, 160], btype='bandpass', fs=sr, output='sos')
    sos_m = scipy.signal.butter(4, [300, 3000], btype='bandpass', fs=sr, output='sos')
    sos_h = scipy.signal.butter(4, 3500, btype='highpass', fs=sr, output='sos')
    
    low = scipy.signal.sosfilt(sos_l, mono)
    mid = scipy.signal.sosfilt(sos_m, mono)
    high = scipy.signal.sosfilt(sos_h, mono)
    
    # Mapear cada frase de 32 beats (8 compases)
    phrases = []
    num_phrases = len(grid) // 32
    for ph in range(num_phrases):
        b_start = ph * 32
        b_end = (ph + 1) * 32
        s_start = grid[b_start]
        s_end = grid[b_end - 1]
        
        rms_l = float(np.sqrt(np.mean(low[s_start:s_end]**2)))
        rms_m = float(np.sqrt(np.mean(mid[s_start:s_end]**2)))
        rms_h = float(np.sqrt(np.mean(high[s_start:s_end]**2)))
        rms_tot = float(np.sqrt(np.mean(mono[s_start:s_end]**2)))
        
        t_sec = s_start / sr
        phrases.append({
            "phrase": ph,
            "beat": b_start,
            "time_sec": t_sec,
            "time_str": f"{int(t_sec//60):02d}:{int(t_sec%60):02d}",
            "low": rms_l,
            "mid": rms_m,
            "high": rms_h,
            "tot": rms_tot
        })
        
    # Detectar subidones (Drops):
    # Un subidón se produce cuando venimos de una frase o compás con bajada de graves (breakdown/parón)
    # o aumento de medios/agudos (melodía/riser) y de repente el bombo (Low) rompe con fuerza.
    drops = []
    for ph in range(1, num_phrases):
        prev = phrases[ph - 1]
        curr = phrases[ph]
        
        # Salto de energía en graves (Low surge)
        low_jump = curr["low"] - prev["low"]
        low_ratio = curr["low"] / max(1e-4, prev["low"])
        
        # Si venía de un parón (< 0.10 o caída fuerte) y ahora rompe con Low >= 0.15 y ratio > 1.3
        if curr["low"] >= 0.12 and (low_jump > 0.04 or low_ratio > 1.35):
            score = low_jump * 10 + curr["low"] * 5 + curr["tot"]
            drops.append({
                "phrase": ph,
                "beat": curr["beat"],
                "time_str": curr["time_str"],
                "time_sec": curr["time_sec"],
                "low_prev": prev["low"],
                "low_curr": curr["low"],
                "low_ratio": low_ratio,
                "score": score
            })
            
    # Ordenar drops por relevancia
    drops = sorted(drops, key=lambda x: x["score"], reverse=True)
    
    return {
        "name": file_path.name,
        "bpm": bpm,
        "duration": len(mono) / sr,
        "phrases": phrases,
        "drops": drops,
        "grid": grid,
        "spb": spb,
        "first_kick": first_kick
    }

for f in input_files:
    info = analyze_track_subidones(f)
    print(f"\n========================================================")
    print(f"ANÁLISIS ESPECTRAL DE: {info['name']}")
    print(f"BPM: {info['bpm']:.2f} | Duración: {int(info['duration']//60):02d}:{int(info['duration']%60):02d} | Frases de 32: {len(info['phrases'])}")
    print(f"--- Subidones detectados (Drops tras parón) ---")
    if not info['drops']:
        print("  (Pista continua / techno lineal sin grandes parones de graves)")
    else:
        for d in info['drops']:
            print(f"  🚀 SUBIDÓN en [{d['time_str']}] (Frase {d['phrase']}, Beat {d['beat']}): Graves suben de {d['low_prev']:.3f} a {d['low_curr']:.3f} (x{d['low_ratio']:.2f}) [Score={d['score']:.2f}]")
            
    print(f"--- Desglose por Frases ---")
    for ph in info['phrases']:
        print(f"  Frase {ph['phrase']:02d} [{ph['time_str']}]: Low={ph['low']:.3f} | Mid={ph['mid']:.3f} | High={ph['high']:.3f} | Tot={ph['tot']:.3f}")
