import librosa
import numpy as np
import scipy.signal
from pathlib import Path

# Analizar la transición real de CuBe en 03:30 - 04:30 de Vol 44
f_p = Path("/cube_music/CubeDanceClub - Volumen 44.mp3")
sr = 22050
y, _ = librosa.load(f_p, sr=sr, offset=200.0, duration=80.0, mono=True) # 03:20 a 04:40

# Encontrar tempo local
tempo = 140.0 # CuBe suele estar en 138-142 BPM
spb = int((60.0 / tempo) * sr) # ~9450 muestras por beat
bar_len = spb * 4 # ~37800 muestras por compás

# Medir energía compás a compás durante 20 compases
sos_low = scipy.signal.butter(4, 160, btype="lowpass", fs=sr, output="sos")
low = scipy.signal.sosfilt(sos_low, y)

print("Análisis Compás a Compás de la Transición Real en CuBe Vol 44:")
print("Compás | Tiempo | RMS Total | Graves (<160Hz) | Observaciones")
print("-" * 65)

n_bars = len(y) // bar_len
for b_idx in range(min(n_bars, 24)):
    s1 = b_idx * bar_len
    s2 = (b_idx + 1) * bar_len
    t_sec = 200.0 + (s1 / sr)
    
    r_t = float(np.sqrt(np.mean(y[s1:s2] ** 2)))
    r_l = float(np.sqrt(np.mean(low[s1:s2] ** 2)))
    
    obs = ""
    if r_l > 0.10:
        obs = "Ritmo pleno (Bombo presente)"
    elif r_l < 0.04:
        obs = "Parón / Bajada (Sin bombo)"
    else:
        obs = "Transición / Bombo atenuado"
        
    print(f"  {b_idx+1:02d}   | {int(t_sec//60):02d}:{int(t_sec%60):02d}  |   {r_t:.3f}   |     {r_l:.3f}     | {obs}")
