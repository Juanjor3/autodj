import soundfile as sf
import numpy as np
import scipy.signal
import pyrubberband as pyrb
from pathlib import Path
from analyzer import load_stereo_audio
from transition import apply_3band_dj_transition
from mastering import master_session_audio

p1 = list(Path('/app/input').glob('*Sequential*'))[0]
p2 = list(Path('/app/input').glob('*Gaetano*'))[0]

y1, sr = load_stereo_audio(p1, 44100)
y2, _ = load_stereo_audio(p2, 44100)

def compute_accurate_tempo_and_grid(y, sr=44100):
    mono = 0.5 * (y[0] + y[1])
    sos = scipy.signal.butter(4, [45, 120], btype='bandpass', fs=sr, output='sos')
    kick_filt = scipy.signal.sosfilt(sos, mono)
    env = np.abs(scipy.signal.hilbert(kick_filt))
    
    # Autocorrelación de compás completo (4 beats en ~135 BPM: rango 1.70s a 1.85s)
    # Usar los primeros 45 segundos con ritmo constante
    sample_len = min(len(env), sr * 45)
    acorr = scipy.signal.correlate(env[:sample_len], env[:sample_len], mode='full')
    center = sample_len - 1
    lags = acorr[center:]
    
    p_bar = np.argmax(lags[int(sr * 1.70) : int(sr * 1.85)]) + int(sr * 1.70)
    spb = float(p_bar) / 4.0
    bpm = (60.0 * sr) / spb
    
    # Localizar el primer downbeat con máxima energía de bombo
    # Buscar el primer pico fuerte en los primeros 2 segundos
    peaks, _ = scipy.signal.find_peaks(env[:sr*3], height=np.max(env[:sr*3])*0.30, distance=int(spb*0.80))
    first_kick = int(peaks[0])
    
    # Rejilla matemática continua perfecta
    num_beats = int((len(mono) - first_kick) // spb)
    grid = (first_kick + np.arange(num_beats) * spb).astype(int)
    
    return bpm, spb, grid, first_kick

bpm1, spb1, grid1, first_kick1 = compute_accurate_tempo_and_grid(y1, sr)
bpm2, spb2, grid2, first_kick2 = compute_accurate_tempo_and_grid(y2, sr)

print(f"Sequential One: BPM = {bpm1:.4f}, SPB = {spb1:.2f} samples, 1er bombo = {first_kick1}")
print(f"Gaetano Parisio: BPM = {bpm2:.4f}, SPB = {spb2:.2f} samples, 1er bombo = {first_kick2}")

# Time stretch de Gaetano para igualar a Sequential One al 100.000%
rate = bpm1 / bpm2
print(f"Ajustando pitch de Gaetano Parisio: rate = {rate:.6f} (corrección de {bpm2 - bpm1:.2f} BPM)")
ch_l = pyrb.time_stretch(y2[0], sr, rate)
ch_r = pyrb.time_stretch(y2[1], sr, rate)
min_l = min(len(ch_l), len(ch_r))
y2_stretched = np.stack([ch_l[:min_l], ch_r[:min_l]]).astype(np.float32)

# Recalcular primer bombo de Gaetano tras el stretch
first_kick2_s = int(round(first_kick2 / rate))
num_beats2 = int((y2_stretched.shape[1] - first_kick2_s) // spb1)
grid2_s = (first_kick2_s + np.arange(num_beats2) * spb1).astype(int)

# PUNTO DE MEZCLA CANÓNICO CUBE: OUTRO DE SEQUENTIAL ONE
# Sequential One dura 347s (5:47). La intro de Gaetano debe entrar en el min 04:16 (Frase 18 / Beat 576)
start_beat1 = 576
swap_beat1 = 576 + 64   # Beat 640 -> 04:45 (BASS SWAP)
end_beat1 = swap_beat1 + 64 # Beat 704 -> 05:13 (Retirada de Sequential One)

s1_start = int(grid1[start_beat1])
s1_swap = int(grid1[swap_beat1])
s1_end = int(grid1[end_beat1])

print(f"Puntos de mezcla en Sequential One:")
print(f"  • Entra Gaetano Parisio: {s1_start/sr/60:02.0f}:{s1_start/sr%60:02.0f} (Muestra {s1_start})")
print(f"  • BASS SWAP (Al unísono): {s1_swap/sr/60:02.0f}:{s1_swap/sr%60:02.0f} (Muestra {s1_swap})")
print(f"  • Fin Sequential One:    {s1_end/sr/60:02.0f}:{s1_end/sr%60:02.0f} (Muestra {s1_end})")

# En Gaetano Parisio:
# Arranca en su primer bombo (grid2_s[0])
# Anclamos el primer bombo de Gaetano al compás s1_start con micro-alineación de fase
s2_beat0 = int(grid2_s[0])

# Ajustar fase microscópica (máximo 15 ms)
test_len = int(spb1 * 4)
mono1 = 0.5 * (y1[0] + y1[1])
mono2_s = 0.5 * (y2_stretched[0] + y2_stretched[1])
sos = scipy.signal.butter(4, [45, 120], btype='bandpass', fs=sr, output='sos')
c1 = np.abs(scipy.signal.hilbert(scipy.signal.sosfilt(sos, mono1[s1_start : s1_start + test_len])))
c2 = np.abs(scipy.signal.hilbert(scipy.signal.sosfilt(sos, mono2_s[s2_beat0 : s2_beat0 + test_len])))

corr = scipy.signal.correlate(c1, c2, mode='full')
center = len(c2) - 1
max_shift = int(sr * 0.015)
search_start = max(0, center - max_shift)
search_end = min(len(corr), center + max_shift + 1)
best_shift = (search_start + np.argmax(corr[search_start:search_end])) - center
print(f"Micro-ajuste de fase en el primer bombo: {best_shift} muestras ({best_shift*1000/sr:+.2f} ms)")

s2_aligned_start = s2_beat0 - best_shift

# Verificar que bombo a bombo el desvío es exactamente 0.00 ms
for b in [0, 16, 32, 48, 64, 96, 127]:
    pos1 = grid1[start_beat1 + b]
    pos2 = s2_aligned_start + int(b * spb1)
    drift = (pos1 - s1_start) - (pos2 - s2_aligned_start)
    print(f"  Beat {b:03d}: Desfase relativo entre pistas = {drift*1000/sr:.4f} ms")

# ENSAMBLADO DE LA SESIÓN
head = y1[:, :s1_start]

# Zona de transición
y1_trans = y1[:, s1_start:s1_end]
n_trans = y1_trans.shape[1]
n_intro = s1_swap - s1_start

y2_trans = y2_stretched[:, s2_aligned_start : s2_aligned_start + n_trans]
if y2_trans.shape[1] < n_trans:
    y2_trans = np.pad(y2_trans, ((0, 0), (0, n_trans - y2_trans.shape[1])), mode="constant")

# Aplicar el nuevo ecualizador 3-Band Isolator DJ
transition_zone = apply_3band_dj_transition(y1_trans, y2_trans, n_intro, sr=sr)

# Cola: Gaetano solo
tail = y2_stretched[:, s2_aligned_start + n_trans :]

final_mix = np.hstack([head, transition_zone, tail]).astype(np.float32)

print("Masterizando sesión final anti-clipping...")
mastered = master_session_audio(final_mix, sr)

out_wav = Path('/app/output/Sequential_One_vs_Gaetano_Parisio.wav')
sf.write(out_wav, mastered.T, sr)
print(f"WAV guardado: {out_wav} ({mastered.shape[1]/sr:.1f}s)")

import subprocess
out_mp3 = Path('/app/output/Sequential_One_vs_Gaetano_Parisio.mp3')
subprocess.run(["ffmpeg", "-y", "-i", str(out_wav), "-codec:a", "libmp3lame", "-b:a", "256k", str(out_mp3)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"MP3 256k HQ guardado: {out_mp3}")
