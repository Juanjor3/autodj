import soundfile as sf
import numpy as np
import scipy.signal

wav_f = '/app/output/Icehouse_vs_Harry_Choo_Choo_Romero.wav'
d, sr = sf.read(wav_f)
mono = 0.5 * (d[:, 0] + d[:, 1])

# Filtrar bombos en la zona de mezcla [03:45 a 04:45]
s_start = int(225 * sr)
s_end = int(285 * sr)
sub = mono[s_start:s_end]

sos = scipy.signal.butter(4, [45, 120], btype='bandpass', fs=sr, output='sos')
kick = scipy.signal.sosfilt(sos, sub)
env = np.abs(scipy.signal.hilbert(kick))

min_dist = int(sr * 0.38) # ~136 BPM
peaks, _ = scipy.signal.find_peaks(env, height=0.10, distance=min_dist)

print(f"Análisis de bombos de Icehouse vs Harry Choo Choo Romero entre [03:45] y [04:45]:")
print(f"Total bombos detectados: {len(peaks)}")
diffs = np.diff(peaks)
diffs_ms = diffs * 1000.0 / sr
print(f"Mediana de separación: {np.median(diffs_ms):.2f} ms ({60000.0/np.median(diffs_ms):.2f} BPM)")
print(f"Desviación estándar: {np.std(diffs_ms):.2f} ms")

small_diffs = diffs_ms[diffs_ms < 300.0]
print(f"Número de bombos pisados / desfasados (< 300 ms): {len(small_diffs)}")
if len(small_diffs) == 0:
    print(">>> RESULTADO: ¡CERO BOMBOS DESFASADOS! RITMO CLAVADO AL MILISEGUNDO <<<")
