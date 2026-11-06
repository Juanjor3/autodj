import soundfile as sf
import numpy as np
import scipy.signal

wav_f = '/app/output/Sequential_One_vs_Gaetano_Parisio.wav'
d, sr = sf.read(wav_f)
mono = 0.5 * (d[:, 0] + d[:, 1])

# Filtrar bombo en la zona de mezcla [04:25 a 05:15]
s_start = int(265 * sr)
s_end = int(315 * sr)
sub = mono[s_start:s_end]

sos = scipy.signal.butter(4, [45, 120], btype='bandpass', fs=sr, output='sos')
kick = scipy.signal.sosfilt(sos, sub)
env = np.abs(scipy.signal.hilbert(kick))

# Buscar picos de bombo
min_dist = int(sr * 0.38) # ~135 BPM
peaks, _ = scipy.signal.find_peaks(env, height=0.10, distance=min_dist)

print(f"Análisis de bombos en la mezcla final entre [04:25] y [05:15]:")
print(f"Total bombos detectados: {len(peaks)}")
diffs = np.diff(peaks)
diffs_ms = diffs * 1000.0 / sr
print(f"Mediana de separación entre bombos: {np.median(diffs_ms):.2f} ms ({60000.0/np.median(diffs_ms):.2f} BPM)")
print(f"Desviación estándar entre golpes de bombo: {np.std(diffs_ms):.2f} ms")

# Si hubiera dos bombos desfasados (descuadre / "caballos"), veríamos picos dobles o diffs de < 200 ms
small_diffs = diffs_ms[diffs_ms < 300.0]
print(f"Número de bombos pisados / desfasados (doble bombo < 300 ms): {len(small_diffs)}")
if len(small_diffs) == 0:
    print(">>> RESULTADO: ¡CERO BOMBOS DESFASADOS! RITMO CLAVADO AL MILISEGUNDO <<<")
