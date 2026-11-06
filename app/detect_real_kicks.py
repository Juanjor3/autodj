import soundfile as sf
import numpy as np
import scipy.signal
from pathlib import Path

p2 = list(Path('/app/input').glob('*Gaetano*'))[0]
d, sr = sf.read(p2)
mono = 0.5 * (d[:, 0] + d[:, 1])

# Filtrar bombo (40 - 120 Hz)
sos = scipy.signal.butter(4, [40, 120], btype='bandpass', fs=sr, output='sos')
kick_filt = scipy.signal.sosfilt(sos, mono)
env = np.abs(scipy.signal.hilbert(kick_filt))

# Encontrar picos de bombo en los primeros 10 segundos
peaks, props = scipy.signal.find_peaks(env[:sr*10], height=0.15, distance=int(sr * 0.35))
print("Picos reales de bombo en Gaetano Parisio (primeros 10 segundos):")
for i, p in enumerate(peaks[:12]):
    diff = (peaks[i] - peaks[i-1]) if i > 0 else 0
    diff_ms = (diff / sr) * 1000 if i > 0 else 0
    print(f"Kick {i:02d}: muestra {p} ({p/sr:.4f}s), diff={diff} ({diff_ms:.2f} ms)")
