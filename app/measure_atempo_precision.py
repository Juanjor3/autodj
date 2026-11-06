import soundfile as sf
import numpy as np
import scipy.signal
from pathlib import Path

f = Path('/app/output/test_gaetano_atempo.wav')
data, sr = sf.read(f)
mono = data[:, 0] if data.ndim > 1 else data

sos = scipy.signal.butter(4, [40, 140], btype='bandpass', fs=sr, output='sos')
k = scipy.signal.sosfilt(sos, mono)
env = np.abs(scipy.signal.hilbert(k))

peaks, _ = scipy.signal.find_peaks(env[50*sr : 150*sr], distance=18000, prominence=0.08)
diffs = np.diff(peaks)
clean = diffs[(diffs >= 19000) & (diffs <= 20000)]
median_spb = np.median(clean)
bpm = sr * 60.0 / median_spb

print(f"Gaetano Parisio tras FFmpeg atempo:")
print(f"Mediana SPB: {median_spb:.2f} (Objetivo Sequential One: 19602.00)")
print(f"BPM resultante: {bpm:.4f} (Objetivo Sequential One: 134.9862)")
print(f"Diferencia: {abs(median_spb - 19602.00):.2f} samples ({abs(median_spb - 19602.00)*1000/sr:.3f} ms por bombo)")
