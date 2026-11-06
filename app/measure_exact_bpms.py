import soundfile as sf
import numpy as np
import scipy.signal
from pathlib import Path
from analyzer import analyze_audio_file

p1 = list(Path('/app/input').glob('*Sequential*'))[0]
p2 = list(Path('/app/input').glob('*Gaetano*'))[0]

prof1 = analyze_audio_file(p1, load_audio=True)
prof2 = analyze_audio_file(p2, load_audio=True)
sr = 44100

sos = scipy.signal.butter(4, [40, 140], btype='bandpass', fs=sr, output='sos')
k1 = scipy.signal.sosfilt(sos, prof1.audio_stereo[0])
k2 = scipy.signal.sosfilt(sos, prof2.audio_stereo[0])

env1 = np.abs(scipy.signal.hilbert(k1))
env2 = np.abs(scipy.signal.hilbert(k2))

# Medir distancia entre 128 bombos consecutivos (32 compases) en Sequential One:
peaks1, _ = scipy.signal.find_peaks(env1[100*sr : 200*sr], distance=18000, prominence=0.08)
diffs1 = np.diff(peaks1)
clean1 = diffs1[(diffs1 >= 19000) & (diffs1 <= 20000)]
print(f"Sequential One: Mediana SPB = {np.median(clean1):.2f} -> BPM = {sr*60/np.median(clean1):.4f}")

# Medir distancia entre 128 bombos consecutivos en Gaetano Parisio:
peaks2, _ = scipy.signal.find_peaks(env2[50*sr : 150*sr], distance=18000, prominence=0.08)
diffs2 = np.diff(peaks2)
clean2 = diffs2[(diffs2 >= 19000) & (diffs2 <= 20000)]
print(f"Gaetano Parisio: Mediana SPB = {np.median(clean2):.2f} -> BPM = {sr*60/np.median(clean2):.4f}")
