import librosa
import numpy as np
import scipy.signal

sr = 44100
# Sequential One at 270.70s (start_beat1 = 608)
y1, _ = librosa.load("/app/input/Sequential One - Imagination (Main DJ Version).flac", sr=sr, offset=270.7, duration=6)

# Frank Biazzi from start (sample 0) stretched to 135.0 BPM
y2, _ = librosa.load("/app/input/Frank biazzi - metamorph.mp3", sr=sr, offset=0, duration=6)
import pyrubberband as pyrb
y2_str = pyrb.time_stretch(y2, sr, 134.995 / 136.599)

sos = scipy.signal.butter(4, [45, 120], btype='bandpass', fs=sr, output='sos')
k1 = scipy.signal.sosfilt(sos, y1)
k2 = scipy.signal.sosfilt(sos, y2_str)

env1 = np.abs(scipy.signal.hilbert(k1))
env2 = np.abs(scipy.signal.hilbert(k2))

peaks1, _ = scipy.signal.find_peaks(env1, height=0.2, distance=int(sr * 0.3))
peaks2, _ = scipy.signal.find_peaks(env2, height=0.2, distance=int(sr * 0.3))

print("Sequential One kicks in this 6s window (relative times in s):")
for p in peaks1:
    print(f"  P1 Kick at {p/sr:.4f}s (sample {p}) | Env: {env1[p]:.3f}")

print("\nFrank Biazzi kicks in this 6s window (relative times in s):")
for p in peaks2:
    print(f"  P2 Kick at {p/sr:.4f}s (sample {p}) | Env: {env2[p]:.3f}")
