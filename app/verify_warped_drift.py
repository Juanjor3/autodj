
import soundfile as sf
import numpy as np
import librosa
from pathlib import Path
from analyzer import analyze_audio_file

p1 = list(Path('/app/input').glob('*Sequential*'))[0]
prof1 = analyze_audio_file(p1, load_audio=False)
sr = 44100
b1 = prof1.beat_samples

# Cargar audio warped
y_warped, _ = librosa.load('/app/output/temp_gaetano_warped.wav', sr=sr, mono=True)
_, bf_w = librosa.beat.beat_track(y=y_warped, sr=sr, start_bpm=135.0, tightness=100)
b_warped = librosa.frames_to_samples(bf_w)

beat1_drop = 288
beat2_drop = 64

# En el timemap, el drop de Gaetano (sample 1251840) fue mapeado a 1251840.
# En el mixer, el offset para alinear los drops es:
drop1_sample = b1[beat1_drop]

# Encontrar el beat mas cercano a sample 1251840 en b_warped:
idx_drop_warped = int(np.argmin(np.abs(b_warped - 1251840)))
print('Drop warped sample:', b_warped[idx_drop_warped], 'esperado: 1251840')

offset = drop1_sample - b_warped[idx_drop_warped]

print('\nComprobando desalineacion beat a beat con WARPED AUDIO:')
max_drift_ms = 0.0
for k in range(0, 449, 32):
    beat_A = beat1_drop + k
    beat_B = idx_drop_warped + k
    if beat_A < len(b1) and beat_B < len(b_warped):
        sample_A = b1[beat_A]
        sample_B = b_warped[beat_B] + offset
        diff_samples = sample_B - sample_A
        diff_ms = (diff_samples / sr) * 1000.0
        if abs(diff_ms) > max_drift_ms:
            max_drift_ms = abs(diff_ms)
        print(f'Beat +{k:03d} (Compas {k//4:02d}): diff = {diff_samples:+d} samples ({diff_ms:+.2f} ms)')

print(f'\nMaxima deriva encontrada con TIMEMAP: {max_drift_ms:.2f} ms')
