
import soundfile as sf
import numpy as np
import scipy.signal
from pathlib import Path
from analyzer import analyze_audio_file
from mixer import align_kick_phase

p1 = list(Path('/app/input').glob('*Sequential*'))[0]
p2 = list(Path('/app/input').glob('*Gaetano*'))[0]

prof1 = analyze_audio_file(p1, load_audio=True)
prof2 = analyze_audio_file(p2, load_audio=True)

# Veamos los compases de prof1 y prof2
b1 = prof1.beat_samples
b2 = prof2.beat_samples

print('Track 1 beats count:', len(b1))
print('Track 2 beats count:', len(b2))

# Distancia media entre beats (samples per beat)
spb1_mean = np.mean(np.diff(b1))
spb2_mean = np.mean(np.diff(b2))

print(f'Track 1 mean SPB: {spb1_mean:.2f} ({44100*60/spb1_mean:.3f} BPM)')
print(f'Track 2 mean SPB: {spb2_mean:.2f} ({44100*60/spb2_mean:.3f} BPM)')

# Diferencias compas a compas a lo largo de 448 beats:
print('\nDiferencia acumulada en ms si no hay sincronizacion continua:')
for n in [32, 64, 128, 256, 384, 448]:
    drift_samples = n * (spb1_mean - spb2_mean)
    drift_ms = (drift_samples / 44100) * 1000
    print(f'Beat +{n:03d} ({n//4:02d} compases): {drift_samples:+.1f} samples ({drift_ms:+.2f} ms)')
