import soundfile as sf
import numpy as np
from pathlib import Path
from test_perfect_dj_mix import get_exact_kick_tempo

p1 = list(Path('/app/input').glob('*Sequential*'))[0]
p2 = list(Path('/app/input').glob('*Gaetano*'))[0]

d1, sr = sf.read(p1)
d2, _ = sf.read(p2)

m1 = d1[:, 0] if d1.ndim > 1 else d1
m2 = d2[:, 0] if d2.ndim > 1 else d2

bpm1 = get_exact_kick_tempo(m1, sr)
bpm2 = get_exact_kick_tempo(m2, sr)

print(f"BPM exacto Sequential One: {bpm1:.4f}")
print(f"BPM exacto Gaetano Parisio: {bpm2:.4f}")
print(f"Ratio exacto: {bpm1 / bpm2:.6f}")
