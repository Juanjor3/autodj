import numpy as np
from pathlib import Path
from analyzer import analyze_audio_file
from mixer import align_kick_phase

p1 = list(Path('/app/input').glob('*Sequential*'))[0]
p2 = list(Path('/app/input').glob('*Gaetano*'))[0]

prof1 = analyze_audio_file(p1, load_audio=True)
prof2 = analyze_audio_file(p2, load_audio=True)

sr = 44100
master_beats = prof1.beat_samples
next_beats = prof2.beat_samples

# Puntos de mezcla:
# Si Track 2 (Gaetano) entra en su Beat 0:
# Gaetano tiene su primer bombo en next_beats[0].
# Sequential One tiene su bombo en master_beats[start_beat1].
# Para que entren 100% compasados, next_beats[0] debe coincidir con master_beats[start_beat1]!

# Alinear la fase del primer bombo de Gaetano con el bombo de Sequential One:
start_beat1 = 352
s1_start = int(master_beats[start_beat1])
s2_beat0 = int(next_beats[0])

# Ajustar fase microscópica (±25 ms)
s2_beat0_locked = align_kick_phase(prof1.audio_stereo, s1_start, prof2.audio_stereo, s2_beat0, sr=sr, bpm=136.0)

print(f"\n--- COMPROBACIÓN DE CUADRE BOMBO A BOMBO ---")
print(f"Sequential One Beat {start_beat1}: muestra {s1_start}")
print(f"Gaetano Parisio Beat 0 (con micro-phase lock): muestra {s2_beat0_locked}")

# Comprobar los 64 bombos de la intro:
max_err_ms = 0.0
for k in range(65):
    b1_sample = master_beats[start_beat1 + k]
    # Posición en Gaetano si reproducimos continuo desde s2_beat0_locked:
    b2_pos_in_gaetano = s2_beat0_locked + (b1_sample - s1_start)
    # Bombo real de Gaetano:
    b2_real_sample = next_beats[k]
    
    diff_samples = b2_pos_in_gaetano - b2_real_sample
    diff_ms = (diff_samples / sr) * 1000.0
    if abs(diff_ms) > max_err_ms:
        max_err_ms = abs(diff_ms)
    if k % 16 == 0:
        print(f"Compas {k//4:02d} (Beat {k:02d}): Desfase = {diff_ms:+.2f} ms ({diff_samples} samples)")

print(f"\n>>> MÁXIMO ERROR EN LOS 64 BOMBOS: {max_err_ms:.2f} ms <<<")
