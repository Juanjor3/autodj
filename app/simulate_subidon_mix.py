import numpy as np
import soundfile as sf
import scipy.signal
from pathlib import Path
from analyzer import load_stereo_audio
from detect_subidones_spectral import analyze_track_subidones

# Analizar las dos pistas en /app/input
p1 = Path('/app/input/136 8a Icehouse - Hey Little Girl (X-Tended Edit).flac')
p2 = Path('/app/input/Harry_choo-choo_romero-i_want_out_(i_cant_believe).flac')

info1 = analyze_track_subidones(p1)
info2 = analyze_track_subidones(p2)

print("=" * 60)
print("ALINEACIÓN ESPECTRAL DE SUBIDONES (CLIMAX DROP ALIGNMENT)")
print("=" * 60)

# Caso 1: Icehouse (Pista 1) -> Harry Choo Choo Romero (Pista 2)
# Icehouse: Subidón principal en Frase 17 [04:02] (Beat 544)
drop1_beat = 544
drop1_time = info1['phrases'][17]['time_sec']

print(f"\nCaso 1: [Pista 1] Icehouse -> [Pista 2] Harry Choo Choo Romero")
print(f"  • Pista 1 (Icehouse):")
print(f"      - Parón (Breakdown): Frase 15-16 [03:33 - 04:02] (Graves caen a 0.198)")
print(f"      - Subidón (Climax Drop): Frase 17 [04:02] (Beat 544, Graves rompen a 0.295)")
print(f"  • Pista 2 (Harry Choo Choo Romero):")
print(f"      - Si entra con intro de 64 beats (16 compases / 2 frases):")
n_intro = 64
play_beat1 = drop1_beat - n_intro # Beat 480 [03:33]
play_time1 = play_beat1 * info1['spb'] / 44100
m_p, s_p = int(play_time1 // 60), int(play_time1 % 60)
m_d, s_d = int(drop1_time // 60), int(drop1_time % 60)

print(f"      - Punto PLAY de Harry en Icehouse: [{m_p:02d}:{s_p:02d}] (Beat {play_beat1})")
print(f"      - ¡JUSTO AL EMPEZAR EL PARÓN DE ICEHOUSE EN [03:33]!")
print(f"      - Durante [03:33 a 04:02] (16 compases): Icehouse tiene su vocal y Harry va asomando sus percusiones.")
print(f"      - En [{m_d:02d}:{s_d:02d}] (Beat {drop1_beat}): ¡LOS DOS TEMAS ROMPEN EN EL SUBIDÓN AL UNÍSONO!")

# Caso 2: Harry Choo Choo Romero (Pista 1) -> Icehouse (Pista 2)
print(f"\nCaso 2: [Pista 1] Harry Choo Choo Romero -> [Pista 2] Icehouse")
print(f"  • Pista 1 (Harry Choo Choo Romero):")
print(f"      - Parón monumental: Frase 13 [02:59] (Graves caen de 0.333 a 0.050)")
print(f"      - Subidón brutal (Drop): Frase 14 [03:13] (Beat 448, Graves explotan a 0.345)")
print(f"  • Pista 2 (Icehouse):")
drop_h_beat = 448
drop_h_time = info2['phrases'][14]['time_sec']
n_intro_ice = 64
play_beat_h = drop_h_beat - n_intro_ice # Beat 384 [02:45]
play_time_h = play_beat_h * info2['spb'] / 44100
m_ph, s_ph = int(play_time_h // 60), int(play_time_h % 60)
m_dh, s_dh = int(drop_h_time // 60), int(drop_h_time % 60)

print(f"      - Punto PLAY de Icehouse en Harry: [{m_ph:02d}:{s_ph:02d}] (Beat {play_beat_h})")
print(f"      - En [{m_dh:02d}:{s_dh:02d}] (Beat {drop_h_beat}): ¡EL SUBIDÓN BRUTAL DE HARRY ROMPE CON ICEHOUSE A LA VEZ!")
