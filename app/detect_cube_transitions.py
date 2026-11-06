import librosa
import numpy as np
import scipy.signal
from pathlib import Path

f_p = Path("/cube_music/CubeDanceClub - Volumen 44.mp3")
print(f"Buscando transiciones en: {f_p.name}")

sr = 22050
y, _ = librosa.load(f_p, sr=sr, offset=0.0, duration=900.0, mono=True) # Primeros 15 min

# Calcular chroma cada compás para ver cuándo cambia la armonía/tema
hop = 512
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
# Similitud recurrente para detectar cambios de track
bounds = librosa.segment.agglomerative(chroma, k=6)
bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop)

print("Puntos clave / cambios de canción detectados:")
for t in bound_times:
    m, s = int(t // 60), int(t % 60)
    print(f"  • [{m:02d}:{s:02d}] ({t:.1f}s)")
