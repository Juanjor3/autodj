# AutoDJ - Motor de Creación y Mezcla Continua de Sesiones

Aplicación en Docker para mezclar automáticamente canciones en una sesión continua (*DJ mix set*), con beat-matching, alineación armónica (Camelot Wheel), cruce de frecuencias EQ (Bass Swap) y masterización final.

---

## 🎧 Tecnologías y Librerías

- **`librosa>=0.10.0`**: Detección de BPM, grid de compases (beatgrid) y estimación de tonalidad armónica (Krumhansl-Schmuckler a notación Camelot).
- **`pyrubberband>=0.3.0` + `rubberband-cli`**: Sincronización de tempo de alta fidelidad (time-stretching y pitch-shifting imperceptible en las transiciones).
- **`pedalboard>=0.8.0`**: Transición profesional con corte de graves (*Bass Swap*), filtros paso-alto/paso-bajo, colas de reverb y bus de masterización con limitador anti-clipping.
- **`soundfile>=0.12.0`**: Lectura y exportación de audio de alta resolución en formato estéreo sin pérdidas.
- **`numpy<2.0.0`**: Operaciones de matriz de audio, curvas de volumen (crossfade suave) y síntesis de señales.

---

## 📁 Estructura del Proyecto

```text
autodj/
├── Dockerfile              # Imagen Python 3.11-slim con rubberband-cli, ffmpeg y libsndfile1
├── docker-compose.yml       # Orquestación con montajes de carpetas locales
├── requirements.txt         # Dependencias Python
├── README.md
├── app/
│   ├── analyzer.py         # Análisis de BPM, beats y clave Camelot
│   ├── sequencer.py        # Ordenación armónica inteligente de pistas
│   ├── transition.py       # Beatmatching con Rubberband y filtros DJ (Bass Swap)
│   ├── mastering.py        # Bus de compresión y limitador de pico
│   ├── mixer.py            # Ensamblado de la sesión continua y tracklist
│   ├── generator.py        # Síntesis de pistas de demostración
│   ├── main.py             # CLI ejecutable principal
│   ├── test_env.py         # Test de verificación de entorno y librerías
│   └── test_mixer.py       # Test integral de mezcla de sesión
├── input/                  # 👈 Coloca aquí tus pistas (.mp3, .wav, .flac, etc.)
└── output/                 # 👈 Aquí se generan las sesiones (.wav) y tracklists (.txt)
```

---

## 🚀 Uso Rápido

### 1. Mezclar canciones automáticamente
1. Copia tus archivos de audio favoritos (`.mp3`, `.wav`, `.flac`, `.m4a`) a la carpeta `input/`.
2. Ejecuta:
```bash
docker compose run --rm autodj
```
3. El motor:
   - Analizará el BPM y la clave musical de cada canción.
   - Ordenará la sesión inteligentemente para que los cambios de tempo y tono sean suaves.
   - Mezclará las transiciones a compás aplicando la técnica de *Bass Swap* para evitar colisiones de bombos.
   - Masterizará la sesión con limitador para evitar distorsión digital.
   - Guardará la sesión resultante en `output/session_YYYYMMDD_HHMMSS.wav` junto a su `output/tracklist_YYYYMMDD_HHMMSS.txt`.

### 2. Opciones avanzadas de mezcla

```bash
# Cambiar la duración de la transición a 64 beats (más larga) o 16 beats (más rápida):
docker compose run --rm autodj python main.py --transition-beats=32

# Respetar el orden manual / alfabético de los archivos sin reordenar:
docker compose run --rm autodj python main.py --order=manual

# Asignar un nombre específico al archivo de salida:
docker compose run --rm autodj python main.py --session-name=mi_sesion_verano.wav
```

### 3. Ejecutar los tests de verificación
```bash
# Comprobar el entorno y dependencias:
docker compose run --rm autodj python test_env.py

# Comprobar el motor de mezcla con pistas sintéticas:
docker compose run --rm autodj python test_mixer.py
```
