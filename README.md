# AutoDJ Studio — Motor de Creación y Mezcla Continua de Sesiones

Aplicación contenerizada en Docker (Python + FastAPI) para analizar, ordenar armónicamente y mezclar sesiones continuas (*DJ mix sets*) con beat-matching, alineación de fase de bombos, separación neuronal de stems (**Meta Demucs v4 HTDemucs**), ecualización zero-phase de 3 bandas y exportación dual en **WAV** y **MP3 256 kbps**.

> **Nota sobre el desarrollo (Ingeniería Asistida por IA):**  
> Este proyecto ha sido concebido, especificado, dirigido y validado por el autor mediante **ingeniería de software asistida por Inteligencia Artificial**. La arquitectura funcional, los criterios de mezcla DJ, las pruebas reales y la puesta en producción en Docker han sido definidos y supervisados por el autor, mientras que la implementación del código DSP/web ha sido generada iterativamente con asistentes de IA.

---

## Características Principales

- **Panel Web & API (`app/web.py`):** Interfaz web moderna (especificación [DESIGN.md](DESIGN.md)) con subida progresiva en tiempo real, ordenación automática por BPM/Camelot y seguimiento de progreso de mezcla.
- **Análisis Rítmico y Armónico (`app/analyzer.py`, `app/sequencer.py`):** Detección de BPM, beatgrid mediante envolvente de graves, estimación de tonalidad Camelot y mapeo estructural de frases.
- **Separación de Stems por IA & Isolator Zero-Phase (`app/stems.py`, `app/transition.py`):**
  - Separación de `drums`, `bass`, `other` y `vocals` con **HTDemucs** sobre la ventana de transición para ejecutar *Bass Swap* limpio y evitar choque vocal (*anti-vocal clash*).
  - Fallback analítico con filtros complementarios zero-phase de 3 bandas.
- **Sincronización y Masterización (`app/mixer.py`, `app/mastering.py`):** Time-stretching de alta fidelidad con `rubberband-cli`, limitador anti-clipping y renderizado simultáneo a **WAV (16-bit 44.1 kHz)** y **MP3 (256 kbps CBR)** vía `ffmpeg`.

---

## Tecnologías

- **Backend / Web:** Python 3.11, FastAPI, Uvicorn
- **Audio & DSP:** `librosa`, `pyrubberband` + `rubberband-cli`, `scipy`, `soundfile`, `pedalboard`, `numpy`, `ffmpeg`
- **IA (Stems):** `demucs` (HTDemucs)
- **Contenerización:** Docker & Docker Compose (healthcheck nativo y rotación de logs `json-file`)

---

## Estructura del Proyecto

```text
autodj/
├── Dockerfile              # Imagen Python 3.11-slim con rubberband-cli, ffmpeg y libsndfile1
├── docker-compose.yml      # Servicio web en puerto 8095 con healthcheck y volúmenes locales
├── DESIGN.md               # Especificación de tokens y componentes de la UI
├── requirements.txt        # Dependencias Python
├── app/
│   ├── web.py              # API REST FastAPI y servidor de la interfaz web
│   ├── analyzer.py         # Análisis de BPM, beatgrid, clave Camelot y frases
│   ├── sequencer.py        # Ordenación armónica de pistas
│   ├── stems.py            # Separación de stems con Demucs HTDemucs y caché .npz
│   ├── transition.py       # Beatmatching, alineación de fase, Bass Swap e Isolator 3-bandas
│   ├── mastering.py        # Normalización y limitador final
│   ├── mixer.py            # Ensamblado completo de la sesión y exportación WAV/MP3
│   ├── main.py             # CLI alternativa para ejecución sin interfaz web
│   └── templates/          # Interfaz web AutoDJ Studio
├── input/                  # Coloca aquí las pistas de entrada (.mp3, .wav, .flac, .m4a...)
└── output/                 # Sesiones masterizadas (.wav, .mp3) y tracklists (.txt)
```

---

## Uso Rápido con Docker Compose

1. Clona el repositorio y coloca tus pistas en `input/` (o súbelas directamente desde el panel web).
2. Levanta el servicio:

```bash
docker compose up -d --build
```

3. Abre el panel web en `http://localhost:8095`.

### Modo CLI (opcional)

```bash
docker compose run --rm autodj python main.py --transition-beats=32
```
