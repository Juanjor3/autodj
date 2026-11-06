FROM python:3.11-slim

# Evitar prompts interactivos durante la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema indispensables:
# - rubberband-cli: motor CLI requerido por pyrubberband para time-stretching y pitch-shifting
# - ffmpeg: decodificación y conversión de formatos de audio (MP3, AAC, FLAC, etc.) para librosa
# - libsndfile1: librería nativa requerida por soundfile para lectura y escritura de audio
# - build-essential: compilador C/C++ en caso de requerirse compilación nativa
RUN apt-get update && apt-get install -y --no-install-recommends \
    rubberband-cli \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias de Python con caché aislada
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY app/ /app/

CMD ["python", "main.py"]
