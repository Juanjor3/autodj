from pathlib import Path
import numpy as np
import soundfile as sf


def generate_demo_track(
    output_path: Path,
    title: str,
    bpm: float,
    root_freq: float,
    duration_beats: int = 48,
    sr: int = 44100
):
    """
    Sintetiza una pista rítmica electrónica con bombo, palmas y bajo para pruebas.
    """
    spb = 60.0 / bpm  # segundos por beat
    total_seconds = duration_beats * spb
    num_samples = int(total_seconds * sr)
    t = np.linspace(0, total_seconds, num_samples, endpoint=False, dtype=np.float32)

    # Inicializar canales estéreo
    left = np.zeros(num_samples, dtype=np.float32)
    right = np.zeros(num_samples, dtype=np.float32)

    samples_per_beat = int(spb * sr)

    for b in range(duration_beats):
        beat_start = b * samples_per_beat

        # 1. Bombo (Kick) en cada beat (Four-on-the-floor)
        kick_len = int(0.20 * sr)
        if beat_start + kick_len <= num_samples:
            t_k = np.linspace(0, 0.20, kick_len, endpoint=False)
            freq_env = 130.0 * np.exp(-t_k * 25.0) + 45.0
            amp_env = np.exp(-t_k * 18.0)
            phase = 2 * np.pi * np.cumsum(freq_env) / sr
            kick = 0.85 * np.sin(phase) * amp_env
            left[beat_start : beat_start + kick_len] += kick
            right[beat_start : beat_start + kick_len] += kick

        # 2. Clap / Snare en los beats 2 y 4 (índices 1 y 3)
        if b % 2 == 1:
            snare_len = int(0.18 * sr)
            if beat_start + snare_len <= num_samples:
                t_s = np.linspace(0, 0.18, snare_len, endpoint=False)
                noise = np.random.uniform(-0.4, 0.4, snare_len) * np.exp(-t_s * 22.0)
                tone = 0.3 * np.sin(2 * np.pi * 180.0 * t_s) * np.exp(-t_s * 25.0)
                snare = noise + tone
                left[beat_start : beat_start + snare_len] += snare * 0.9
                right[beat_start : beat_start + snare_len] += snare * 1.1

        # 3. Línea de bajo sintetizada (Off-beat bassline)
        bass_start = beat_start + samples_per_beat // 2
        bass_len = int(samples_per_beat * 0.45)
        if bass_start + bass_len <= num_samples:
            t_b = np.linspace(0, bass_len / sr, bass_len, endpoint=False)
            bass_freq = root_freq * (1.0 if (b % 4 < 2) else 1.25)
            bass = 0.5 * np.sin(2 * np.pi * bass_freq * t_b) * np.exp(-t_b * 6.0)
            left[bass_start : bass_start + bass_len] += bass
            right[bass_start : bass_start + bass_len] += bass

    # Normalizar audio a -1 dB
    stereo = np.vstack([left, right])
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo * (0.85 / peak)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, stereo.T, sr)
    print(f"🎵 Pista demo generada: {output_path.name} ({bpm} BPM, {duration_beats} beats, {total_seconds:.1f}s)")
