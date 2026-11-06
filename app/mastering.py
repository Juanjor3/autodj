import numpy as np
import pedalboard


def master_session_audio(audio_stereo: np.ndarray, sr: int) -> np.ndarray:
    """
    Cadena de masterización para la sesión completa:
    - Filtro subsónico a 25 Hz para limpiar frecuencias inaudibles que saturan altavoces.
    - Compresión dinámica suave para cohesión sonora entre temas.
    - Limitador de pico (Peak Limiter) a -0.5 dBFS para garantizar cero clipping digital.
    """
    mastering_board = pedalboard.Pedalboard([
        pedalboard.HighpassFilter(cutoff_frequency_hz=25.0),
        pedalboard.Compressor(
            threshold_db=-12.0,
            ratio=2.0,
            attack_ms=20.0,
            release_ms=120.0
        ),
        pedalboard.Gain(gain_db=1.0),
        pedalboard.Limiter(
            threshold_db=-0.5,
            release_ms=80.0
        )
    ])

    processed = mastering_board(audio_stereo, sr)

    # Protección adicional contra picos residuales
    peak = np.max(np.abs(processed))
    if peak > 0.99:
        processed = processed * (0.98 / peak)

    return processed.astype(np.float32)
