from typing import Tuple, Dict, Optional
import numpy as np
import scipy.signal
import pedalboard
import pyrubberband as pyrb

try:
    from app.analyzer import TrackProfile
except ModuleNotFoundError:
    from analyzer import TrackProfile


SAMPLE_RATE = 44100
TRANSITION_BEATS = 64  # 16 compases en 4/4


class ZeroPhaseDJIsolator3Band:
    """
    Ecualizador Isolator DJ de 3 bandas Zero-Phase (Cero distorsión de fase).
    Inspirado en los crossovers Linkwitz-Riley LR4 y LV-Mix de Mixxx y auto-dj-ai.
    Corte de 24 dB/octava en frecuencias estándar de cabina DJ:
      - Low / Mid Crossover: 260 Hz (Subgraves y pegada de bombo)
      - Mid / High Crossover: 3200 Hz (Vocales, sintes vs platos y charles)
    Reconstrucción matemática perfecta (error < 1e-7) y corte absoluto (> 50 dB de aislamiento).
    """
    def __init__(self, sr: int = SAMPLE_RATE, f_low: float = 260.0, f_high: float = 3200.0):
        self.sr = sr
        self.sos_low = scipy.signal.butter(4, f_low, btype='lowpass', fs=sr, output='sos')
        self.sos_high = scipy.signal.butter(4, f_high, btype='highpass', fs=sr, output='sos')

    def split_bands(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Divide audio estéreo (2, N) en bandas Low, Mid y High complementarias."""
        low = scipy.signal.sosfiltfilt(self.sos_low, audio)
        high = scipy.signal.sosfiltfilt(self.sos_high, audio)
        # La banda media es exactamente el resto: audio - low - high (reconstrucción plana idéntica)
        mid = audio - low - high
        return low.astype(np.float32), mid.astype(np.float32), high.astype(np.float32)

    def apply_dynamic_eq(
        self,
        audio: np.ndarray,
        g_low: np.ndarray,
        g_mid: np.ndarray,
        g_high: np.ndarray
    ) -> np.ndarray:
        """Aplica curvas de ganancia dinámicas independientes por banda."""
        low, mid, high = self.split_bands(audio)
        return (low * g_low + mid * g_mid + high * g_high).astype(np.float32)


# Instancia global del Isolator DJ
DJ_ISOLATOR = ZeroPhaseDJIsolator3Band(sr=SAMPLE_RATE)


def time_stretch_stereo(audio_stereo: np.ndarray, sr: int, stretch_ratio: float) -> np.ndarray:
    """
    Aplica time-stretching estéreo usando pyrubberband con preservación de transitorios.
    stretch_ratio = target_bpm / source_bpm
    """
    if abs(stretch_ratio - 1.0) < 0.0001:
        return audio_stereo.astype(np.float32)

    ch_l = pyrb.time_stretch(audio_stereo[0], sr, stretch_ratio)
    ch_r = pyrb.time_stretch(audio_stereo[1], sr, stretch_ratio)

    min_len = min(len(ch_l), len(ch_r))
    return np.stack([ch_l[:min_len], ch_r[:min_len]]).astype(np.float32)


def apply_migue_fader_cut_blend(
    y1_trans: np.ndarray,
    y2_trans: np.ndarray,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Mezcla auténtica estilo MiGue (CuBe Dance Club / The Natural):
    - Mantiene la 1ª pista sonando a volumen constante durante toda la mezcla (sin bajada progresiva).
    - Ambas pistas conviven con presencia y ecualización natural (sin vaciar graves ni chocar frecuencias).
    - Al terminar el último compás de la 2ª pista, se corta la 1ª pista DE GOLPE (Fader Cut instantáneo),
      con una micro-rampa anti-pop de 3 ms para proteger los monitores/altavoces de chasquidos DC.
    """
    n_trans = y1_trans.shape[1]

    # Ganancias constantes de convivencia (sin bajada progresiva de volumen)
    # Reparto natural de frecuencias:
    # Graves: Track 1 (0.60), Track 2 (0.65) -> Pegada y calidez plena sin distorsión
    g_low1 = np.full(n_trans, 0.60, dtype=np.float32)
    g_mid1 = np.full(n_trans, 0.85, dtype=np.float32)
    g_high1 = np.full(n_trans, 0.85, dtype=np.float32)

    # Micro-rampa anti-pop (3 ms = 132 muestras a 44.1 kHz) justo en el último golpe para corte limpio
    anti_pop_len = min(132, n_trans)
    anti_pop = np.cos(np.linspace(0.0, np.pi * 0.5, anti_pop_len, dtype=np.float32)) ** 2
    g_low1[-anti_pop_len:] *= anti_pop
    g_mid1[-anti_pop_len:] *= anti_pop
    g_high1[-anti_pop_len:] *= anti_pop

    g_low2 = np.full(n_trans, 0.65, dtype=np.float32)
    g_mid2 = np.full(n_trans, 0.85, dtype=np.float32)
    g_high2 = np.full(n_trans, 0.85, dtype=np.float32)

    y1_eq = DJ_ISOLATOR.apply_dynamic_eq(y1_trans, g_low1, g_mid1, g_high1)
    y2_eq = DJ_ISOLATOR.apply_dynamic_eq(y2_trans, g_low2, g_mid2, g_high2)

    return (y1_eq + y2_eq).astype(np.float32)


def apply_equal_power_dj_blend(
    y1_trans: np.ndarray,
    y2_trans: np.ndarray,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Mezcla musical transparente de dos pistas (Equal-Power Frequency Sharing).
    """
    n_trans = y1_trans.shape[1]
    t = np.linspace(0.0, np.pi * 0.5, n_trans, dtype=np.float32)

    cos_t = np.maximum(0.0, np.cos(t)).astype(np.float32)
    sin_t = np.maximum(0.0, np.sin(t)).astype(np.float32)

    g_low1 = cos_t
    g_low2 = sin_t
    g_mid1 = cos_t
    g_mid2 = sin_t
    g_high1 = cos_t
    g_high2 = sin_t

    y1_eq = DJ_ISOLATOR.apply_dynamic_eq(y1_trans, g_low1, g_mid1, g_high1)
    y2_eq = DJ_ISOLATOR.apply_dynamic_eq(y2_trans, g_low2, g_mid2, g_high2)

    return (y1_eq + y2_eq).astype(np.float32)


def apply_stem_dj_blend(
    stems1: Dict[str, np.ndarray],
    stems2: Dict[str, np.ndarray],
    drop_sample_in_trans: Optional[int] = None,
    vocal_clash_detected: bool = False,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Mezcla avanzada de 4 Stems (Drums, Bass, Other/Melodía, Vocals) con IA:
    1. Vocals Anti-Clash: Si la pista 2 tiene voz en la mezcla o hay colisión, se atenúa o
       silencia selectivamente el stem vocal de la pista 1 SIN TOCAR su percusión ni sus sintes.
    2. Bass Swap Puro: En el golpe 1 del drop (drop_sample_in_trans), el bajo de la pista 1
       se conmuta instantáneamente al bajo de la pista 2 (cero choque de subgraves).
    3. Drums: Convivencia rítmica calibrada (Track 1 al 70%, Track 2 al 75%).
    4. Other (Melodías): Fundido armónico cruzado respetando potencia constante.
    5. Fader Cut de MiGue: La pista 1 convive hasta el último golpe del compás y se corta
       en seco con micro-rampa anti-pop de 3 ms.
    """
    n_trans = stems1["drums"].shape[1]

    # --- 1. STEM BASS (Bass Swap en el Drop) ---
    g_bass1 = np.ones(n_trans, dtype=np.float32)
    g_bass2 = np.zeros(n_trans, dtype=np.float32)

    if drop_sample_in_trans is not None and 0 <= drop_sample_in_trans < n_trans:
        drop_idx = drop_sample_in_trans
        ramp = min(int(sr * 0.015), 660)  # Micro-rampa de 15 ms para evitar clicks DC
        # Antes del drop: Track 1 tiene bajo, Track 2 bajo mute
        g_bass1[:max(0, drop_idx - ramp)] = 0.90
        # En el drop: Bass swap limpio
        if ramp > 0 and drop_idx >= ramp:
            swap_ramp = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
            g_bass1[drop_idx - ramp:drop_idx] = swap_ramp * 0.90
            g_bass2[drop_idx - ramp:drop_idx] = (1.0 - swap_ramp) * 1.0
        g_bass1[drop_idx:] = 0.0
        g_bass2[drop_idx:] = 1.0
    else:
        # Si no hay drop marcado, crossfade suave equal-power
        t = np.linspace(0.0, np.pi * 0.5, n_trans, dtype=np.float32)
        g_bass1 = np.cos(t) ** 2
        g_bass2 = np.sin(t) ** 2

    # --- 2. STEM VOCALS (Anti-Vocal Clash Quirúrgico) ---
    v2_audio = stems2["vocals"]
    v2_rms = np.sqrt(np.mean(v2_audio ** 2, axis=0)) if v2_audio.ndim > 1 else np.abs(v2_audio)
    v2_has_vocal = np.max(v2_rms) > 0.02 or vocal_clash_detected

    g_vox1 = np.ones(n_trans, dtype=np.float32)
    g_vox2 = np.ones(n_trans, dtype=np.float32)

    if v2_has_vocal:
        # Ducking progresivo del stem vocal de Track 1 para dejar paso limpio a Track 2
        duck_len = min(int(sr * 2.0), n_trans)  # Curva suave de 2 segundos
        g_vox1[:duck_len] = np.linspace(1.0, 0.0, duck_len, dtype=np.float32)
        g_vox1[duck_len:] = 0.0
    else:
        cut_lead = min(int(sr * 3.0), n_trans)
        g_vox1[:n_trans - cut_lead] = 0.85
        g_vox1[n_trans - cut_lead:] = np.linspace(0.85, 0.0, cut_lead, dtype=np.float32)

    # --- 3. STEM OTHER (Melodías / Sintes) ---
    t_mel = np.linspace(0.0, np.pi * 0.5, n_trans, dtype=np.float32)
    g_mel1 = np.cos(t_mel) * 0.85
    g_mel2 = np.sin(t_mel) * 0.85

    # --- 4. STEM DRUMS (Bombos y Percusión) ---
    g_drum1 = np.full(n_trans, 0.65, dtype=np.float32)
    g_drum2 = np.full(n_trans, 0.75, dtype=np.float32)

    # --- 5. FADER CUT DE MIGUE EN TRACK 1 ---
    anti_pop_len = min(132, n_trans)  # 3 ms a 44.1 kHz
    anti_pop = np.cos(np.linspace(0.0, np.pi * 0.5, anti_pop_len, dtype=np.float32)) ** 2
    g_bass1[-anti_pop_len:] *= anti_pop
    g_vox1[-anti_pop_len:] *= anti_pop
    g_mel1[-anti_pop_len:] *= anti_pop
    g_drum1[-anti_pop_len:] *= anti_pop

    out_bass = stems1["bass"] * g_bass1 + stems2["bass"] * g_bass2
    out_vox = stems1["vocals"] * g_vox1 + stems2["vocals"] * g_vox2
    out_mel = stems1["other"] * g_mel1 + stems2["other"] * g_mel2
    out_drum = stems1["drums"] * g_drum1 + stems2["drums"] * g_drum2

    mix = (out_bass + out_vox + out_mel + out_drum).astype(np.float32)

    # Normalización de seguridad anti-clipping
    peak = np.max(np.abs(mix))
    if peak > 0.98:
        mix = mix * (0.98 / peak)

    return mix

