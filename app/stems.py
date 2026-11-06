import os
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

# Intentar importar PyTorch y Demucs
try:
    import torch
    import demucs.api
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False


class StemSeparator:
    """
    Motor de separación de fuentes con IA (Meta Demucs HTDemucs).
    Aísla:
    - drums: Bombos, cajas, platos y percusión rítmica.
    - bass: Líneas de bajo sintético y subgraves.
    - other: Sintetizadores, leads, pianos, pads y efectos melódicos.
    - vocals: Voces solistas, coros y cantaditas.

    Optimizado para CPU mediante inferencia segmentada y lazy-loading.
    """
    _instance: Optional["StemSeparator"] = None
    _separator = None

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            base_dir = Path("/app/output") if Path("/app/output").exists() else Path(__file__).resolve().parent.parent / "output"
            cache_dir = base_dir / ".stem_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls) -> "StemSeparator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_model(self):
        if not DEMUCS_AVAILABLE:
            return None
        if self._separator is None:
            print("[Stems] Inicializando modelo HTDemucs en CPU (shifts=0 para máxima velocidad)...")
            self._separator = demucs.api.Separator(
                model="htdemucs",
                device="cpu",
                segment=7,
                shifts=0,
                overlap=0.1,
                split=True,
                progress=False
            )
            print("[Stems] Modelo HTDemucs listo para inferencia.")
        return self._separator

    def separate_slice(
        self,
        audio_stereo: np.ndarray,
        sr: int = 44100,
        cache_id: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Separa un fragmento de audio estéreo (2, N) en 4 stems.
        Si cache_id está definido y existe en caché, devuelve el resultado al instante.
        """
        if cache_id:
            cache_file = self.cache_dir / f"{cache_id}.npz"
            if cache_file.exists():
                try:
                    data = np.load(cache_file)
                    return {
                        "drums": data["drums"],
                        "bass": data["bass"],
                        "other": data["other"],
                        "vocals": data["vocals"]
                    }
                except Exception as e:
                    print(f"[Stems] Error leyendo caché {cache_file}: {e}")

        # Si Demucs está disponible, usar la red neuronal
        model = self._get_model()
        if model is not None:
            try:
                # Asegurar estéreo float32
                if audio_stereo.ndim == 1:
                    audio_stereo = np.vstack([audio_stereo, audio_stereo])
                audio_stereo = audio_stereo.astype(np.float32)

                tensor = torch.from_numpy(audio_stereo)
                _, separated = model.separate_tensor(tensor)

                stems = {
                    "drums": separated["drums"].cpu().numpy().astype(np.float32),
                    "bass": separated["bass"].cpu().numpy().astype(np.float32),
                    "other": separated["other"].cpu().numpy().astype(np.float32),
                    "vocals": separated["vocals"].cpu().numpy().astype(np.float32)
                }

                # Guardar en caché persistente
                if cache_id:
                    np.savez_compressed(
                        self.cache_dir / f"{cache_id}.npz",
                        drums=stems["drums"],
                        bass=stems["bass"],
                        other=stems["other"],
                        vocals=stems["vocals"]
                    )

                gc.collect()
                return stems
            except Exception as e:
                print(f"[Stems] Inferencia Demucs falló ({e}), usando fallback 3-Band...")

        # Fallback analítico si no hay Demucs
        return self._fallback_filter_stems(audio_stereo, sr)

    def _fallback_filter_stems(self, audio_stereo: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Fallback matemático de alta fidelidad basado en filtrado por bandas si Demucs no está instalado."""
        import scipy.signal
        if audio_stereo.ndim == 1:
            audio_stereo = np.vstack([audio_stereo, audio_stereo])

        sos_sub = scipy.signal.butter(4, 120, btype="lowpass", fs=sr, output="sos")
        sos_drums = scipy.signal.butter(4, [80, 250], btype="bandpass", fs=sr, output="sos")
        sos_mid = scipy.signal.butter(4, [250, 3200], btype="bandpass", fs=sr, output="sos")
        sos_high = scipy.signal.butter(4, 3200, btype="highpass", fs=sr, output="sos")

        bass = scipy.signal.sosfiltfilt(sos_sub, audio_stereo, axis=1)
        drums_low = scipy.signal.sosfiltfilt(sos_drums, audio_stereo, axis=1)
        highs = scipy.signal.sosfiltfilt(sos_high, audio_stereo, axis=1)
        mids = scipy.signal.sosfiltfilt(sos_mid, audio_stereo, axis=1)

        drums = (drums_low + 0.6 * highs).astype(np.float32)
        other = (0.7 * mids + 0.4 * highs).astype(np.float32)
        vocals = (0.5 * mids).astype(np.float32)

        return {
            "drums": drums,
            "bass": bass.astype(np.float32),
            "other": other,
            "vocals": vocals
        }
