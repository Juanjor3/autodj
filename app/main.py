import argparse
import datetime
import sys
from pathlib import Path
from typing import List

try:
    from app.analyzer import analyze_audio_file, TrackProfile
    from app.sequencer import optimize_track_sequence
    from app.mixer import mix_session
    from app.generator import generate_demo_track
except ModuleNotFoundError:
    from analyzer import analyze_audio_file, TrackProfile
    from sequencer import optimize_track_sequence
    from mixer import mix_session
    from generator import generate_demo_track


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoDJ - Motor de Creación y Mezcla Continua de Sesiones de Audio"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/app/input"),
        help="Carpeta donde se encuentran las canciones a mezclar",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/output"),
        help="Carpeta donde se guardará la sesión exportada",
    )
    parser.add_argument(
        "--order",
        choices=["auto", "manual"],
        default="auto",
        help="Orden de mezcla: 'auto' (orden armónico y tempo óptimo) o 'manual' (orden alfabético)",
    )
    parser.add_argument(
        "--transition-beats",
        type=int,
        default=64,
        help="Número de compases/beats para la duración de la transición (por defecto 64: 16 compases en 4/4)",
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="Nombre del archivo de audio resultante (por defecto: session_YYYYMMDD_HHMMSS.wav)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generar una sesión demo sintética automáticamente si no hay canciones",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("      🎧  AUTODJ - MOTOR DE MEZCLA CONTINUA DE SESIONES  🎧")
    print("=" * 65)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Buscar archivos de audio compatibles
    audio_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not audio_files:
        print(f"\n[INFO] No se encontraron archivos de audio en '{input_dir}'.")
        print("💡 Generando 3 pistas sintéticas demostrativas en 'input/' para crear tu primera sesión de prueba...")

        generate_demo_track(input_dir / "01_Sunset_Groove_120bpm.wav", "Sunset Groove", bpm=120.0, root_freq=110.0, duration_beats=48)
        generate_demo_track(input_dir / "02_Neon_Night_124bpm.wav", "Neon Night", bpm=124.0, root_freq=130.8, duration_beats=48)
        generate_demo_track(input_dir / "03_Sunrise_Drop_126bpm.wav", "Sunrise Drop", bpm=126.0, root_freq=146.8, duration_beats=48)

        audio_files = sorted([
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

    print(f"\n📂 Pistas encontradas ({len(audio_files)}):")
    for idx, f in enumerate(audio_files, 1):
        print(f"   [{idx}] {f.name}")

    # 1. Análisis musical de cada pista
    print("\n🔍 [1/3] Analizando BPM, beatgrid y tonalidad armónica (Camelot)...")
    profiles: List[TrackProfile] = []
    for f in audio_files:
        if not f.exists():
            continue
        print(f"   • Analizando: {f.name}...", end=" ", flush=True)
        prof = analyze_audio_file(f)
        print(f"BPM: {prof.bpm:.1f} | Key: {prof.camelot} ({prof.key_name}) | Duración: {prof.duration:.1f}s")
        profiles.append(prof)

    # 2. Secuenciación
    if args.order == "auto" and len(profiles) > 1:
        print("\n🔀 [2/3] Optimizando orden armónico y curva de tempo...")
        ordered_tracks = optimize_track_sequence(profiles)
    else:
        print("\n📄 [2/3] Manteniendo orden manual de las pistas...")
        ordered_tracks = profiles

    print("\n📋 Secuencia final de la sesión:")
    for idx, t in enumerate(ordered_tracks, 1):
        print(f"   {idx}. {t.title} - {t.bpm:.1f} BPM [{t.camelot}]")

    # 3. Mezcla continua
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_filename = args.session_name or f"session_{timestamp_str}.wav"
    cue_filename = f"tracklist_{timestamp_str}.txt"

    out_audio_path = output_dir / session_filename
    out_cue_path = output_dir / cue_filename

    print(f"\n🎚️ [3/3] Mezclando sesión (Transición: {args.transition_beats} beats, Bass Swap, Time-stretch)...")
    cue_list = mix_session(
        ordered_tracks,
        output_audio_path=out_audio_path,
        output_cue_path=out_cue_path,
        transition_beats=args.transition_beats
    )

    print("\n" + "=" * 65)
    print("🎉 ¡SESIÓN MEZCLADA Y EXPORTADA CON ÉXITO!")
    print(f"🎵 Audio continuo:  {out_audio_path}")
    print(f"📝 Marcas de tiempo: {out_cue_path}")
    print("=" * 65 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
