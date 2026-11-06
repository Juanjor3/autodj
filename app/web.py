import os
import shutil
import threading
import asyncio
import datetime
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

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


app = FastAPI(title="AutoDJ Studio API")

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = Path("/app/input") if Path("/app/input").exists() else BASE_DIR.parent / "input"
OUTPUT_DIR = Path("/app/output") if Path("/app/output").exists() else BASE_DIR.parent / "output"
TEMPLATES_DIR = BASE_DIR / "templates"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"}

# Caché en memoria para evitar reanalizar audio innecesariamente
track_cache: Dict[str, dict] = {}

# Estado de la mezcla actual
mix_state = {
    "progress": 0,
    "logs": [],
    "done": False,
    "error": None,
    "result": None,
}


class MixRequest(BaseModel):
    filenames: List[str]
    transition_beats: int = 64
    session_name: Optional[str] = None
    use_stems: bool = True


class AutoOrderRequest(BaseModel):
    tracks: List[dict]


def get_track_metadata(file_path: Path) -> dict:
    """Obtiene y cachea los metadatos musicales de un archivo."""
    key = f"{file_path.name}_{file_path.stat().st_mtime}"
    if key in track_cache:
        return track_cache[key]

    prof = analyze_audio_file(file_path, load_audio=False)
    meta = {
        "filename": file_path.name,
        "title": file_path.stem,
        "duration": prof.duration,
        "bpm": prof.bpm,
        "camelot": prof.camelot,
        "key_name": prof.key_name,
    }
    track_cache[key] = meta
    return meta


from fastapi.concurrency import run_in_threadpool


@app.get("/", response_class=HTMLResponse)
def get_index():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Plantilla index.html no encontrada")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/tracks")
def list_tracks():
    tracks = []
    for f in sorted(INPUT_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                meta = get_track_metadata(f)
                tracks.append(meta)
            except Exception as e:
                print(f"Error analizando {f.name}: {e}")
    return tracks


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_meta = []
    for file in files:
        safe_filename = Path(file.filename).name
        dest_path = INPUT_DIR / safe_filename
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            meta = await run_in_threadpool(get_track_metadata, dest_path)
            uploaded_meta.append(meta)
        except Exception as e:
            print(f"Error analizando {safe_filename}: {e}")
    return uploaded_meta


@app.delete("/api/tracks/{filename}")
async def delete_track(filename: str):
    target = INPUT_DIR / filename
    if target.exists():
        target.unlink()
        return {"status": "deleted", "filename": filename}
    raise HTTPException(status_code=404, detail="Pista no encontrada")


@app.delete("/api/tracks")
async def clear_tracks():
    for f in INPUT_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            f.unlink()
    track_cache.clear()
    return {"status": "all_cleared"}


@app.post("/api/demo-tracks")
async def generate_demos():
    generate_demo_track(INPUT_DIR / "01_Sunset_Groove_120bpm.wav", "Sunset Groove", bpm=120.0, root_freq=110.0, duration_beats=48)
    generate_demo_track(INPUT_DIR / "02_Neon_Night_124bpm.wav", "Neon Night", bpm=124.0, root_freq=130.8, duration_beats=48)
    generate_demo_track(INPUT_DIR / "03_Sunrise_Drop_126bpm.wav", "Sunrise Drop", bpm=126.0, root_freq=146.8, duration_beats=48)
    return {"status": "demos_created"}


@app.post("/api/auto-order")
async def auto_order(req: AutoOrderRequest):
    profiles = []
    for item in req.tracks:
        f_path = INPUT_DIR / item["filename"]
        if f_path.exists():
            profiles.append(analyze_audio_file(f_path))

    if len(profiles) <= 1:
        return req.tracks

    ordered = optimize_track_sequence(profiles)
    result = []
    for p in ordered:
        result.append(get_track_metadata(p.path))
    return result


def _run_mix_worker(filenames: List[str], transition_beats: int, session_name: Optional[str], use_stems: bool = True):
    global mix_state
    try:
        mix_state["progress"] = 10
        mix_state["logs"].append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Cargando {len(filenames)} pistas de audio...")

        profiles = []
        for i, fname in enumerate(filenames):
            p_path = INPUT_DIR / fname
            if not p_path.exists():
                raise FileNotFoundError(f"No se encontró: {fname}")
            mix_state["logs"].append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Analizando beatgrid de: {fname}...")
            prof = analyze_audio_file(p_path)
            profiles.append(prof)
            mix_state["progress"] = 10 + int(30 * (i + 1) / len(filenames))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_wav_name = session_name if session_name and session_name.endswith(".wav") else f"{session_name or 'session_' + timestamp}.wav"
        out_cue_name = f"tracklist_{Path(out_wav_name).stem}.txt"

        out_audio_path = OUTPUT_DIR / out_wav_name
        out_cue_path = OUTPUT_DIR / out_cue_name

        stem_mode_str = "con Separación IA de Stems" if use_stems else "con Isolator 3-bandas"
        mix_state["logs"].append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Iniciando mezcla ({stem_mode_str}) con {transition_beats} beats de transición...")
        mix_state["progress"] = 55

        def _on_mix_log(msg: str):
            clean_msg = msg.strip()
            if clean_msg:
                mix_state["logs"].append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {clean_msg}")

        def _on_mix_progress(pct: int, msg: str):
            mix_state["progress"] = pct
            if msg:
                mix_state["logs"].append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⚡ {msg}")

        cues = mix_session(
            profiles,
            output_audio_path=out_audio_path,
            output_cue_path=out_cue_path,
            transition_beats=transition_beats,
            log_callback=_on_mix_log,
            progress_callback=_on_mix_progress,
            use_stems=use_stems
        )

        mix_state["progress"] = 90
        mix_state["logs"].append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Masterización final WAV completada sin clipping.")

        # Codificación MP3 de alta fidelidad a 256 kbps con FFmpeg
        out_mp3_name = f"{Path(out_wav_name).stem}.mp3"
        out_mp3_path = OUTPUT_DIR / out_mp3_name
        mix_state["logs"].append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Codificando sesión en MP3 a 256 kbps (Calidad HQ)...")
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(out_audio_path),
                "-codec:a", "libmp3lame",
                "-b:a", "256k",
                str(out_mp3_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            mix_state["logs"].append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Archivo MP3 256 kbps generado exitosamente.")
        except Exception as err_mp3:
            mix_state["logs"].append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Error codificando MP3: {err_mp3}")
            out_mp3_name = None

        info = sf.info(out_audio_path)
        mix_state["progress"] = 100
        mix_state["logs"].append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Sesión finalizada exitosamente ({info.duration:.1f}s).")

        cues_data = [
            {
                "index": c.index,
                "title": c.title,
                "start_seconds": c.start_seconds,
                "format_timestamp": c.format_timestamp(),
                "bpm": round(c.bpm, 1),
                "camelot": c.camelot,
                "key_name": c.key_name
            }
            for c in cues
        ]

        mix_state["result"] = {
            "audio_filename": out_wav_name,
            "mp3_filename": out_mp3_name,
            "cue_filename": out_cue_name,
            "duration_seconds": info.duration,
            "cues": cues_data
        }
        mix_state["done"] = True

    except Exception as e:
        mix_state["error"] = str(e)
        mix_state["logs"].append(f"[ERROR] {str(e)}")
        mix_state["done"] = True


@app.post("/api/mix")
async def trigger_mix(req: MixRequest):
    global mix_state
    mix_state = {
        "progress": 0,
        "logs": [],
        "done": False,
        "error": None,
        "result": None,
    }

    thread = threading.Thread(
        target=_run_mix_worker,
        args=(req.filenames, req.transition_beats, req.session_name, req.use_stems),
        daemon=True
    )
    thread.start()
    return {"status": "mix_started"}


@app.get("/api/status")
async def get_mix_status():
    return mix_state


@app.api_route("/api/download/session/{filename}", methods=["GET", "HEAD"])
async def download_session(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Archivo de sesión no encontrado")
    media_type = "audio/mpeg" if filename.lower().endswith(".mp3") else "audio/wav"
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename
    )


@app.api_route("/api/download/tracklist/{filename}", methods=["GET", "HEAD"])
async def download_cue(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Archivo de tracklist no encontrado")
    return FileResponse(
        path=str(file_path),
        media_type="text/plain",
        filename=filename
    )


PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
CUBE_STATE_PATH = PROFILES_DIR / "cube_learn_state.json"
CUBE_MODEL_PATH = PROFILES_DIR / "cube_style_model.json"


@app.get("/api/cube/status")
async def get_cube_status():
    state = {"status": "idle", "current_step": "Listo para aprender"}
    state_file = Path("/app/profiles/cube_learn_state.json")
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            state["error"] = str(e)

    model = {}
    model_file = Path("/app/profiles/cube_style_model.json")
    if model_file.exists():
        try:
            with open(model_file, "r", encoding="utf-8") as f:
                model = json.load(f)
        except Exception:
            pass

    return {
        "state": state,
        "model": model
    }


@app.get("/api/stream/{filename}")
async def stream_input_audio(filename: str):
    file_path = INPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio no encontrado")
    return FileResponse(path=str(file_path))


@app.get("/api/stream/output/{filename}")
async def stream_output_audio(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio no encontrado")
    return FileResponse(path=str(file_path))


@app.get("/api/design")
async def get_design_system():
    """Retorna la especificación RICOUI DESIGN.md canónica para AutoDJ Studio."""
    design_file = BASE_DIR / "DESIGN.md"
    if not design_file.exists():
        design_file = BASE_DIR.parent / "DESIGN.md"
    if not design_file.exists():
        raise HTTPException(status_code=404, detail="DESIGN.md no encontrado")
    
    content = design_file.read_text(encoding="utf-8")
    return {
        "status": "ok",
        "standard": "RICOUI DESIGN.md",
        "theme": "dark",
        "markdown": content
    }

