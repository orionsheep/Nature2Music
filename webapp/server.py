"""Nature2Music local web server.

FastAPI wrapper around the existing CLI pipeline:
- POST /api/identify        upload audio -> recognition + audio features (sync, ~15s)
- POST /api/generate        start an async generation task (subprocess, one at a time)
- GET/DELETE /api/tasks/{id}  query / cancel a task
- GET /api/results/{id}/audio|report  download generated WAV / JSON report
- GET/DELETE /api/history[/{id}]      persisted generation history
- GET /api/samples          bundled example clips
- POST /api/preview-prompt  rebuild the music prompt for a style

Runs in the project main venv; binds 127.0.0.1:8321 and serves webapp/static/.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBAPP_DIR / "static"
OUT_DIR = PROJECT_ROOT / "outputs" / "webapp"
UPLOAD_DIR = OUT_DIR / "uploads"
RESULTS_DIR = OUT_DIR / "results"
JOBS_DIR = OUT_DIR / "jobs"
HISTORY_PATH = OUT_DIR / "history.json"

VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
CLI = PROJECT_ROOT / ".venv" / "bin" / "nature2music"
ASR_MODEL = PROJECT_ROOT / "runs" / "funasr-bioacoustic"
ASR_REMOTE_CODE = PROJECT_ROOT / "external" / "Fun-ASR" / "model_lora.py"
THINKSOUND_DIR = PROJECT_ROOT / "external" / "ThinkSound"
THINKSOUND_PYTHON = THINKSOUND_DIR / ".venv" / "bin" / "python"

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
IDENTIFY_TIMEOUT_S = 300
HISTORY_LIMIT = 20

SAMPLES = [
    {"id": "crow", "name_zh": "乌鸦", "species": "crow", "url": "/samples/crow.wav"},
    {"id": "crickets", "name_zh": "蟋蟀", "species": "crickets", "url": "/samples/crickets.wav"},
    {"id": "rain", "name_zh": "雨声", "species": "rain", "url": "/samples/rain.wav"},
    {"id": "thunderstorm", "name_zh": "雷雨", "species": "thunderstorm", "url": "/samples/thunderstorm.wav"},
    {"id": "frog", "name_zh": "蛙鸣", "species": "frog", "url": "/samples/frog.wav"},
]

STAGE_LABELS = {
    "pending": "排队中",
    "preparing": "提交创作描述",
    "extracting": "整理生成条件",
    "sampling": "云端作曲生成（约 20-40 秒）",
    "finalizing": "音频转码与写出",
    "done": "完成",
    "failed": "失败",
    "cancelled": "已取消",
}

# task_id -> task dict; tasks survive until the server restarts.
tasks: dict[str, dict] = {}
_active_generate_id: str | None = None


def _load_history() -> list[dict]:
    if HISTORY_PATH.is_file():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_history(history: list[dict]) -> None:
    HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )


def _add_history(entry: dict) -> None:
    history = _load_history()
    history.insert(0, entry)
    dropped = history[HISTORY_LIMIT:]
    history = history[:HISTORY_LIMIT]
    _save_history(history)
    for old in dropped:
        shutil.rmtree(RESULTS_DIR / old["id"], ignore_errors=True)


def _result_dir(task_id: str) -> Path:
    # Guard against path traversal in {id} path params.
    if not task_id or any(c not in "0123456789abcdef-" for c in task_id):
        raise HTTPException(status_code=404, detail="结果不存在")
    directory = RESULTS_DIR / task_id
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="结果不存在")
    return directory


async def _convert_to_wav(source: Path, target: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", str(source), "-ar", "44100", str(target),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0 or not target.is_file():
        detail = stderr.decode("utf-8", "replace")[-500:]
        raise HTTPException(status_code=422, detail=f"音频解码失败，请检查文件格式：{detail}")


async def _run_identify(wav_path: Path) -> dict:
    output_json = wav_path.with_suffix(".recognition.json")
    command = [
        str(CLI), "identify", str(wav_path),
        "--model", str(ASR_MODEL),
        "--remote-code", str(ASR_REMOTE_CODE),
        "--output", str(output_json),
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=PROJECT_ROOT,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=IDENTIFY_TIMEOUT_S)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise HTTPException(status_code=504, detail="识别超时，请重试") from None
    if process.returncode != 0 or not output_json.is_file():
        detail = stderr.decode("utf-8", "replace")[-800:]
        raise HTTPException(status_code=500, detail=f"识别失败：{detail}")
    return json.loads(output_json.read_text(encoding="utf-8"))


def _analyze(wav_path: Path) -> dict:
    from nature2music.audio_analysis import analyze_audio

    return analyze_audio(wav_path).to_dict()


async def _watch_generate(task_id: str, process: asyncio.subprocess.Process) -> None:
    """Consume worker stdout stage markers and finalize the task."""
    global _active_generate_id
    task = tasks[task_id]
    try:
        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if text.startswith("N2M_STAGE:"):
                task["stage"] = text.split(":", 1)[1]
            elif text.startswith("N2M_ERROR:"):
                task["error"] = text.split(":", 1)[1]
        returncode = await process.wait()
        result_dir = RESULTS_DIR / task_id
        audio_path = result_dir / "audio.wav"
        report_path = result_dir / "report.json"
        if task["stage"] == "cancelled":
            pass  # cancelled via DELETE; leave state as-is
        elif returncode == 0 and audio_path.is_file() and report_path.is_file():
            task["stage"] = "done"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            recognition = report.get("recognition", {})
            _add_history(
                {
                    "id": task_id,
                    "species": recognition.get("species", ""),
                    "common_name_zh": recognition.get("common_name_zh", ""),
                    "group": recognition.get("group", ""),
                    "style": task["style"],
                    "duration_s": task["duration_s"],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        else:
            task["stage"] = "failed"
            if not task.get("error"):
                task["error"] = f"生成进程退出码 {returncode}"
    finally:
        if _active_generate_id == task_id:
            _active_generate_id = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    for directory in (UPLOAD_DIR, RESULTS_DIR, JOBS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Nature2Music", lifespan=_lifespan)


@app.post("/api/identify")
async def identify(file: UploadFile = File(...)):
    original_name = file.filename or "audio"
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的格式 {extension or '(无扩展名)'}，请使用 WAV / MP3 / FLAC / OGG / M4A",
        )
    upload_id = uuid.uuid4().hex[:12]
    raw_path = UPLOAD_DIR / f"{upload_id}.raw{extension}"
    size = 0
    with raw_path.open("wb") as handle:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                handle.close()
                raw_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="文件超过 50MB 限制")
            handle.write(chunk)
    wav_path = UPLOAD_DIR / f"{upload_id}.wav"
    await _convert_to_wav(raw_path, wav_path)
    if raw_path != wav_path:
        raw_path.unlink(missing_ok=True)

    recognition = await _run_identify(wav_path)
    loop = asyncio.get_running_loop()
    features = await loop.run_in_executor(None, _analyze, wav_path)
    return {
        "upload_id": upload_id,
        "filename": original_name,
        "recognition": recognition,
        "audio_features": features,
    }


class PreviewRequest(BaseModel):
    recognition: dict
    audio_features: dict | None = None
    style: str = "cinematic ambient world music"


@app.post("/api/preview-prompt")
async def preview_prompt(request: PreviewRequest):
    from nature2music.audio_analysis import AudioFeatures
    from nature2music.prompting import build_music_prompt
    from nature2music.schema import Recognition

    recognition = Recognition(
        **{k: v for k, v in request.recognition.items() if k in Recognition.__dataclass_fields__}
    )
    features = None
    if request.audio_features:
        features = AudioFeatures(
            **{
                k: v
                for k, v in request.audio_features.items()
                if k in AudioFeatures.__dataclass_fields__
            }
        )
    prompt = build_music_prompt(recognition, features=features, style=request.style)
    return {"caption": prompt.caption, "chain_of_thought": prompt.chain_of_thought}


class GenerateRequest(BaseModel):
    recognition: dict
    audio_features: dict
    style: str = "cinematic ambient world music"
    duration_s: float = 10.0
    caption: str | None = None
    chain_of_thought: str | None = None
    input_audio: str = ""
    backend: str = "lyria"


@app.post("/api/generate", status_code=202)
async def generate(request: GenerateRequest):
    global _active_generate_id
    if _active_generate_id is not None and tasks.get(_active_generate_id, {}).get("stage") not in (
        "done",
        "failed",
        "cancelled",
    ):
        raise HTTPException(status_code=409, detail="已有生成任务正在进行，请等待完成或先取消")

    duration_s = max(2.0, min(30.0, float(request.duration_s)))
    backend = request.backend.strip().lower() or "lyria"
    if backend not in ("lyria", "thinksound"):
        raise HTTPException(status_code=400, detail=f"不支持的后端 {backend}，可选：lyria / thinksound")
    task_id = uuid.uuid4().hex
    result_dir = RESULTS_DIR / task_id
    result_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "recognition": request.recognition,
        "audio_features": request.audio_features,
        "style": request.style,
        "duration_s": duration_s,
        "caption": request.caption,
        "chain_of_thought": request.chain_of_thought,
        "input_audio": request.input_audio,
        "backend": backend,
        "output_audio": str(result_dir / "audio.wav"),
        "report_path": str(result_dir / "report.json"),
    }
    job_path = JOBS_DIR / f"{task_id}.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    log_path = result_dir / "worker.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = await asyncio.create_subprocess_exec(
        str(VENV_PYTHON), str(WEBAPP_DIR / "worker_generate.py"),
        "--job", str(job_path),
        "--thinksound-dir", str(THINKSOUND_DIR),
        "--python", str(THINKSOUND_PYTHON),
        cwd=PROJECT_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=log_handle,
        start_new_session=True,  # own process group so cancel kills ThinkSound children too
    )
    tasks[task_id] = {
        "id": task_id,
        "stage": "pending",
        "stage_label": STAGE_LABELS["pending"],
        "style": request.style,
        "duration_s": duration_s,
        "backend": backend,
        "error": None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "process": process,
        "log_handle": log_handle,
    }
    _active_generate_id = task_id
    asyncio.create_task(_watch_generate(task_id, process))
    return {"task_id": task_id}


def _task_view(task: dict) -> dict:
    return {
        "id": task["id"],
        "stage": task["stage"],
        "stage_label": STAGE_LABELS.get(task["stage"], task["stage"]),
        "style": task["style"],
        "duration_s": task["duration_s"],
        "backend": task.get("backend", "lyria"),
        "error": task.get("error"),
        "created_at": task["created_at"],
    }


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _task_view(task)


@app.delete("/api/tasks/{task_id}")
async def cancel_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["stage"] in ("done", "failed", "cancelled"):
        return _task_view(task)
    task["stage"] = "cancelled"
    process: asyncio.subprocess.Process = task["process"]
    if process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return _task_view(task)


@app.get("/api/results/{task_id}/audio")
async def result_audio(task_id: str):
    directory = _result_dir(task_id)
    audio = directory / "audio.wav"
    if not audio.is_file():
        raise HTTPException(status_code=404, detail="音频不存在")
    return FileResponse(audio, media_type="audio/wav", filename=f"nature2music-{task_id[:8]}.wav")


@app.get("/api/results/{task_id}/report")
async def result_report(task_id: str):
    directory = _result_dir(task_id)
    report = directory / "report.json"
    if not report.is_file():
        raise HTTPException(status_code=404, detail="报告不存在")
    return FileResponse(report, media_type="application/json", filename=f"nature2music-{task_id[:8]}.json")


@app.get("/api/history")
async def list_history():
    return _load_history()


@app.delete("/api/history/{entry_id}")
async def delete_history(entry_id: str):
    history = _load_history()
    remaining = [entry for entry in history if entry["id"] != entry_id]
    if len(remaining) == len(history):
        raise HTTPException(status_code=404, detail="记录不存在")
    _save_history(remaining)
    shutil.rmtree(RESULTS_DIR / entry_id, ignore_errors=True)
    return {"ok": True}


@app.get("/api/samples")
async def list_samples():
    return SAMPLES


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8321, log_level="info")
