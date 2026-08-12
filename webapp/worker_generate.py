"""Run one Nature2Music music-generation job as a subprocess.

Reads a job JSON (recognition + audio features + style/duration + optional
user-edited prompt + backend selection), runs the chosen generator
(Google Lyria 3 by default, ThinkSound as a local alternative), writes the
WAV and a pipeline report. Progress is reported to the parent process as
``N2M_STAGE:<name>`` stdout lines so the web server can expose staged
progress.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from nature2music.audio_analysis import AudioFeatures
from nature2music.prompting import MusicPrompt, build_music_prompt
from nature2music.schema import Recognition


def stage(name: str) -> None:
    print(f"N2M_STAGE:{name}", flush=True)


def run_lyria(
    prompt: MusicPrompt,
    output_audio: Path,
    duration_s: float,
    recognition: Recognition,
    features: AudioFeatures,
    user_edited: bool,
) -> tuple[Path, str | None]:
    from nature2music.lyria_backend import LyriaGenerator

    stage("preparing")
    generator = LyriaGenerator()
    stage("sampling")
    generated = generator.generate(
        prompt,
        output_audio,
        duration_s=duration_s,
        recognition=recognition,
        features=features,
        user_edited=user_edited,
    )
    return generated, generator.last_prompt


def run_thinksound(
    prompt: MusicPrompt,
    output_audio: Path,
    duration_s: float,
    thinksound_dir: str,
    python: str,
) -> Path:
    from nature2music.thinksound_backend import ThinkSoundGenerator

    # Emit stage markers around the ThinkSound internal steps. The generator
    # shells out to ffmpeg (carrier), extract_latents.py, then predict.py.
    original_run = ThinkSoundGenerator._run

    def staged_run(command, cwd=None, env=None):  # noqa: ANN001, ANN202
        joined = " ".join(str(part) for part in command)
        if "extract_latents" in joined:
            stage("extracting")
        elif "predict" in joined:
            stage("sampling")
        else:
            stage("preparing")
        return original_run(command, cwd=cwd, env=env)

    ThinkSoundGenerator._run = staticmethod(staged_run)

    generator = ThinkSoundGenerator(thinksound_dir, python=python)
    stage("preparing")
    return generator.generate(prompt, output_audio, duration_s=duration_s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Path to the job JSON file")
    parser.add_argument("--thinksound-dir", help="ThinkSound checkout (thinksound backend only)")
    parser.add_argument("--python", help="ThinkSound venv python (thinksound backend only)")
    args = parser.parse_args()

    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    backend = str(job.get("backend") or "lyria")
    recognition = Recognition(
        **{key: val for key, val in job["recognition"].items() if key in Recognition.__dataclass_fields__}
    )
    features = AudioFeatures(
        **{key: val for key, val in job["audio_features"].items() if key in AudioFeatures.__dataclass_fields__}
    )
    style = str(job.get("style") or "cinematic ambient world music")
    duration_s = float(job.get("duration_s") or 10.0)

    prompt = build_music_prompt(recognition, features=features, style=style)
    # Allow the user-edited prompt from the parameter panel to win.
    if job.get("caption"):
        prompt.caption = str(job["caption"])
    if job.get("chain_of_thought"):
        prompt.chain_of_thought = str(job["chain_of_thought"])

    output_audio = Path(job["output_audio"])
    lyria_prompt = None
    if backend == "thinksound":
        if not args.thinksound_dir or not args.python:
            raise SystemExit("thinksound backend requires --thinksound-dir and --python")
        generated = run_thinksound(prompt, output_audio, duration_s, args.thinksound_dir, args.python)
    elif backend == "lyria":
        generated, lyria_prompt = run_lyria(
            prompt,
            output_audio,
            duration_s,
            recognition,
            features,
            user_edited=bool(job.get("caption") or job.get("chain_of_thought")),
        )
    else:
        raise SystemExit(f"unknown backend: {backend}")

    stage("finalizing")
    report = {
        "input_audio": job.get("input_audio", ""),
        "output_audio": str(generated),
        "backend": backend,
        "lyria_prompt": lyria_prompt,
        "recognition": recognition.to_dict(),
        "audio_features": features.to_dict(),
        "prompt": asdict(prompt),
        "style": style,
        "duration_s": duration_s,
    }
    report_path = Path(job["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    stage("done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # surface a clean error line for the server log
        print(f"N2M_ERROR:{exc}", flush=True)
        raise
