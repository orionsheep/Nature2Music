from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .prompting import MusicPrompt


class ThinkSoundGenerator:
    """Invoke the official ThinkSound two-stage video/text-to-audio pipeline."""

    def __init__(
        self,
        project_dir: str | Path,
        python: str = "python",
        ffmpeg: str = "ffmpeg",
        use_half: bool = True,
        checkpoint: str | Path | None = None,
        lora: bool = False,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        # The ThinkSound subprocesses run with cwd=project_dir, so a relative
        # python path would break; make it absolute up front. abspath (not
        # resolve) is deliberate: .venv/bin/python is a symlink to the base
        # interpreter, and resolving it would escape the venv.
        if os.sep in python or (os.altsep and os.altsep in python):
            python = os.path.abspath(python)
        self.python = python
        self.ffmpeg = ffmpeg
        self.use_half = use_half
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self.lora = lora
        self.predict_script = "predict_lora.py" if lora else "predict.py"
        for required in ("extract_latents.py", self.predict_script):
            if not (self.project_dir / required).is_file():
                raise FileNotFoundError(self.project_dir / required)
        if self.checkpoint and not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)

    @staticmethod
    def _run(command: list[str], cwd: Path | None = None, env: dict | None = None) -> str:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.returncode:
            output = (process.stdout + "\n" + process.stderr)[-8000:]
            raise RuntimeError(f"command failed ({process.returncode}): {' '.join(command)}\n{output}")
        return process.stdout

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        if self.lora:
            environment["N2M_THINKSOUND_LORA"] = "1"
        return environment

    def generate(
        self,
        prompt: MusicPrompt,
        destination: str | Path,
        duration_s: float = 10.0,
    ) -> Path:
        """Generate a music/soundscape WAV using a neutral carrier video plus text conditioning."""

        duration_s = max(2.0, min(30.0, float(duration_s)))
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="nature2music_") as temp_name:
            session = Path(temp_name)
            videos = session / "videos"
            results = session / "results"
            videos.mkdir()
            results.mkdir()
            carrier = videos / "demo.mp4"
            self._run(
                [
                    self.ffmpeg,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=black:s=512x512:r=24:d={duration_s:.3f}",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(carrier),
                ]
            )
            csv_path = session / "cot.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "caption", "caption_cot"])
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "demo",
                        "caption": prompt.caption,
                        "caption_cot": prompt.chain_of_thought,
                    }
                )
            extract = [
                self.python,
                "extract_latents.py",
                "--duration_sec",
                f"{duration_s:.3f}",
                "--root",
                str(videos),
                "--tsv_path",
                str(csv_path),
                "--save-dir",
                str(results),
            ]
            if self.use_half:
                extract.append("--use_half")
            self._run(extract, cwd=self.project_dir, env=self._environment())
            predict = [
                self.python,
                self.predict_script,
                "--duration-sec",
                f"{duration_s:.3f}",
                "--results-dir",
                str(results),
                "--save-dir",
                str(results),
            ]
            if self.checkpoint:
                # ThinkSound expects an unwrapped model state file here. Use its
                # upstream unwrap.py first when starting from a Lightning checkpoint.
                predict.extend(["--ckpt-dir", str(self.checkpoint)])
            self._run(predict, cwd=self.project_dir, env=self._environment())
            candidates = sorted(results.rglob("demo.wav"), key=lambda item: item.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError(f"ThinkSound produced no demo.wav under {results}")
            shutil.copy2(candidates[-1], target)
        return target
