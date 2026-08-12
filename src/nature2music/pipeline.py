from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .audio_analysis import AudioFeatures, analyze_audio
from .funasr_backend import FunASRRecognizer
from .prompting import MusicPrompt, build_music_prompt
from .schema import Recognition
from .thinksound_backend import ThinkSoundGenerator


@dataclass(slots=True)
class PipelineReport:
    input_audio: str
    output_audio: str
    recognition: Recognition
    audio_features: AudioFeatures
    prompt: MusicPrompt

    def to_dict(self) -> dict:
        return {
            "input_audio": self.input_audio,
            "output_audio": self.output_audio,
            "recognition": self.recognition.to_dict(),
            "audio_features": self.audio_features.to_dict(),
            "prompt": asdict(self.prompt),
        }


def run_pipeline(
    input_audio: str | Path,
    output_audio: str | Path,
    recognizer: FunASRRecognizer,
    generator: ThinkSoundGenerator,
    style: str = "cinematic ambient world music",
    duration_s: float | None = None,
    report_path: str | Path | None = None,
) -> PipelineReport:
    source = Path(input_audio).resolve()
    recognition = recognizer.recognize(source)
    features = analyze_audio(source)
    prompt = build_music_prompt(recognition, features=features, style=style)
    duration = duration_s if duration_s is not None else min(20.0, max(5.0, features.duration_s))
    generated = generator.generate(prompt, output_audio, duration_s=duration)
    report = PipelineReport(
        input_audio=str(source),
        output_audio=str(generated),
        recognition=recognition,
        audio_features=features,
        prompt=prompt,
    )
    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
    return report

