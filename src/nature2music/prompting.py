from __future__ import annotations

from dataclasses import dataclass

from .audio_analysis import AudioFeatures
from .schema import Recognition


@dataclass(slots=True)
class MusicPrompt:
    caption: str
    chain_of_thought: str


def build_music_prompt(
    recognition: Recognition,
    features: AudioFeatures | None = None,
    style: str = "cinematic ambient world music",
    preserve_calls: bool = True,
) -> MusicPrompt:
    display_name = recognition.common_name_zh or recognition.species
    background = ", ".join(recognition.background) or "subtle natural ambience"
    bpm = features.estimated_bpm if features and features.estimated_bpm > 0 else 76
    dominant = features.dominant_frequency_hz if features else 0
    caption = f"{style} inspired by the call of {display_name}"
    steps = [
        f"Create a coherent {style} composition at approximately {bpm:.0f} BPM.",
        f"The ecological focus is {display_name} ({recognition.scientific_name or recognition.species}), "
        f"a {recognition.group} vocalization with call type {recognition.call_type or 'unspecified'}.",
        f"Use {background} as a quiet spatial bed and keep the mix spacious, organic, and non-vocal.",
    ]
    if dominant > 0:
        steps.append(
            f"Derive the lead motif contour from the source call near {dominant:.0f} Hz, transposed into a "
            "comfortable musical register; do not turn it into human speech."
        )
    if preserve_calls:
        steps.append(
            "Keep sparse, recognizable bioacoustic call gestures as foreground accents without cloning a "
            "specific copyrighted recording."
        )
    else:
        steps.append("Use only abstract melodic and rhythmic properties; do not reproduce the source call.")
    steps.append("Build a clear introduction, gradual development, and gentle ending with no abrupt cuts.")
    return MusicPrompt(caption=caption, chain_of_thought=" ".join(steps))
