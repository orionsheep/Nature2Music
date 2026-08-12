"""Google Lyria 3 Pro (Gemini API) music-generation backend.

Sends the condensed text prompt to the ``lyria-3-pro-preview`` model — the
strongest music model exposed on the Gemini API (Lyria 3.5 is newer but
only ships inside Google Flow Music, not the API). The model returns an
MP3 clip (base64 inline data), which is then trimmed to the requested
duration and converted to the project's standard 44.1 kHz stereo WAV with
ffmpeg.

Only the text prompt leaves the machine; the uploaded audio never does.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .prompting import MusicPrompt

DEFAULT_MODEL = "lyria-3-pro-preview"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# Shared local convention from the gemini-image-global skill.
_GEMINI_ENV_FILE = Path.home() / ".agents" / "skills" / "gemini-image-global" / "config" / "gemini.env"

# Lyria 3 Clip always generates a fixed ~30 s clip.
# Lyria 3 Pro returns a full song segment (observed ~30-64 s); we trim to the
# requested duration, which the UI caps at 30 s anyway.
_CLIP_SECONDS = 30.0


def resolve_api_key(explicit: str | None = None) -> str:
    """Resolve a Gemini API key: explicit arg > env vars > shared gemini.env file."""
    if explicit:
        return explicit
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if _GEMINI_ENV_FILE.is_file():
        for line in _GEMINI_ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and value.strip():
                return value.strip().strip("'\"")
    raise RuntimeError(
        "未找到 Gemini API key：请设置 GEMINI_API_KEY 环境变量，"
        f"或在 {_GEMINI_ENV_FILE} 中配置"
    )


def condense_prompt(prompt: MusicPrompt) -> str:
    """Compress caption + chain_of_thought into one Lyria-friendly text prompt.

    Lyria takes a single text; keep the caption (style + species imagery),
    pull the BPM out of the CoT, and pin the output to instrumental music.
    """
    text = prompt.caption.strip().rstrip(".")
    cot = prompt.chain_of_thought or ""
    match = re.search(r"(\d+(?:\.\d+)?)\s*BPM", cot, re.IGNORECASE)
    if match:
        bpm = float(match.group(1))
        text += f", {bpm:.0f} BPM"
    text += ". Instrumental only, no vocals, clear musical structure with a gentle ending."
    return text


# ---------------------------------------------------------------------------
# Acoustic-feature → musical-language translation.
# ThinkSound could ingest a long CoT, but Lyria takes one short paragraph, so
# the source audio's character must be spelled out as concrete musical
# directions or the output drifts to generic ambient.
# ---------------------------------------------------------------------------

# How each source sound's call/surface character maps to a musical gesture.
_SOURCE_GESTURES = {
    "crow": ("harsh, rhythmic caws", "short staccato pizzicato strings and a repetitive two-note motif"),
    "chirping_birds": ("bright, rapid chirps", "light flute trills and high pizzicato answering phrases"),
    "rooster": ("a piercing dawn crow", "a bold rising brass call answered by warm strings"),
    "hen": ("soft, busy clucking", "gentle staccato woodwind chatter over a calm pulse"),
    "cat": ("drawn-out, pleading meows", "sliding cello glissandi and a yearning violin line"),
    "dog": ("sharp, insistent barks", "percussive hits and a low, alert rhythmic pulse"),
    "sheep": ("trembling bleats", "vibrato reed motifs with a wobbling, pastoral feel"),
    "frog": ("rhythmic croaks", "a bouncing bass ostinato with syncopated hops"),
    "insects": ("dense high-frequency buzzing", "shimmering tremolo strings and a humming high drone"),
    "crickets": ("steady, pulsing night chirps", "delicate mallet patterns and tremolo strings in a slow sway"),
    "rain": ("steady rainfall", "soft arpeggios falling like drops over a calm pad"),
    "thunderstorm": ("rolling thunder over rain", "deep timpani rolls, dramatic swells and rain-like arpeggios"),
    "sea_waves": ("slow surf", "slow wave-like swells that rise and recede"),
    "wind": ("gusting wind", "airy pads and sweeping glissandi that build and release"),
    "crackling_fire": ("crackling fire", "warm plucked textures with sparkling, irregular percussion"),
}


def _register_descriptor(freq_hz: float) -> str:
    if freq_hz <= 0:
        return "a comfortable middle register"
    if freq_hz < 150:
        return "the deep bass register"
    if freq_hz < 400:
        return "a low cello register"
    if freq_hz < 900:
        return "a warm alto register"
    if freq_hz < 2000:
        return "a singing violin/flute register"
    return "a high flute register"


def _brightness_descriptor(centroid_hz: float) -> str:
    if centroid_hz <= 0:
        return "balanced"
    if centroid_hz < 1500:
        return "warm and dark"
    if centroid_hz < 4000:
        return "balanced and natural"
    return "bright and airy"


def _dynamics_descriptor(rms: float) -> str:
    if rms <= 0:
        return "moderate"
    if rms < 0.03:
        return "delicate and quiet"
    if rms < 0.12:
        return "moderate"
    return "powerful and full"


def compose_lyria_prompt(
    prompt: MusicPrompt,
    recognition=None,
    features=None,
    user_edited: bool = False,
) -> str:
    """Build the single text prompt sent to Lyria.

    When ``recognition``/``features`` are available, translate the source
    audio's acoustic character (species gesture, tempo, register, brightness,
    dynamics) into concrete musical direction so the output stays related to
    the input sound. A user-edited caption always leads; the acoustic
    direction is derived from the audio itself and is appended regardless.
    """
    lead = prompt.caption.strip().rstrip(".")
    parts = [lead + "."]

    if recognition is not None:
        name = recognition.common_name_zh or recognition.species
        gesture, instrument = _SOURCE_GESTURES.get(
            str(recognition.species),
            (f"the call of {name}", "motifs that echo the call's contour"),
        )
        call_type = str(getattr(recognition, "call_type", "") or "")
        rhythm = ""
        if call_type == "drumming":
            rhythm = " Give the rhythm a drumming, percussive drive."
        elif call_type == "song":
            rhythm = " Let the melody flow in longer, song-like phrases."
        parts.append(
            f"The source sound is {gesture}; translate it into {instrument}.{rhythm}"
        )

    if features is not None:
        if features.estimated_bpm and features.estimated_bpm > 0:
            parts.append(f"Tempo around {features.estimated_bpm:.0f} BPM.")
        if features.dominant_frequency_hz and features.dominant_frequency_hz > 0:
            parts.append(
                f"Center the lead motif in {_register_descriptor(features.dominant_frequency_hz)}."
            )
        mood = _brightness_descriptor(features.spectral_centroid_hz)
        dynamics = _dynamics_descriptor(features.rms)
        parts.append(f"Overall mood {mood}, with {dynamics} dynamics.")

    parts.append(
        "Clear introduction, gradual development, gentle ending. Instrumental only, no vocals."
    )
    return " ".join(parts)


class LyriaGenerator:
    """Generate music with Google Lyria 3 through the Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        ffmpeg: str = "ffmpeg",
        timeout_s: float = 240.0,
    ) -> None:
        self.api_key = resolve_api_key(api_key)
        self.model = model
        self.ffmpeg = ffmpeg
        self.timeout_s = timeout_s
        self.last_prompt: str | None = None

    def _request(self, text: str) -> tuple[bytes, str]:
        """Call generateContent; return (audio_bytes, mime_type)."""
        url = f"{API_BASE}/models/{self.model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"responseModalities": ["AUDIO", "TEXT"]},
        }
        last_error: RuntimeError | None = None
        for _attempt in range(3):
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:800]
                raise RuntimeError(f"Lyria API 返回 HTTP {exc.code}: {detail}") from None
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Lyria API 请求失败（检查网络/代理）: {exc.reason}") from None

            candidates = body.get("candidates") or []
            if not candidates:
                feedback = body.get("promptFeedback", {})
                raise RuntimeError(f"Lyria 未返回结果（可能被安全过滤拦截）: {feedback}")
            # The preview model occasionally returns no audio part; scan every
            # candidate/part and retry the request a couple of times.
            for candidate in candidates:
                for part in candidate.get("content", {}).get("parts", []):
                    inline = part.get("inlineData") or part.get("inline_data")
                    if inline and str(inline.get("mimeType", "")).startswith("audio/"):
                        return base64.b64decode(inline["data"]), str(inline["mimeType"])
            last_error = RuntimeError("Lyria 响应中没有音频数据")
        raise last_error

    def generate(
        self,
        prompt: MusicPrompt,
        destination: str | Path,
        duration_s: float = 10.0,
        recognition=None,
        features=None,
        user_edited: bool = False,
    ) -> Path:
        """Generate music for ``prompt`` and write a 44.1 kHz stereo WAV.

        ``recognition``/``features`` let the composer keep the output related
        to the source audio; ``user_edited`` marks that the caption came from
        the parameter panel. The final text sent to Lyria is kept on
        ``self.last_prompt`` for the report.
        """
        duration_s = max(2.0, min(_CLIP_SECONDS, float(duration_s)))
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        if recognition is not None or features is not None:
            text = compose_lyria_prompt(
                prompt, recognition=recognition, features=features, user_edited=user_edited
            )
        else:
            text = condense_prompt(prompt)
        self.last_prompt = text

        audio_bytes, mime = self._request(text)
        suffix = ".mp3" if "mpeg" in mime or "mp3" in mime else ".bin"
        with tempfile.TemporaryDirectory(prefix="nature2music_lyria_") as temp_name:
            raw_path = Path(temp_name) / f"clip{suffix}"
            raw_path.write_bytes(audio_bytes)
            # Trim to the requested length with a short fade-out to avoid an
            # abrupt cut, then normalize to 44.1 kHz stereo WAV.
            fade_start = max(0.0, duration_s - 0.8)
            command = [
                self.ffmpeg,
                "-y",
                "-i",
                str(raw_path),
                "-t",
                f"{duration_s:.3f}",
                "-af",
                f"afade=t=out:st={fade_start:.3f}:d=0.8",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(target),
            ]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode or not target.is_file():
                raise RuntimeError(f"ffmpeg 转码失败: {process.stderr[-500:]}")
        return target
