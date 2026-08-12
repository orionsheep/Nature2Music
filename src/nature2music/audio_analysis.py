from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class AudioFeatures:
    duration_s: float
    sample_rate: int
    rms: float
    spectral_centroid_hz: float
    dominant_frequency_hz: float
    estimated_bpm: float

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_audio(path: str | Path) -> AudioFeatures:
    """Extract lightweight descriptors used to condition the arrangement prompt."""

    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError('Audio analysis requires: pip install -e ".[asr]"') from exc

    resolved = str(Path(path).resolve())
    try:
        waveform, sample_rate = torchaudio.load(resolved)
    except ImportError:
        # torchaudio >= 2.9 requires torchcodec for load(); fall back to
        # soundfile (already a dependency) when torchcodec is unavailable.
        import soundfile as sf

        data, sample_rate = sf.read(resolved, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # (channels, samples)
    mono = waveform.mean(dim=0).float()
    if mono.numel() == 0:
        raise ValueError("audio is empty")
    duration = mono.numel() / sample_rate
    rms = torch.sqrt(torch.mean(mono.square()) + 1e-12).item()

    n_fft = min(4096, max(256, 2 ** int(math.log2(max(256, mono.numel())))))
    spec = torch.stft(
        mono,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        window=torch.hann_window(n_fft),
        return_complex=True,
    ).abs()
    mean_spec = spec.mean(dim=1)
    freqs = torch.linspace(0, sample_rate / 2, mean_spec.numel())
    centroid = (freqs * mean_spec).sum() / (mean_spec.sum() + 1e-12)
    dominant_index = int(mean_spec[1:].argmax().item() + 1) if mean_spec.numel() > 1 else 0
    dominant = float(freqs[dominant_index].item())

    energy = spec.square().mean(dim=0)
    onset = torch.relu(energy[1:] - energy[:-1])
    bpm = 0.0
    if onset.numel() >= 8 and float(onset.max()) > 0:
        centered = onset - onset.mean()
        fft_size = 1 << (2 * centered.numel() - 1).bit_length()
        spectrum = torch.fft.rfft(centered, n=fft_size)
        correlation = torch.fft.irfft(spectrum.conj() * spectrum, n=fft_size)[: centered.numel()]
        frames_per_second = sample_rate / (n_fft // 4)
        min_lag = max(1, round(frames_per_second * 60 / 220))
        max_lag = min(correlation.numel() - 1, round(frames_per_second * 60 / 45))
        if max_lag > min_lag:
            lag = int(
                correlation[min_lag : max_lag + 1].argmax().item() + min_lag
            )
            bpm = float(60 * frames_per_second / lag)
    return AudioFeatures(
        duration_s=round(duration, 3),
        sample_rate=sample_rate,
        rms=round(rms, 6),
        spectral_centroid_hz=round(float(centroid.item()), 2),
        dominant_frequency_hz=round(dominant, 2),
        estimated_bpm=round(bpm, 2),
    )
