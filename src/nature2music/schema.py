from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


VALID_GROUPS = {"bird", "insect", "amphibian", "mammal", "environment", "unknown"}
VALID_SPLITS = {"train", "validation", "test", "unassigned"}


@dataclass(slots=True)
class Recording:
    """One weakly or strongly labelled bioacoustic recording."""

    audio_path: str
    species: str
    group: str
    source: str
    recording_id: str
    split: str = "unassigned"
    common_name_zh: str = ""
    common_name_en: str = ""
    scientific_name: str = ""
    call_type: str = ""
    background: list[str] = field(default_factory=list)
    site: str = ""
    recordist: str = ""
    license: str = ""
    latitude: float | None = None
    longitude: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.audio_path = str(Path(self.audio_path))
        self.species = self.species.strip()
        self.group = self.group.strip().lower() or "unknown"
        self.split = self.split.strip().lower() or "unassigned"
        if not self.species:
            raise ValueError("species must not be empty")
        if self.group not in VALID_GROUPS:
            raise ValueError(f"unsupported group: {self.group!r}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"unsupported split: {self.split!r}")
        if self.end_s is not None and self.start_s is not None and self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")

    @property
    def leakage_group(self) -> str:
        """Group clips from the same original recording/site during splitting."""

        return f"{self.source}:{self.recording_id or self.site or self.audio_path}"

    def label_payload(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "species": self.species,
            "scientific_name": self.scientific_name,
            "common_name_zh": self.common_name_zh,
            "common_name_en": self.common_name_en,
            "call_type": self.call_type,
            "background": self.background,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Recording":
        known = set(cls.__dataclass_fields__)
        payload = {key: val for key, val in value.items() if key in known}
        extras = {key: val for key, val in value.items() if key not in known}
        payload.setdefault("metadata", {}).update(extras)
        if isinstance(payload.get("background"), str):
            payload["background"] = [
                item.strip() for item in payload["background"].split("|") if item.strip()
            ]
        for key in ("latitude", "longitude", "start_s", "end_s"):
            if payload.get(key) in ("", None):
                payload[key] = None
            else:
                payload[key] = float(payload[key])
        return cls(**payload)


@dataclass(slots=True)
class Recognition:
    group: str
    species: str
    confidence: float
    scientific_name: str = ""
    common_name_zh: str = ""
    call_type: str = ""
    background: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

