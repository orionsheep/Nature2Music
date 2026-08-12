from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

from .manifest import read_jsonl
from .prompting import build_music_prompt
from .schema import Recognition


def _item_id(source: str, recording_id: str, path: str) -> str:
    digest = hashlib.sha1(f"{source}:{recording_id}:{path}".encode()).hexdigest()[:12]
    return f"n2m_{digest}"


def export_thinksound_pairs(
    manifest_path: str | Path,
    output_dir: str | Path,
    stage_mode: str = "none",
    style: str = "cinematic ambient world music",
) -> dict[str, int]:
    """Create ThinkSound audio/text CSVs and optionally stage source audio."""

    if stage_mode not in {"none", "hardlink", "copy"}:
        raise ValueError("stage_mode must be none, hardlink, or copy")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for record in read_jsonl(manifest_path):
        if record.split not in rows:
            raise ValueError("manifest contains unassigned rows; run split-manifest first")
        recognition = Recognition(
            group=record.group,
            species=record.species,
            confidence=1.0,
            scientific_name=record.scientific_name,
            common_name_zh=record.common_name_zh,
            call_type=record.call_type,
            background=record.background,
        )
        prompt = build_music_prompt(recognition, style=style)
        source = Path(record.audio_path).resolve()
        item_id = _item_id(record.source, record.recording_id, str(source))
        staged_path = source
        if stage_mode != "none":
            staged_dir = output / record.split / "audio"
            staged_dir.mkdir(parents=True, exist_ok=True)
            staged_path = staged_dir / f"{item_id}{source.suffix.lower()}"
            if not staged_path.exists():
                if stage_mode == "hardlink":
                    try:
                        os.link(source, staged_path)
                    except OSError:
                        shutil.copy2(source, staged_path)
                else:
                    shutil.copy2(source, staged_path)
        rows[record.split].append(
            {
                "id": item_id,
                "caption": prompt.caption,
                "caption_cot": prompt.chain_of_thought,
                "audio_path": str(staged_path),
            }
        )

    counts: dict[str, int] = {}
    for split, split_rows in rows.items():
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "pairs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["id", "caption", "caption_cot", "audio_path"]
            )
            writer.writeheader()
            writer.writerows(split_rows)
        (split_dir / "expected_features.txt").write_text(
            "".join(f"{row['id']}.pth\n" for row in split_rows), encoding="utf-8", newline="\n"
        )
        counts[split] = len(split_rows)
    (output / "export_summary.json").write_text(
        json.dumps({"stage_mode": stage_mode, "counts": counts}, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return counts

