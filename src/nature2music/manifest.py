from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from .schema import Recording


def read_jsonl(path: str | Path) -> Iterator[Recording]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield Recording.from_dict(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid manifest row {path}:{line_no}: {exc}") from exc


def write_jsonl(records: Iterable[Recording], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def csv_to_manifest(
    csv_path: str | Path,
    output_path: str | Path,
    audio_root: str | Path | None = None,
    require_files: bool = True,
) -> int:
    root = Path(audio_root).resolve() if audio_root else None
    records: list[Recording] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row_no, row in enumerate(csv.DictReader(handle), 2):
            raw_path = row.get("audio_path", "").strip()
            if not raw_path:
                raise ValueError(f"CSV row {row_no}: audio_path is required")
            audio_path = Path(raw_path)
            if root and not audio_path.is_absolute():
                audio_path = root / audio_path
            audio_path = audio_path.resolve()
            if require_files and not audio_path.is_file():
                raise FileNotFoundError(f"CSV row {row_no}: missing audio {audio_path}")
            row["audio_path"] = str(audio_path)
            row.setdefault("recording_id", audio_path.stem)
            records.append(Recording.from_dict(row))
    return write_jsonl(records, output_path)


def _stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def stratified_group_split(
    records: Iterable[Recording],
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> list[Recording]:
    """Split global recording groups, stratified by their multi-species label signature."""

    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be >= 0 and sum to < 1")

    global_groups: dict[str, list[Recording]] = defaultdict(list)
    for record in records:
        global_groups[record.leakage_group].append(record)

    # A recording can contain several labelled species. It must be treated as one
    # indivisible group; stratifying each species independently would leak mixtures.
    by_signature: dict[tuple[str, ...], dict[str, list[Recording]]] = defaultdict(dict)
    for key, items in global_groups.items():
        signature = tuple(sorted({item.species for item in items}))
        by_signature[signature][key] = items

    result: list[Recording] = []
    for signature, groups in sorted(by_signature.items()):
        keys = list(groups)
        random.Random(_stable_seed("|".join(signature), seed)).shuffle(keys)
        n_groups = len(keys)
        n_test = round(n_groups * test_ratio) if n_groups >= 3 else 0
        n_val = round(n_groups * validation_ratio) if n_groups >= 3 else 0
        if test_ratio > 0 and n_groups >= 5:
            n_test = max(1, n_test)
        if validation_ratio > 0 and n_groups >= 5:
            n_val = max(1, n_val)
        while n_test + n_val >= n_groups:
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
        split_for = {
            key: ("test" if idx < n_test else "validation" if idx < n_test + n_val else "train")
            for idx, key in enumerate(keys)
        }
        for key, items in groups.items():
            for item in items:
                item.split = split_for[key]
                result.append(item)
    return sorted(result, key=lambda item: (item.split, item.species, item.recording_id))


def validate_manifest(records: Iterable[Recording], min_train_per_species: int = 1) -> dict:
    rows = list(records)
    paths = Counter(row.audio_path for row in rows)
    ids = Counter((row.source, row.recording_id, row.start_s, row.end_s) for row in rows)
    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_groups[row.leakage_group].add(row.split)
    leaked = sorted(key for key, splits in split_groups.items() if len(splits) > 1)
    train_counts = Counter(row.species for row in rows if row.split == "train")
    species = sorted({row.species for row in rows})
    return {
        "rows": len(rows),
        "species": len(species),
        "groups": dict(sorted(Counter(row.group for row in rows).items())),
        "splits": dict(sorted(Counter(row.split for row in rows).items())),
        "duplicate_paths": sorted(path for path, count in paths.items() if count > 1),
        "duplicate_segments": sum(count - 1 for count in ids.values() if count > 1),
        "leakage_groups": leaked,
        "underfilled_train_species": sorted(
            name for name in species if train_counts[name] < min_train_per_species
        ),
        "ok": not leaked and all(train_counts[name] >= min_train_per_species for name in species),
    }
