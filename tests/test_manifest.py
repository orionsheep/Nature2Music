from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nature2music.manifest import (
    read_jsonl,
    stratified_group_split,
    validate_manifest,
    write_jsonl,
)
from nature2music.schema import Recording


def make_record(species: str, recording: int, segment: int) -> Recording:
    return Recording(
        audio_path=f"C:/dataset/{species}_{recording}_{segment}.wav",
        species=species,
        group="bird" if species == "blackbird" else "insect",
        source="unit-test",
        recording_id=f"recording-{recording}",
        start_s=float(segment * 5),
        end_s=float(segment * 5 + 5),
    )


class ManifestTests(unittest.TestCase):
    def test_group_split_has_no_recording_leakage(self) -> None:
        records = [
            make_record(species, recording, segment)
            for species in ("blackbird", "cricket")
            for recording in range(6)
            for segment in range(2)
        ]
        split = stratified_group_split(records, validation_ratio=0.2, test_ratio=0.2, seed=7)
        seen: dict[str, set[str]] = {}
        for row in split:
            seen.setdefault(row.leakage_group, set()).add(row.split)
        self.assertTrue(all(len(value) == 1 for value in seen.values()))
        self.assertEqual({"train", "validation", "test"}, {row.split for row in split})
        self.assertTrue(validate_manifest(split)["ok"])

    def test_jsonl_round_trip(self) -> None:
        record = make_record("blackbird", 1, 0)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manifest.jsonl"
            self.assertEqual(1, write_jsonl([record], target))
            loaded = list(read_jsonl(target))
        self.assertEqual(record.to_dict(), loaded[0].to_dict())


if __name__ == "__main__":
    unittest.main()

