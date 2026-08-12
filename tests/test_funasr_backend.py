from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nature2music.funasr_backend import (
    export_funasr_splits,
    parse_recognition,
    recording_to_chatml,
)
from nature2music.manifest import write_jsonl
from nature2music.schema import Recording


class FunASRBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = Recording(
            audio_path="C:/audio/call.wav",
            species="eurasian_blackbird",
            group="bird",
            source="xc",
            recording_id="XC1",
            split="train",
            scientific_name="Turdus merula",
            common_name_zh="乌鸫",
            background=["wind"],
        )

    def test_chatml_uses_official_prompt_envelope(self) -> None:
        value = recording_to_chatml(self.record)
        self.assertTrue(value["messages"][1]["content"].startswith("语音转写："))
        answer = json.loads(value["messages"][2]["content"])
        self.assertEqual("eurasian_blackbird", answer["species"])

    def test_parse_json_after_thinking_block(self) -> None:
        recognition = parse_recognition(
            '<think>ignored</think>\n```json\n{"group":"bird","species":"blackbird",'
            '"confidence":1.4,"background":"wind, rain"}\n```'
        )
        self.assertEqual("blackbird", recognition.species)
        self.assertEqual(1.0, recognition.confidence)
        self.assertEqual(["wind", "rain"], recognition.background)

    def test_export_splits(self) -> None:
        validation = Recording.from_dict(self.record.to_dict() | {"split": "validation"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            write_jsonl([self.record, validation], manifest)
            counts = export_funasr_splits(manifest, root / "funasr")
            self.assertEqual(1, counts["train"])
            self.assertEqual(1, counts["validation"])
            self.assertTrue((root / "funasr" / "train.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()

