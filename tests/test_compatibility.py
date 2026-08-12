from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nature2music.funasr_backend import recording_to_chatml
from nature2music.patchers import patch_thinksound_predict
from nature2music.schema import Recording


class UpstreamCompatibilityTests(unittest.TestCase):
    def test_funasr_chatml_includes_dynamic_batch_lengths(self) -> None:
        record = Recording(
            audio_path="C:/audio/example.wav",
            species="cricket",
            group="insect",
            source="test",
            recording_id="1",
            start_s=2,
            end_s=12,
        )
        value = recording_to_chatml(record)
        self.assertEqual(1000, value["speech_length"])
        self.assertGreater(value["text_length"], 0)
        self.assertTrue(value["nature2music"]["lengths_are_estimates"])

    def test_predict_patch_injects_before_checkpoint_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "predict.py"
            target = Path(directory) / "predict_lora.py"
            source.write_text(
                "import os\nimport torch\n\ndef main():\n"
                "    model = create_model_from_config(model_config)\n"
                "    model.load_state_dict(torch.load(args.ckpt_dir))\n",
                encoding="utf-8",
            )
            patch_thinksound_predict(source, target)
            text = target.read_text(encoding="utf-8")
        self.assertIn("nature2music: inject the same structure", text)
        self.assertLess(text.index("nature2music: inject"), text.index("load_state_dict"))


if __name__ == "__main__":
    unittest.main()
