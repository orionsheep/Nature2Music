from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nature2music.patchers import patch_funasr_model, patch_thinksound_train


class PatcherTests(unittest.TestCase):
    def test_funasr_patch_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model.py"
            target = Path(directory) / "model_lora.py"
            source.write_text(
                "class M:\n    def f(self):\n        self.llm = model.to(dtype_map[self.llm_dtype])\n",
                encoding="utf-8",
            )
            patch_funasr_model(source, target)
            patch_funasr_model(target, target)
            text = target.read_text(encoding="utf-8")
        self.assertEqual(1, text.count("nature2music: optional PEFT"))

    def test_thinksound_patch_injects_before_training_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.py"
            target = Path(directory) / "train_lora.py"
            source.write_text(
                "import torch\n\ndef main():\n"
                "    model = object()\n"
                "    training_wrapper = create_training_wrapper_from_config(model_config, model)\n",
                encoding="utf-8",
            )
            patch_thinksound_train(source, target)
            text = target.read_text(encoding="utf-8")
        self.assertIn("from nature2music.lora_layers import inject_lora", text)
        self.assertLess(
            text.index("nature2music: inject the same structure"), text.index("training_wrapper")
        )


if __name__ == "__main__":
    unittest.main()
