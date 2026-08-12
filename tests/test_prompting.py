from __future__ import annotations

import unittest

from nature2music.audio_analysis import AudioFeatures
from nature2music.prompting import build_music_prompt
from nature2music.schema import Recognition


class PromptingTests(unittest.TestCase):
    def test_prompt_contains_species_and_audio_features(self) -> None:
        prompt = build_music_prompt(
            Recognition(
                group="insect",
                species="field_cricket",
                confidence=0.9,
                common_name_zh="田野蟋蟀",
            ),
            AudioFeatures(10, 48000, 0.1, 3500, 4200, 96),
            style="minimal ambient music",
        )
        self.assertIn("田野蟋蟀", prompt.caption)
        self.assertIn("96 BPM", prompt.chain_of_thought)
        self.assertIn("4200 Hz", prompt.chain_of_thought)


if __name__ == "__main__":
    unittest.main()

