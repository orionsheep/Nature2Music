from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn

    from nature2music.lora_layers import adapter_state_dict, inject_lora
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class LoRALayerTests(unittest.TestCase):
    def test_injection_preserves_initial_output_and_freezes_base(self) -> None:
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        value = torch.randn(3, 4)
        expected = model(value).detach()
        names = inject_lora(model, [r"0$", r"2$"], rank=2, alpha=4, dropout=0)
        actual = model(value).detach()
        self.assertEqual(["0", "2"], names)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name for name in trainable))
        self.assertEqual(4, len(adapter_state_dict(model)))


if __name__ == "__main__":
    unittest.main()

