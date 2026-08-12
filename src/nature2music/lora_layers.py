from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A transparent LoRA wrapper for an existing torch Linear layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        self.lora_a.to(dtype=base.weight.dtype)
        self.lora_b.to(dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.lora_b(self.lora_a(self.dropout(value))) * self.scaling


def inject_lora(
    model: nn.Module,
    patterns: Iterable[str],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    freeze_base: bool = True,
) -> list[str]:
    """Replace matching Linear modules and return their fully qualified names."""

    compiled = [re.compile(pattern) for pattern in patterns]
    if freeze_base:
        for parameter in model.parameters():
            parameter.requires_grad = False
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear) and any(regex.search(name) for regex in compiled)
    ]
    for name, module in matches:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
    return [name for name, _ in matches]


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name
    }


def save_adapter(model: nn.Module, path: str | Path, metadata: dict | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter_state_dict(model), "metadata": metadata or {}}, target)


def load_adapter(model: nn.Module, path: str | Path, strict: bool = True) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = payload["adapter"]
    result = model.load_state_dict(state, strict=False)
    if strict and (result.unexpected_keys or any("lora_" in key for key in result.missing_keys)):
        raise RuntimeError(f"adapter mismatch: {result}")
    return payload.get("metadata", {})

