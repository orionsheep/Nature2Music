from __future__ import annotations

from pathlib import Path


FUNASR_ANCHOR = "        self.llm = model.to(dtype_map[self.llm_dtype])\n"
FUNASR_LORA_BLOCK = '''        # nature2music: optional PEFT LoRA on Fun-ASR's Qwen decoder
        lora_conf = llm_conf.get("lora_conf", {})
        if lora_conf and lora_conf.get("enabled", False):
            from peft import LoraConfig, TaskType, get_peft_model
            target_modules = lora_conf.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            if isinstance(target_modules, str):
                target_modules = [item.strip() for item in target_modules.split(",") if item.strip()]
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(lora_conf.get("r", 16)),
                lora_alpha=int(lora_conf.get("alpha", 32)),
                lora_dropout=float(lora_conf.get("dropout", 0.05)),
                target_modules=target_modules,
                bias="none",
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()
'''

THINKSOUND_IMPORT_ANCHOR = "import torch\n"
THINKSOUND_IMPORT_BLOCK = "from nature2music.lora_layers import inject_lora\n"
THINKSOUND_TRAIN_ANCHOR = (
    "    training_wrapper = create_training_wrapper_from_config(model_config, model)\n"
)
THINKSOUND_PREDICT_ANCHOR = "    model = create_model_from_config(model_config)\n"
THINKSOUND_LORA_BLOCK = '''    # nature2music: inject the same structure for training and checkpoint inference.
    if os.environ.get("N2M_THINKSOUND_LORA", "0") == "1":
        patterns = [
            item.strip()
            for item in os.environ.get(
                "N2M_THINKSOUND_LORA_TARGETS",
                r"(to_q|to_k|to_v|to_out\\.0|q_proj|k_proj|v_proj|out_proj)$",
            ).split(",")
            if item.strip()
        ]
        replaced = inject_lora(
            model,
            patterns=patterns,
            rank=int(os.environ.get("N2M_THINKSOUND_LORA_R", "16")),
            alpha=float(os.environ.get("N2M_THINKSOUND_LORA_ALPHA", "32")),
            dropout=float(os.environ.get("N2M_THINKSOUND_LORA_DROPOUT", "0.05")),
            freeze_base=True,
        )
        if not replaced:
            raise RuntimeError("ThinkSound LoRA matched no Linear modules; inspect model.named_modules()")
        print(f"nature2music: injected LoRA into {len(replaced)} ThinkSound modules")
'''


def _patch_once(text: str, anchor: str, block: str, marker: str, before: bool = False) -> str:
    if marker in text:
        return text
    count = text.count(anchor)
    if count != 1:
        raise ValueError(f"expected exactly one patch anchor, found {count}: {anchor!r}")
    replacement = block + anchor if before else anchor + block
    return text.replace(anchor, replacement, 1)


def patch_funasr_model(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    target = Path(destination)
    text = source_path.read_text(encoding="utf-8")
    text = _patch_once(text, FUNASR_ANCHOR, FUNASR_LORA_BLOCK, "nature2music: optional PEFT")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


def _patch_thinksound_import(text: str) -> str:
    return _patch_once(
        text,
        THINKSOUND_IMPORT_ANCHOR,
        THINKSOUND_IMPORT_BLOCK,
        "from nature2music.lora_layers import inject_lora",
    )


def patch_thinksound_train(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    target = Path(destination)
    text = _patch_thinksound_import(source_path.read_text(encoding="utf-8"))
    text = _patch_once(
        text,
        THINKSOUND_TRAIN_ANCHOR,
        THINKSOUND_LORA_BLOCK,
        "nature2music: inject the same structure",
        before=True,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


def patch_thinksound_predict(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    target = Path(destination)
    text = _patch_thinksound_import(source_path.read_text(encoding="utf-8"))
    text = _patch_once(
        text,
        THINKSOUND_PREDICT_ANCHOR,
        THINKSOUND_LORA_BLOCK,
        "nature2music: inject the same structure",
        before=False,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    return target
