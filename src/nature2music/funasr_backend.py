from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from .manifest import read_jsonl
from .schema import Recognition, Recording


SYSTEM_PROMPT = "You are a helpful assistant."
TRAIN_PROMPT = "语音转写："


def recording_to_chatml(record: Recording) -> dict[str, Any]:
    """Build the ChatML envelope expected by Fun-ASR-Nano fine-tuning."""

    answer = record.label_payload() | {"confidence": 1.0}
    answer_text = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    duration_s = float(record.metadata.get("duration_s", 10.0))
    if record.start_s is not None and record.end_s is not None:
        duration_s = record.end_s - record.start_s
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{TRAIN_PROMPT}<|startofspeech|>!{Path(record.audio_path).resolve()}"
                    "<|endofspeech|>"
                ),
            },
            {"role": "assistant", "content": answer_text},
        ],
        # Fun-ASR uses these values for dynamic batching. The official scp2jsonl.py
        # can provide exact lengths; these conservative estimates work for uniform clips.
        "speech_length": max(1, round(duration_s * 100)),
        "text_length": max(1, math.ceil(len(answer_text.encode("utf-8")) / 3)),
        "nature2music": {
            "source": record.source,
            "recording_id": record.recording_id,
            "split": record.split,
            "lengths_are_estimates": True,
        },
    }


def export_funasr_splits(
    manifest_path: str | Path,
    output_dir: str | Path,
    include_test: bool = True,
) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    handles: dict[str, Any] = {}
    counts = {"train": 0, "validation": 0, "test": 0}
    try:
        for split in counts:
            if split == "test" and not include_test:
                continue
            handles[split] = (output / f"{split}.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            )
        for record in read_jsonl(manifest_path):
            if record.split == "unassigned":
                raise ValueError("manifest contains unassigned rows; run split-manifest first")
            if record.split not in handles:
                continue
            handles[record.split].write(
                json.dumps(recording_to_chatml(record), ensure_ascii=False) + "\n"
            )
            counts[record.split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"Fun-ASR output does not contain a JSON object: {text[:300]!r}")


def parse_recognition(text: str) -> Recognition:
    value = _extract_json_object(text)
    species = str(value.get("species", "")).strip()
    if not species:
        raise ValueError("Fun-ASR JSON output is missing species")
    background = value.get("background", [])
    if isinstance(background, str):
        background = [item.strip() for item in background.split(",") if item.strip()]
    confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0))))
    return Recognition(
        group=str(value.get("group", "unknown")).strip().lower() or "unknown",
        species=species,
        confidence=confidence,
        scientific_name=str(value.get("scientific_name", "")).strip(),
        common_name_zh=str(value.get("common_name_zh", "")).strip(),
        call_type=str(value.get("call_type", "")).strip(),
        background=[str(item) for item in background],
        raw_text=text,
    )


class FunASRRecognizer:
    """Lazy Fun-ASR-Nano inference adapter for a fine-tuned bioacoustic checkpoint."""

    def __init__(
        self,
        model: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        device: str = "cuda:0",
        hub: str = "hf",
        remote_code: str | None = None,
        hotwords: Iterable[str] = (),
    ) -> None:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError('Install the ASR dependencies with: pip install -e ".[asr]"') from exc
        # A fine-tuned output dir only stores trainable weights (audio adaptor +
        # LoRA); the frozen LLM/encoder weights are excluded. Always build from the
        # base model, then overlay the trained checkpoint non-strictly.
        base_model = model
        checkpoint: Path | None = None
        model_dir = Path(model)
        if model_dir.is_dir():
            for name in ("model.pt.best", "model.pt"):
                if (model_dir / name).is_file():
                    checkpoint = model_dir / name
                    break
            if checkpoint is not None:
                base_model = "FunAudioLLM/Fun-ASR-Nano-2512"
        kwargs: dict[str, Any] = {
            "model": base_model,
            "trust_remote_code": True,
            "device": device,
            "hub": hub,
        }
        if remote_code:
            # funasr only imports remote_code on the ModelScope path; the HF
            # path skips it, which would silently fall back to the built-in
            # (non-peft) FunASRNano class. Import it here so the LoRA-aware
            # class is registered before the model is built.
            from funasr.utils.dynamic_import import import_module_from_path

            import_module_from_path(str(Path(remote_code).resolve()))
            kwargs["remote_code"] = remote_code
        state_dict: dict[str, Any] | None = None
        if checkpoint is not None:
            import torch

            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state_dict = state.get("state_dict", state)
            # Rebuild the LoRA wrappers before loading, otherwise the trained
            # adapter weights have no matching module in the freshly built model.
            if any("lora_" in k for k in state_dict):
                lora_conf = {"enabled": True, "r": 16, "alpha": 32, "dropout": 0.0}
                llm_conf: dict[str, Any] = {"lora_conf": lora_conf}
                # FunASR's hub download merges a user-supplied llm_conf
                # wholesale (cfg.update(kwargs)), which drops the
                # file_path_metas rewrite of llm_conf.init_param_path and
                # leaves the relative "Qwen3-0.6B" from config.yaml. Resolve it
                # to the downloaded snapshot ourselves so offline runs work.
                try:
                    from huggingface_hub import snapshot_download

                    snapshot = snapshot_download(base_model)
                    candidate = Path(snapshot) / "Qwen3-0.6B"
                    if candidate.is_dir():
                        llm_conf["init_param_path"] = str(candidate)
                except Exception:
                    pass
                kwargs["llm_conf"] = llm_conf
        self._model = AutoModel(**kwargs)
        self._finetuned = state_dict is not None
        # funasr falls back to CPU when the requested device is unavailable;
        # read back the resolved device instead of trusting the request.
        self._device = str(getattr(self._model, "kwargs", {}).get("device", device))
        if state_dict is not None:
            missing, unexpected = self._model.model.load_state_dict(state_dict, strict=False)
            leftover = [k for k in unexpected if "lora_" in k or "adaptor" in k]
            if leftover:
                raise RuntimeError(f"checkpoint keys not applied: {leftover[:5]}")
        self._hotwords = list(hotwords)

    def recognize(self, audio_path: str | Path) -> Recognition:
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if self._finetuned:
            return parse_recognition(self._generate_finetuned(path))
        result = self._model.generate(
            input=[str(path)],
            cache={},
            batch_size=1,
            language=None,
            itn=True,
            hotwords=self._hotwords,
        )
        if not result:
            raise RuntimeError("Fun-ASR returned no result")
        return parse_recognition(str(result[0].get("text", "")))

    def _generate_finetuned(self, path: Path) -> str:
        """Generate the structured JSON answer from a fine-tuned checkpoint.

        The Fun-ASR LLM was pretrained on speech transcripts and its very first
        answer token ('{') sits right at the prompt/answer boundary, which the
        fine-tune never reproduces reliably. Seed the generation with the '{"'
        token embedding (a transition the fine-tune did learn) and prepend the
        token text to the decoded output.
        """
        import torch

        model = self._model.model
        tokenizer = self._model.kwargs["tokenizer"]
        frontend = self._model.kwargs["frontend"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{TRAIN_PROMPT}<|startofspeech|>!{path}<|endofspeech|>",
            },
            {"role": "assistant", "content": "null"},
        ]
        inputs_embeds, _contents, batch, _source_ids, _meta = model.inference_prepare(
            [messages],
            None,
            ["n2m"],
            tokenizer,
            frontend,
            device=self._device,
            batch_size=1,
            cache={},
        )
        start_id = tokenizer.encode('{"')[0]
        prefix_emb = model.llm.model.get_input_embeddings()(
            torch.tensor([[start_id]], device=inputs_embeds.device)
        ).to(inputs_embeds.dtype)
        embeds = torch.cat([inputs_embeds, prefix_emb], dim=1)
        attention_mask = batch.get("attention_mask")
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, 1, dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        core = model.llm.base_model.model  # bypass peft's generate wrapper
        generated = core.generate(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            max_new_tokens=256,
            pad_token_id=model.llm.config.pad_token_id or model.llm.config.eos_token_id,
        )
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return '{"' + text
