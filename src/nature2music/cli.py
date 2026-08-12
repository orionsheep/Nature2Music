from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .funasr_backend import FunASRRecognizer, export_funasr_splits
from .manifest import (
    csv_to_manifest,
    read_jsonl,
    stratified_group_split,
    validate_manifest,
    write_jsonl,
)
from .patchers import patch_funasr_model, patch_thinksound_predict, patch_thinksound_train
from .prompting import build_music_prompt
from .schema import Recognition


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _add_recognizer_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hub", default="hf", choices=["hf", "ms"])
    parser.add_argument("--remote-code")
    parser.add_argument("--hotwords", help="UTF-8 file containing one candidate species per line")


def _add_thinksound_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--thinksound-dir", required=True)
    parser.add_argument("--thinksound-checkpoint", help="Unwrapped ThinkSound state file")
    parser.add_argument("--thinksound-lora", action="store_true")
    parser.add_argument("--python", default="python")
    parser.add_argument("--no-half", action="store_true")


def _recognizer(args: argparse.Namespace) -> FunASRRecognizer:
    hotwords: list[str] = []
    if args.hotwords:
        hotwords = [
            line.strip()
            for line in Path(args.hotwords).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return FunASRRecognizer(
        model=args.model,
        device=args.device,
        hub=args.hub,
        remote_code=args.remote_code,
        hotwords=hotwords,
    )


def _generator(args: argparse.Namespace):
    from .thinksound_backend import ThinkSoundGenerator

    return ThinkSoundGenerator(
        args.thinksound_dir,
        python=args.python,
        use_half=not args.no_half,
        checkpoint=args.thinksound_checkpoint,
        lora=args.thinksound_lora,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nature2music",
        description="Bioacoustic recognition with Fun-ASR and music generation with ThinkSound.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("manifest-from-csv", help="Normalize a metadata CSV into JSONL")
    command.add_argument("csv")
    command.add_argument("output")
    command.add_argument("--audio-root")
    command.add_argument("--allow-missing", action="store_true")

    command = sub.add_parser("split-manifest", help="Leakage-safe grouped stratified split")
    command.add_argument("manifest")
    command.add_argument("output")
    command.add_argument("--validation-ratio", type=float, default=0.1)
    command.add_argument("--test-ratio", type=float, default=0.1)
    command.add_argument("--seed", type=int, default=42)

    command = sub.add_parser("validate-manifest", help="Audit duplicates and split leakage")
    command.add_argument("manifest")
    command.add_argument("--min-train-per-species", type=int, default=1)

    command = sub.add_parser("export-funasr", help="Export Fun-ASR ChatML training JSONL")
    command.add_argument("manifest")
    command.add_argument("output_dir")

    command = sub.add_parser("export-thinksound", help="Export ThinkSound audio/text pairs")
    command.add_argument("manifest")
    command.add_argument("output_dir")
    command.add_argument("--stage-mode", choices=["none", "hardlink", "copy"], default="none")
    command.add_argument("--style", default="cinematic ambient world music")

    command = sub.add_parser("patch-funasr-lora", help="Create a LoRA-enabled Fun-ASR model.py")
    command.add_argument("source")
    command.add_argument("destination")

    command = sub.add_parser("patch-thinksound-lora", help="Create a LoRA-enabled ThinkSound train.py")
    command.add_argument("source")
    command.add_argument("destination")

    command = sub.add_parser(
        "patch-thinksound-predict-lora", help="Create a LoRA-enabled ThinkSound predict.py"
    )
    command.add_argument("source")
    command.add_argument("destination")

    command = sub.add_parser("identify", help="Identify a species using a fine-tuned Fun-ASR")
    command.add_argument("audio")
    command.add_argument("--output")
    _add_recognizer_options(command)

    command = sub.add_parser("build-prompt", help="Build a ThinkSound prompt from recognition JSON")
    command.add_argument("recognition_json")
    command.add_argument("--style", default="cinematic ambient world music")

    command = sub.add_parser("compose", help="Generate audio from recognition JSON with ThinkSound")
    command.add_argument("recognition_json")
    command.add_argument("output_audio")
    command.add_argument("--style", default="cinematic ambient world music")
    command.add_argument("--duration", type=float, default=10.0)
    _add_thinksound_options(command)

    command = sub.add_parser("run", help="End-to-end identify, analyze, prompt, and generate")
    command.add_argument("audio")
    command.add_argument("output_audio")
    command.add_argument("--report")
    command.add_argument("--style", default="cinematic ambient world music")
    command.add_argument("--duration", type=float)
    _add_thinksound_options(command)
    _add_recognizer_options(command)
    return parser


def _load_recognition(path: str | Path) -> Recognition:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if "recognition" in value:
        value = value["recognition"]
    allowed = set(Recognition.__dataclass_fields__)
    return Recognition(**{key: val for key, val in value.items() if key in allowed})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "manifest-from-csv":
        count = csv_to_manifest(
            args.csv, args.output, args.audio_root, require_files=not args.allow_missing
        )
        _print({"rows": count, "output": str(Path(args.output).resolve())})
    elif args.command == "split-manifest":
        records = stratified_group_split(
            read_jsonl(args.manifest), args.validation_ratio, args.test_ratio, args.seed
        )
        count = write_jsonl(records, args.output)
        _print({"rows": count, "output": str(Path(args.output).resolve())})
    elif args.command == "validate-manifest":
        report = validate_manifest(read_jsonl(args.manifest), args.min_train_per_species)
        _print(report)
        return 0 if report["ok"] else 2
    elif args.command == "export-funasr":
        _print(export_funasr_splits(args.manifest, args.output_dir))
    elif args.command == "export-thinksound":
        from .thinksound_data import export_thinksound_pairs

        _print(export_thinksound_pairs(args.manifest, args.output_dir, args.stage_mode, args.style))
    elif args.command == "patch-funasr-lora":
        _print({"output": str(patch_funasr_model(args.source, args.destination).resolve())})
    elif args.command == "patch-thinksound-lora":
        _print({"output": str(patch_thinksound_train(args.source, args.destination).resolve())})
    elif args.command == "patch-thinksound-predict-lora":
        _print({"output": str(patch_thinksound_predict(args.source, args.destination).resolve())})
    elif args.command == "identify":
        recognition = _recognizer(args).recognize(args.audio)
        payload = recognition.to_dict()
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
            )
        _print(payload)
    elif args.command == "build-prompt":
        _print(asdict(build_music_prompt(_load_recognition(args.recognition_json), style=args.style)))
    elif args.command == "compose":
        prompt = build_music_prompt(_load_recognition(args.recognition_json), style=args.style)
        _print({"output": str(_generator(args).generate(prompt, args.output_audio, args.duration))})
    elif args.command == "run":
        from .pipeline import run_pipeline

        report = run_pipeline(
            args.audio,
            args.output_audio,
            _recognizer(args),
            _generator(args),
            style=args.style,
            duration_s=args.duration,
            report_path=args.report,
        )
        _print(report.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
