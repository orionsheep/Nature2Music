#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <Fun-ASR-upstream> <exported-data-dir> <output-dir> [model-id]"
  exit 2
fi

UPSTREAM="$(cd "$1" && pwd)"
DATA_DIR="$(cd "$2" && pwd)"
OUTPUT_DIR="$3"
MODEL_ID="${4:-FunAudioLLM/Fun-ASR-Nano-2512}"
NPROC="${NPROC_PER_NODE:-1}"

python -m nature2music.cli patch-funasr-lora \
  "$UPSTREAM/model.py" "$UPSTREAM/model_lora.py"

TRAIN_TOOL="$(command -v funasr-train-ds)"
torchrun --nproc_per_node="$NPROC" "$TRAIN_TOOL" \
  ++model="$MODEL_ID" \
  ++trust_remote_code=true \
  ++remote_code="$UPSTREAM/model_lora.py" \
  ++train_data_set_list="$DATA_DIR/train.jsonl" \
  ++valid_data_set_list="$DATA_DIR/validation.jsonl" \
  ++dataset_conf.data_split_num=1 \
  ++dataset_conf.batch_sampler=BatchSampler \
  ++dataset_conf.batch_size=6000 \
  ++dataset_conf.batch_type=token \
  ++dataset_conf.num_workers=4 \
  ++train_conf.max_epoch=30 \
  ++train_conf.validate_interval=1000 \
  ++train_conf.save_checkpoint_interval=1000 \
  ++train_conf.resume=true \
  ++train_conf.use_deepspeed=false \
  ++optim_conf.lr=0.0002 \
  ++audio_encoder_conf.freeze=true \
  ++audio_adaptor_conf.freeze=false \
  ++llm_conf.freeze=true \
  ++llm_conf.lora_conf.enabled=true \
  ++llm_conf.lora_conf.r=16 \
  ++llm_conf.lora_conf.alpha=32 \
  ++llm_conf.lora_conf.dropout=0.05 \
  ++output_dir="$OUTPUT_DIR"

