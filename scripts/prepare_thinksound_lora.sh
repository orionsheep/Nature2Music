#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <ThinkSound-upstream>"
  exit 2
fi

UPSTREAM="$(cd "$1" && pwd)"
python -m nature2music.cli patch-thinksound-lora \
  "$UPSTREAM/train.py" "$UPSTREAM/train_lora.py"
python -m nature2music.cli patch-thinksound-predict-lora \
  "$UPSTREAM/predict.py" "$UPSTREAM/predict_lora.py"

echo "Patched training:  $UPSTREAM/train_lora.py"
echo "Patched inference: $UPSTREAM/predict_lora.py"
echo "Set N2M_THINKSOUND_LORA=1 and keep rank/alpha identical for training and inference."
