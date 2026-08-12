# ThinkSound LoRA 训练与推理闭环

运行：

```bash
bash scripts/prepare_thinksound_lora.sh external/ThinkSound
```

会生成两个文件：

- `external/ThinkSound/train_lora.py`：在基础 checkpoint 和 VAE 加载后注入 LoRA，再创建 Lightning training wrapper。
- `external/ThinkSound/predict_lora.py`：先创建同样的 LoRA 结构，再载入微调 checkpoint。

训练与推理必须使用完全相同的环境变量：

```bash
export N2M_THINKSOUND_LORA=1
export N2M_THINKSOUND_LORA_R=16
export N2M_THINKSOUND_LORA_ALPHA=32
export N2M_THINKSOUND_LORA_DROPOUT=0.05
```

按上游 `scripts/train.sh` 启动训练时，将入口由 `train.py` 改成 `train_lora.py`。Lightning checkpoint 如不能直接被 `predict.py` 使用，先按 ThinkSound 上游方式运行 `unwrap.py`，获得无 training-wrapper 前缀的模型 state 文件。

LoRA 推理：

```bash
nature2music compose outputs/recognition.json outputs/nature_music.wav \
  --thinksound-dir external/ThinkSound \
  --thinksound-checkpoint runs/thinksound/model-unwrapped.ckpt \
  --thinksound-lora \
  --duration 12
```

`--thinksound-lora` 会自动选择 `predict_lora.py` 并向子进程设置 `N2M_THINKSOUND_LORA=1`。如果 rank、alpha 或目标层表达式与训练时不同，checkpoint 会因键或形状不匹配而失败，这是有意的快速失败行为。

基础模型推理不要传 `--thinksound-lora`：

```bash
nature2music compose outputs/recognition.json outputs/base.wav \
  --thinksound-dir external/ThinkSound
```

