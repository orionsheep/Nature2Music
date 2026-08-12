# Nature2Music

把自然的声音变成音乐：上传一段鸟叫、虫鸣或雨声，自训练的 **Fun-ASR + Qwen3-0.6B LoRA** 模型把它识别成结构化的声源信息（15 类声源），再由翻译层转成音乐语言，最终由 **Google Lyria 3 Pro** 生成一段可循环的自然主题音乐。

- 在线体验：<https://music.orionsheep.com>
- 识别模型（魔搭开源）：<https://modelscope.cn/models/OrionSheep/Nature2Music-FunASR-Bioacoustic>
- 项目图文介绍：[`docs/项目说明.md`](docs/项目说明.md) / [`docs/Nature2Music-项目说明.pdf`](docs/Nature2Music-项目说明.pdf)

> 重要边界：ASR 负责“音频到结构化文本”，不能直接生成音乐。早期方案曾用 ThinkSound 做音频合成（见第 4 节，已弃用），生产环境已切换为 Lyria 3 Pro API。

## 系统结构

```mermaid
flowchart LR
    A["鸟叫/虫鸣/自然录音"] --> B["数据清洗与按原始录音分组切分"]
    B --> C["Fun-ASR 音频编码器"]
    C --> D["Qwen3 解码器 LoRA + 音频适配器"]
    D --> E["物种、类群、叫声类型、环境事件 JSON"]
    A --> F["时长、能量、频谱中心、主频、节拍估计"]
    E --> G["音乐编排提示词"]
    F --> G
    G --> H["Google Lyria 3 Pro（API）"]
    H --> I["自然主题音乐 WAV"]
```

模型职责是分开的：

- Fun-ASR：封闭物种集合内的音频到文本分类/转写。训练标签是严格 JSON，而不是把鸟鸣误写成人类语言。
- Lyria 3 Pro：根据物种语义、节奏和编曲风格提示词生成音乐（生产环境通过 API 调用）。
- LoRA：Fun-ASR 内部 Qwen3 使用 PEFT LoRA。
- 音频适配器：默认采用“Qwen LoRA + 解冻 Fun-ASR audio adaptor”的混合参数高效配置。严格只训练 LLM LoRA 更省显存，但通常不如混合配置适合跨越“语音→生物声”的域迁移。

## 0. Web 应用

`webapp/` 是完整的线上服务（对应 <https://music.orionsheep.com>）：

```bash
cd webapp
./启动.sh   # 或: python server.py
```

- `server.py`：HTTP 服务与识别接口，加载 Fun-ASR 识别模型。
- `worker_generate.py`：后台生成 worker，调用 Lyria 3 Pro（需要 `GEMINI_API_KEY` 及可用的网络代理）。
- `static/`：前端页面与场景素材。

识别模型权重不随仓库分发，请从[魔搭](https://modelscope.cn/models/OrionSheep/Nature2Music-FunASR-Bioacoustic)下载后放到 `runs/funasr-bioacoustic/`。

## 1. 环境安装

推荐在 Linux + NVIDIA GPU 上训练；Windows 可以进行数据准备和推理，训练脚本也提供了 PowerShell 版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[asr,lora,dev]"
```

安装上游模型：

```bash
git clone https://github.com/QwenAudio/Fun-ASR.git external/Fun-ASR
git clone --branch master https://github.com/QwenAudio/ThinkSound.git external/ThinkSound

# 按 ThinkSound 官方说明安装依赖并下载 ckpts
cd external/ThinkSound
pip install thinksound
git lfs install
git clone https://huggingface.co/liuhuadai/ThinkSound ckpts
```

ThinkSound 需要 FFmpeg，并且官方建议 Python 3.10、`ffmpeg<7`。请先确认：

```bash
ffmpeg -version
nvidia-smi
```

## 2. 数据选择与许可

建议的三层数据：

1. 鸟类：使用 [Xeno-canto](https://xeno-canto.org/) 或 BirdCLEF 的合规下载版本。Xeno-canto 的许可按每条录音分别给出，训练清单必须保留 `recordist`、`source`、`license` 和原始 ID。
2. 昆虫：可使用 [InsectSet459](https://zenodo.org/records/18554693)；先核对记录页当前版本的许可和引用要求。
3. 环境负样本/背景：[ESC-50](https://github.com/karolpiczak/ESC-50) 可用于基础环境事件和“非目标物种”样本，但只有 2,000 条，不能替代大规模物种数据。

数据许可与模型许可需要分别检查。ThinkSound [官方仓库](https://github.com/QwenAudio/ThinkSound)注明研究/教育使用限制，并包含受 Stability AI Community License 约束的 VAE；商业使用不能仅依据仓库顶层的 Apache-2.0 字样判断。

### CSV 格式

参考 `examples/metadata.example.csv`。必需字段：

```text
audio_path,species,group,source,recording_id
```

推荐同时保留：`scientific_name`、中英文常用名、叫声类型、背景物种、地点、录音者、许可、切片起止时间。

物种切分不能随机按切片进行。来自同一个原始录音的多个 10 秒切片必须进入同一个 split，否则测试准确率会虚高。

```bash
nature2music manifest-from-csv metadata.csv data/processed/manifest.jsonl \
  --audio-root /datasets/nature

nature2music split-manifest \
  data/processed/manifest.jsonl \
  data/processed/manifest.split.jsonl \
  --validation-ratio 0.1 --test-ratio 0.1 --seed 42

nature2music validate-manifest data/processed/manifest.split.jsonl \
  --min-train-per-species 20
```

生产实验建议每个物种至少 50–100 条相互独立的原始录音，并按地点/日期做外部验证。类别极不平衡时应使用分层采样或类别加权，不要简单复制同一录音的切片。

## 3. Fun-ASR 物种识别 LoRA

导出官方 Fun-ASR ChatML 格式：

```bash
nature2music export-funasr \
  data/processed/manifest.split.jsonl \
  data/processed/funasr
```

每条训练答案类似：

```json
{
  "group": "bird",
  "species": "eurasian_blackbird",
  "scientific_name": "Turdus merula",
  "common_name_zh": "乌鸫",
  "common_name_en": "Eurasian Blackbird",
  "call_type": "song",
  "background": ["wind", "leaves"],
  "confidence": 1.0
}
```

启动 LoRA。脚本会从官方 `model.py` 生成 `model_lora.py`，不会覆盖上游文件：

```bash
bash scripts/train_funasr_lora.sh \
  external/Fun-ASR \
  data/processed/funasr \
  runs/funasr-bioacoustic
```

Windows：

```powershell
./scripts/train_funasr_lora.ps1 `
  -Upstream external/Fun-ASR `
  -DataDir data/processed/funasr `
  -OutputDir runs/funasr-bioacoustic
```

默认训练参数：LoRA `r=16, alpha=32, dropout=0.05`，目标为 Qwen 的注意力和 MLP 投影；冻结音频编码器，训练 audio adaptor。显存不足时先减小 token batch size，再考虑冻结 audio adaptor。

单文件识别：

```bash
nature2music identify sample.wav \
  --model runs/funasr-bioacoustic/best-model \
  --remote-code external/Fun-ASR/model_lora.py \
  --output outputs/recognition.json
```

推理输出无法解析为 JSON 时命令会明确失败，避免把幻觉文本传入音乐生成器。

## 4. ThinkSound 数据与 LoRA（已弃用，仅作记录）

> 早期方案使用 ThinkSound 做音频合成，实测产出质量不满足要求，生产环境已改为 Google Lyria 3 Pro。本节内容仅保留作实验记录，新部署可跳过。

先为 ThinkSound 生成 `id, caption, caption_cot, audio_path` 数据表：

```bash
nature2music export-thinksound \
  data/processed/manifest.split.jsonl \
  data/processed/thinksound \
  --stage-mode hardlink
```

然后使用 ThinkSound 官方音频特征提取器：

```bash
torchrun --nproc_per_node=1 external/ThinkSound/data_utils/extract_training_audio.py \
  --root data/processed/thinksound/train/audio \
  --tsv_path data/processed/thinksound/train/pairs.csv \
  --save-dir data/processed/thinksound/train/features \
  --duration_sec 10 \
  --audio_samples 441000
```

创建 LoRA 版训练入口：

```bash
bash scripts/prepare_thinksound_lora.sh external/ThinkSound
export N2M_THINKSOUND_LORA=1
export N2M_THINKSOUND_LORA_R=16
export N2M_THINKSOUND_LORA_ALPHA=32
```

复制上游 `scripts/train.sh` 的正式训练命令，把入口 `train.py` 改成 `train_lora.py`。适配器在载入基础 checkpoint 之后注入，并在匹配不到目标层时立即失败。

两种 ThinkSound 训练集不可混淆：

- 原始鸟鸣/虫鸣作为目标音频，只会增强自然声与 Foley 生成能力。
- 若目标是“自然声音→音乐”，训练目标必须是获得授权的音乐化结果，并配有物种/节奏/配器提示词。只有自然录音而没有音乐目标时，建议先不微调 ThinkSound，直接使用本项目的结构化提示词生成。

## 5. 生成音乐

生产环境（webapp）由 `worker_generate.py` 把识别 JSON 转成音乐提示词并调用 **Lyria 3 Pro** 生成音频，无需本地 GPU。

命令行下只构造提示词，可先检查识别结果如何影响编曲：

```bash
nature2music build-prompt outputs/recognition.json \
  --style "ambient Chinese orchestral music"
```

以下 `compose` / `run` 子命令基于已弃用的 ThinkSound 本地方案，仅作记录：

```bash
nature2music compose outputs/recognition.json outputs/nature_music.wav \
  --thinksound-dir external/ThinkSound \
  --style "ambient Chinese orchestral music" \
  --duration 12

nature2music run sample.wav outputs/nature_music.wav \
  --model runs/funasr-bioacoustic/best-model \
  --remote-code external/Fun-ASR/model_lora.py \
  --thinksound-dir external/ThinkSound \
  --style "ambient Chinese orchestral music" \
  --report outputs/run_report.json
```

报告包含物种、置信度、主频、频谱中心、估计 BPM、最终 caption 与 CoT，便于复现实验。

## 6. 评估标准

识别模型至少报告：

- species macro-F1、balanced accuracy、Top-1/Top-5；
- bird/insect/environment 分层指标；
- 按未见地点和未见录音设备划分的测试结果；
- `unknown`/非目标声音的拒识率；
- 多物种混合录音的 mAP，而不是强迫只输出一个物种。

生成模型建议报告 FAD/CLAP 相似度、人工 MOS、物种语义一致性，以及“是否错误复制训练录音”的最近邻检查。识别置信度低于阈值时应输出“未知类群”，不要自动写入具体物种名称。

## 7. 已知限制

- 当前 Fun-ASR 流程是单条录音输出一个主物种；真实生态录音经常是多标签问题，生产版本应增加时间切片、事件检测和多标签聚合。
- Fun-ASR 的语音前端并非专门为高频昆虫或超声动物设计。对 8 kHz 以上关键信息的物种，应保留 48/96 kHz 原始文件并增加高采样率生物声编码分支；蝙蝠超声不在当前范围内。
- ThinkSound 是 Any2Audio/声景生成框架，不是专用 MIDI 作曲器。它能生成音乐化音频，但旋律结构和长时一致性需要人工评审或后续音乐模型。
- 本项目不自动下载受不同许可约束的数据，也不将上游模型权重打包进仓库。

## 上游依据

- [Fun-ASR 官方仓库与微调说明](https://github.com/QwenAudio/Fun-ASR)
- [FunASR 工具箱](https://github.com/modelscope/FunASR)
- [ThinkSound 官方仓库](https://github.com/QwenAudio/ThinkSound)
- [ThinkSound 项目页](https://thinksound-project.github.io/)

