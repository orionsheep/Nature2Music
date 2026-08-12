# Nature2Music 交付说明

## 已完成

- 鸟类、昆虫、两栖类、哺乳类、环境声音统一 JSONL 清单。
- 按全局原始录音组和多物种标签组合切分，防止相同录音片段泄漏到不同集合。
- Fun-ASR-Nano ChatML 训练数据导出，包含 `speech_length`、`text_length`。
- Fun-ASR 内部 Qwen3 解码器 PEFT LoRA 注入；默认同时训练 audio adaptor。
- 结构化物种 JSON 解析与错误拒绝。
- 输入音频 RMS、频谱中心、主频、节拍估计。
- 识别结果到 ThinkSound caption/CoT 编曲条件的转换。
- ThinkSound 扩散 Transformer LoRA 训练与推理同构补丁。
- 基础模型与 LoRA checkpoint 两种 ThinkSound 推理入口。
- Windows PowerShell / Linux Bash 训练脚本、示例配置、中文 README 和专项 LoRA 文档。

## 验证结果

- Python 语法编译：通过。
- 单元测试：11 项通过。
- LoRA 数值测试：当前 Codex 内置 Python 未安装 PyTorch，因此跳过；安装项目 `asr,lora` 依赖后可运行。
- CLI 冒烟测试：CSV → manifest → 分组切分 → 泄漏审计 → Fun-ASR/ThinkSound 导出，全流程通过。
- Fun-ASR 动态批处理长度字段：通过额外断言。

## 尚未在本机执行

没有下载大模型权重或数据集，也没有在 GPU 上实际训练/生成 WAV。实际训练需要用户提供合规数据、Fun-ASR/ThinkSound checkpoint、FFmpeg 和 NVIDIA GPU。

## 推荐启动顺序

1. 解压项目，阅读根目录 `README.md`。
2. 准备 `metadata.csv`，运行 `manifest-from-csv`、`split-manifest`、`validate-manifest`。
3. 先训练并评估 Fun-ASR 物种识别；确认 macro-F1 和未知类拒识达到要求。
4. 使用基础 ThinkSound 验证结构化提示词生成。
5. 只有在拥有“自然语义条件→授权音乐目标音频”配对数据时，再训练 ThinkSound 音乐 LoRA。

## 上游资料

- [Fun-ASR](https://github.com/QwenAudio/Fun-ASR)
- [FunASR 工具箱](https://github.com/modelscope/FunASR)
- [ThinkSound](https://github.com/QwenAudio/ThinkSound)
- [ThinkSound 项目页](https://thinksound-project.github.io/)
- [InsectSet459](https://zenodo.org/records/18554693)
- [ESC-50](https://github.com/karolpiczak/ESC-50)
