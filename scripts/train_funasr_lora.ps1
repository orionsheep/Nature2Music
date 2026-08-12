param(
    [Parameter(Mandatory = $true)][string]$Upstream,
    [Parameter(Mandatory = $true)][string]$DataDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$ModelId = "FunAudioLLM/Fun-ASR-Nano-2512",
    [int]$Gpus = 1
)

$ErrorActionPreference = "Stop"
$upstreamPath = (Resolve-Path -LiteralPath $Upstream).Path
$dataPath = (Resolve-Path -LiteralPath $DataDir).Path
$patchedModel = Join-Path $upstreamPath "model_lora.py"

python -m nature2music.cli patch-funasr-lora `
    (Join-Path $upstreamPath "model.py") $patchedModel

$trainer = (Get-Command funasr-train-ds).Source
& torchrun --nproc_per_node=$Gpus $trainer `
    "++model=$ModelId" `
    "++trust_remote_code=true" `
    "++remote_code=$patchedModel" `
    "++train_data_set_list=$(Join-Path $dataPath 'train.jsonl')" `
    "++valid_data_set_list=$(Join-Path $dataPath 'validation.jsonl')" `
    "++dataset_conf.data_split_num=1" `
    "++dataset_conf.batch_sampler=BatchSampler" `
    "++dataset_conf.batch_size=6000" `
    "++dataset_conf.batch_type=token" `
    "++dataset_conf.num_workers=4" `
    "++train_conf.max_epoch=30" `
    "++train_conf.validate_interval=1000" `
    "++train_conf.save_checkpoint_interval=1000" `
    "++train_conf.resume=true" `
    "++train_conf.use_deepspeed=false" `
    "++optim_conf.lr=0.0002" `
    "++audio_encoder_conf.freeze=true" `
    "++audio_adaptor_conf.freeze=false" `
    "++llm_conf.freeze=true" `
    "++llm_conf.lora_conf.enabled=true" `
    "++llm_conf.lora_conf.r=16" `
    "++llm_conf.lora_conf.alpha=32" `
    "++llm_conf.lora_conf.dropout=0.05" `
    "++output_dir=$OutputDir"

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

