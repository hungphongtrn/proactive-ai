# Training Guide

## Prerequisites

Install dependencies using either method:
### Using uv (recommended)
```bash
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Using pip
```bash
pip install -r requirements.txt
```

## Configuration
- Model Configuration: Update `train_config.yaml` with your desired model and training parameters
- Categories: Define your classification categories in categories.yaml
- Prompt: Set your training prompt in prompt.md

## Training
### Single GPU
```bash
python train.py
```

### Multi-GPU
```bash
# For 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file fsdp_config.yaml train.py
```
Important: When changing models for multi-GPU training, update fsdp_transformer_layer_cls_to_wrap in fsdp_config.yaml to match your model's transformer layer class.
Output

## Saving
Model checkpoints: `outputs/` directory

Final model: `proactive_grpo_classification/`

Training logs: W&B dashboard (if enabled)
