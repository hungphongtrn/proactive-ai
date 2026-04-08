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

## Evaluation

### Reward-based evaluation (eval.py)

```bash
# Evaluate a trained model
python eval.py --model_path <path_or_hf_id> --categories_path categories.yaml --prompt_path prompts/prompt.md

# Evaluate via OpenAI API
python eval.py --model_path <path> --api_model gpt-5 --tokenizer_name Qwen/Qwen3-1.7B
```

### Unified evaluation (src.eval)

Evaluates all columns (intent, emotion, speech_act, maxims, implicature) in a single run.

```bash
# Using default eval_config.yaml
python -m src.eval

# Custom config
python -m src.eval my_config.yaml
```

### Running all 6 scenarios for a model

The helper script runs all-columns + 5 per-column prompts:

```bash
bash run_model.sh "model/name:free" 3   # args: model, max_workers
```

Scenarios: `all_columns`, `speech_act_only`, `intent_only`, `emotion_only`, `maxims_only`, `implicature_only`

### Per-column experiments

```bash
python -m src.eval --tag all_columns
python -m src.eval --prompt prompts/prompt_eval_speech_act.md --tag speech_act_only --no-judge
python -m src.eval --prompt prompts/prompt_eval_intent.md --tag intent_only --no-judge
python -m src.eval --prompt prompts/prompt_eval_emotion.md --tag emotion_only --no-judge
python -m src.eval --prompt prompts/prompt_eval_maxims.md --tag maxims_only --no-judge
python -m src.eval --prompt prompts/prompt_eval_implicature_only.md --tag implicature_only
```

### Using OpenRouter models

```bash
python -m src.eval --provider openrouter --model "arcee-ai/trinity-large-preview:free" --tag all_columns --workers 3
```

### Repairing failed samples

Failed samples are saved to `{tag}_skipped.csv`. Use `--repair` to retry only those:

```bash
python -m src.eval --repair outputs/eval/{model_folder}/{tag}.csv \
  --provider openrouter --model "model/name" --prompt prompts/prompt_eval_{task}.md --workers 1
```

For free-tier models with rate limits, use `--workers 1` to avoid 429 errors.

### CLI overrides

`--model`, `--provider`, `--tag`, `--samples`, `--prompt`, `--judge-model`, `--no-judge`, `--workers`, `--split`, `--repair`

### Evaluation output

- Results: `outputs/eval/{tag}_{input}_{model}_{date}.csv`
- Skipped: `outputs/eval/{tag}_skipped.csv` (auto-created if samples fail)
- Metrics: F1 for multi-label (intent/emotion/speech_act), exact match for maxims, LLM-as-a-judge for implicature
