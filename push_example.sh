#!/bin/bash

# Example usage of push_to_hf.py script
#
# Before running, make sure you:
# 1. Have logged in to HuggingFace: huggingface-cli login
# 2. Set your HF_TOKEN environment variable if needed
# 3. The outputs_merged_16bit directory exists

# Example 1: Push to a private repository
python push_to_hf.py \
    --model_path outputs_merged_16bit \
    --repo_id hungphongtran/qwen3-1.7B-proactive-3010  \
    --private false\
    --commit_message "Upload proactive AI GRPO model"

# Example 2: Push to a public repository
# python push_to_hf.py \
#     --model_path outputs_merged_16bit \
#     --repo_id your-username/your-model-name \
#     --no-private \
#     --commit_message "Upload proactive AI GRPO model - public release"