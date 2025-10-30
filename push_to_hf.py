#!/usr/bin/env python3
"""
Script to load model and tokenizer from outputs_merged_16bit and push to HuggingFace Hub
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
from huggingface_hub import HfApi, login

def push_model_to_hub(model_path, repo_id, private=True, commit_message="Upload trained model"):
    """
    Load model and tokenizer from local path and push to HuggingFace Hub
    """
    print(f"Loading model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Pushing to repository: {repo_id}")

    # Push to hub
    model.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message=commit_message
    )

    tokenizer.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message=commit_message
    )

    print(f"Successfully pushed model and tokenizer to {repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Push model to HuggingFace Hub")
    parser.add_argument("--model_path", type=str, default="outputs_merged_16bit",
                        help="Path to the model directory")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace Hub repository ID (e.g., 'username/model-name')")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Make repository private")
    parser.add_argument("--commit_message", type=str, default="Upload trained model",
                        help="Commit message for the push")

    args = parser.parse_args()

    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        return

    # Login to HuggingFace Hub
    try:
        login()
    except Exception as e:
        print(f"Error logging into HuggingFace Hub: {e}")
        print("Please run `huggingface-cli login` or set HF_TOKEN environment variable")
        return

    # Push model
    try:
        push_model_to_hub(
            model_path=args.model_path,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message
        )
    except Exception as e:
        print(f"Error pushing model: {e}")

if __name__ == "__main__":
    main()