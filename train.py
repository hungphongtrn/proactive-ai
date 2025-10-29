import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import torch
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from src.data_loader import DatasetProcessor
from src.reward_funcs import (
    f1_score_intent_reward,
    f1_score_emotion_reward,
    accuracy_intent_reward,
    accuracy_emotion_reward,
    format_structure_reward,
    category_validity_intent_reward,
    category_validity_emotion_reward,
    squared_match_intent_reward,
    squared_match_emotion_reward,
    thinking_efficiency_reward,
    set_global_params
)
import yaml
from loguru import logger


REWARD_FUNCTION_REGISTRY = {
    "f1_score_intent_reward": f1_score_intent_reward,
    "f1_score_emotion_reward": f1_score_emotion_reward,
    "accuracy_intent_reward": accuracy_intent_reward,
    "accuracy_emotion_reward": accuracy_emotion_reward,
    "format_structure_reward": format_structure_reward,
    "category_validity_intent_reward": category_validity_intent_reward,
    "category_validity_emotion_reward": category_validity_emotion_reward,
    "squared_match_intent_reward": squared_match_intent_reward,
    "squared_match_emotion_reward": squared_match_emotion_reward,
    "thinking_efficiency_reward": thinking_efficiency_reward,
}


def main(
    config_path="train_config.yaml",
    categories_path="categories.yaml",
    prompt_path="prompt.md",
):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    model_config = config["model"]
    lora_config = config["lora"]
    train_config = config["trainer"]
    reward_config = config["reward_funcs"]

    logger.info(
        "Starting multi-label classification training with GRPO using Unsloth + LoRA"
    )

    # Load LoRA configuration
    # Load model and tokenizer with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        **model_config,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        **lora_config,
    )

    # Load dataset
    processor = DatasetProcessor(categories_path, prompt_path, tokenizer)
    train_dataset, test_dataset, intent_categories, emotion_categories = (
        processor.load_and_process_dataset(test_size=0.1)
    )

    max_completion_length = train_config.get("max_completion_length", 1024)
    set_global_params(intent_categories, emotion_categories, max_completion_length)

    logger.info(
        f"Dataset loaded with {len(train_dataset)} + {len(test_dataset)} examples"
    )
    logger.info(f"Intent categories: {intent_categories}")
    logger.info(f"Emotion categories: {emotion_categories}")

    if "model_init_kwargs" in train_config:
        if "torch_dtype" in train_config["model_init_kwargs"]:
            dtype_str = train_config["model_init_kwargs"]["torch_dtype"]
            train_config["model_init_kwargs"]["torch_dtype"] = getattr(torch, dtype_str)

    if "learning_rate" in train_config:
        train_config["learning_rate"] = float(train_config["learning_rate"])

    training_args = GRPOConfig(**train_config)

    enabled_rewards = []
    for rf_config in reward_config:
        if rf_config["enabled"]:
            func = REWARD_FUNCTION_REGISTRY[rf_config["name"]]
            enabled_rewards.append(func)

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=enabled_rewards,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")

    # Save the trained LoRA model
    logger.info("Saving LoRA model...")
    model.save_pretrained(train_config["output_dir"])
    tokenizer.save_pretrained(train_config["output_dir"])

    # Optional: Save to 16bit for GGUF /hf merge
    if train_config.get("save_16bit", False):
        model.save_pretrained_merged(
            train_config["output_dir"] + "_merged",
            tokenizer,
            save_method="merged_16bit",
        )

    # Optional: Save to 4bit
    if train_config.get("save_4bit", False):
        model.save_pretrained_merged(
            train_config["output_dir"] + "_merged_4bit",
            tokenizer,
            save_method="merged_4bit_forced",
        )

    model.push_to_hub_merged(f"hungphongtran/{train_config['run_name']}", save_method = "merged_16bit")


if __name__ == "__main__":
    main(
        config_path="./train_config.yaml",
        categories_path="./categories.yaml",
        prompt_path="./prompt.md",
    )
