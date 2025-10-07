import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
from src.data_loader import DatasetProcessor
from src.reward_funcs import (
    format_structure_reward,
    hamming_loss_reward,
    f1_score_reward,
    accuracy_reward,
    category_validity_reward,
    set_global_params,
    squared_match_reward,
    thinking_efficiency_reward
)
import yaml
from loguru import logger


REWARD_FUNCTION_REGISTRY = {
    "f1_score_reward": f1_score_reward,
    "accuracy_reward": accuracy_reward,
    "format_structure_reward": format_structure_reward,
    "category_validity_reward": category_validity_reward,
    "hamming_loss_reward": hamming_loss_reward,
    "squared_match_reward": squared_match_reward,
    "thinking_efficiency_reward": thinking_efficiency_reward
}


def main(
    config_path="train_config.yaml",
    categories_path="categories.yaml",
    prompt_path="prompt.md",
):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    model_name = config["model_name"]
    train_config = config["training"]
    reward_config = config["reward_funcs"]

    logger.info("Starting multi-label classification training with GRPO")

    # Load model and tokenizer
    model = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    # Save the trained model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(train_config["output_dir"])


if __name__ == "__main__":
    main(
        config_path="train_config.yaml",
        categories_path="/workspace/proactive-ai/categories.yaml",
        prompt_path="/workspace/proactive-ai/prompt.md",
    )
