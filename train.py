import torch
from trl import GRPOConfig, GRPOTrainer
from src.data_loader import DatasetProcessor
from src.reward_funcs import format_structure_reward, hamming_loss_reward, f1_score_reward, accuracy_reward, category_validity_reward, set_categories
import yaml
from loguru import logger
import wandb


def main(
        config_path='train_config.yaml',
        categories_path='categories.yaml',
        prompt_path='prompt.md'
    ):
    with open(config_path, 'r') as file:
        train_config = yaml.safe_load(file)
    logger.info("Starting multi-label classification training with GRPO")

    wandb.init(
        project="grpo-classification",
        name=f"proactive_grpo_classification"
    )
    logger.info(f"W&B dashboard: {wandb.run.url}")

    # Load dataset
    processor = DatasetProcessor(categories_path, prompt_path)
    train_dataset, test_dataset, intent_categories, emotion_categories = processor.load_and_process_dataset(test_size=0.1)
    set_categories(intent_categories, emotion_categories)

    logger.info(f"Dataset loaded with {len(train_dataset)} + {len(test_dataset)} examples")
    logger.info(f"Intent categories: {intent_categories}")
    logger.info(f"Emotion categories: {emotion_categories}")

    # Training configuration
    training_args = GRPOConfig(
        output_dir="outputs",
        learning_rate=float(train_config["learning_rate"]),
        per_device_train_batch_size=train_config["batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        max_steps=train_config["max_steps"],
        num_generations=train_config["num_generations"],
        max_prompt_length=train_config["max_prompt_length"],
        max_completion_length=train_config["max_completion_length"],
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        max_grad_norm=1.0,
        # Monitoring
        report_to="wandb",
        run_name=f"proactive_grpo_classification",
        # Model loading args
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
        },
        # Mixed precision
        bf16=True,
        gradient_checkpointing=False,  # Disable - use FSDP activation checkpointing instead
        # Additional GRPO settings
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",

        # Data loading
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # Checkpointing
        save_total_limit=3,
        load_best_model_at_end=False,
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=train_config["model_name"],
        reward_funcs=[
            accuracy_reward,
            format_structure_reward,
            category_validity_reward,
            f1_score_reward,
            hamming_loss_reward
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")

    # Save the trained model
    trainer.save_model(f"proactive_grpo_classification")
    logger.info(f"Model saved as proactive_grpo_classification")


if __name__ == "__main__":
    main(
        config_path='train_config.yaml',
        categories_path='/workspace/proactive-ai/categories.yaml',
        prompt_path='/workspace/proactive-ai/prompt.md'
    )