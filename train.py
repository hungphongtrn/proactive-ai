import torch
from trl import GRPOConfig, GRPOTrainer
from src.data_loader import DatasetProcessor
from src.reward_funcs import multi_label_accuracy_reward, format_reward_func, category_validity_reward, set_categories, hamming_loss_reward, f1_score_reward
import yaml


def main(
        config_path='train_config.yaml',
        categories_path='categories.yaml',
        prompt_path='prompt.md'
    ):
    with open(config_path, 'r') as file:
        train_config = yaml.safe_load(file)
    print("Starting multi-label classification training with GRPO")

    # Load dataset
    data_processor = DatasetProcessor(categories_path, prompt_path)
    dataset, global_categories = data_processor.load_and_process_dataset(train_config["output_col"])
    set_categories(global_categories)

    print(f"Dataset loaded with {len(dataset)} examples")
    print(f"Categories: {global_categories}")

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
        logging_steps=5,
        save_steps=50,
        max_grad_norm=1.0,
        report_to="none",
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
        weight_decay=0.1,
        warmup_ratio=0.1,
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
            multi_label_accuracy_reward,
            format_reward_func,
            category_validity_reward,
            f1_score_reward,
            hamming_loss_reward
        ],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training completed!")

    # Save the trained model
    trainer.save_model(f"grpo_{train_config['output_col']}_classification")
    print(f"Model saved as grpo_{train_config['output_col']}_classification")


if __name__ == "__main__":
    main(
        config_path='train_config.yaml',
        categories_path='/workspace/proactive-ai/categories.yaml',
        prompt_path='/workspace/proactive-ai/prompt.md'
    )