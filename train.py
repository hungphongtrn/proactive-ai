import torch
from trl import GRPOConfig, GRPOTrainer
from src.data_loader import load_and_process_dataset
from src.reward_funcs import multi_label_accuracy_reward, format_reward_func, category_validity_reward, set_categories

CONFIG = {
    "model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "output_col": "intent",  # Can be "intent" or "emotion"
    "learning_rate": 5e-6,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,  # Increase for effective batch size
    "max_steps": 10,
    "num_generations": 2,
    "max_prompt_length": 256,
    "max_completion_length": 100,
    "max_seq_length": 356,  # prompt + completion length
}

def main():
    print("Starting multi-label classification training with GRPO")

    # Load dataset
    dataset, global_categories = load_and_process_dataset(CONFIG["output_col"])
    set_categories(global_categories)

    print(f"Dataset loaded with {len(dataset)} examples")
    print(f"Categories: {global_categories}")

    # Training configuration
    training_args = GRPOConfig(
        output_dir="outputs",
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        max_steps=CONFIG["max_steps"],
        num_generations=CONFIG["num_generations"],
        max_prompt_length=CONFIG["max_prompt_length"],
        max_completion_length=CONFIG["max_completion_length"],
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
        model=CONFIG["model_name"],
        reward_funcs=[
            multi_label_accuracy_reward,
            format_reward_func,
            category_validity_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training completed!")

    # Save the trained model
    trainer.save_model(f"grpo_{CONFIG['output_col']}_classification")
    print(f"Model saved as grpo_{CONFIG['output_col']}_classification")


def inference_example():
    """Example inference function - to be expanded later."""
    print("Inference functionality to be implemented...")
    # TODO: Implement inference logic
    pass


if __name__ == "__main__":
    main()