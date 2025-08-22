import torch
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from src.data_loader import load_and_process_dataset
from src.reward_funcs import multi_label_accuracy_reward, format_reward_func, category_validity_reward, set_categories

CONFIG = {
    "model_name": "microsoft/Phi-4-mini-instruct",
    "output_col": "intent",  # Can be "intent" or "emotion"
    "lora_rank": 4,
    "learning_rate": 5e-6,
    "batch_size": 1,
    "max_steps": 10,
    "num_generations": 2,
    "max_prompt_length": 256,
    "max_completion_length": 100,
    "max_seq_length": 256,
}

def main():
    print("Starting multi-label classification training with GRPO")

    # Load dataset
    dataset, global_categories = load_and_process_dataset(CONFIG["output_col"])
    set_categories(global_categories)

    print(f"Dataset loaded with {len(dataset)} examples")
    print(f"Categories: {global_categories}")

    # Configure LoRA
    peft_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_rank"],
        target_modules=["gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        # ADD THESE LINES:
        lora_dropout=0.1,
        bias="none",
        use_rslora=False,  # Disable for multi-GPU stability
    )

    # Training configuration
    training_args = GRPOConfig(
        output_dir="outputs",
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=1,
        max_steps=CONFIG["max_steps"],
        num_generations=CONFIG["num_generations"],
        max_prompt_length=CONFIG["max_prompt_length"],
        max_completion_length=CONFIG["max_completion_length"],
        logging_steps=1,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",
        # Multi-GPU settings
        use_vllm=False,
        ddp_find_unused_parameters=False,  # Important for LoRA
        # dataloader_pin_memory=False,
        # Model loading args
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
        },
        # Mixed precision
        bf16=True,
        gradient_checkpointing=False,  # Set to False because Phi4 dont support
        # Additional GRPO settings
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=CONFIG["model_name"],
        peft_config=peft_config,
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