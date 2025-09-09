import torch
from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

CONFIG = {
    "model_name": "microsoft/Phi-4-mini-instruct",
    "lora_rank": 4,
    "learning_rate": 5e-6,
    "batch_size": 1,
    "max_steps": 10,
    "max_prompt_length": 256,
    "max_completion_length": 100,
    "max_length": 512,
}


def placeholder_dataset():
    """Placeholder"""
    data = []

    # 10 placeholder preference pairs
    examples = [
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a subset of artificial intelligence.",
            "rejected": "Machine learning is just computers doing stuff with numbers."
        } for _ in range(10)
    ]

    return Dataset.from_list(examples)


def main():
    print("Starting DPO training with multi-GPU support")

    dataset = placeholder_dataset()
    print(f"Dataset loaded with {len(dataset)} preference pairs")

    # Configure LoRA
    peft_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_rank"],
        target_modules=["gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.1,
        bias="none",
        use_rslora=False,  # Disable for multi-GPU stability
    )

    # Training configuration
    training_args = DPOConfig(
        output_dir="outputs",
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=1,
        max_steps=CONFIG["max_steps"],
        max_prompt_length=CONFIG["max_prompt_length"],
        max_completion_length=CONFIG["max_completion_length"],
        max_length=CONFIG["max_length"],
        logging_steps=1,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",
        # Multi-GPU settings
        ddp_find_unused_parameters=False,  # Important for LoRA
        # Model loading args
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
        },
        # Mixed precision
        bf16=True,
        gradient_checkpointing=False,  # Set to False because Phi4 doesn't support
        # DPO specific settings
        beta=0.1,  # DPO temperature parameter
        loss_type=["sigmoid"],  # Default DPO loss
        # Optimizer settings
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
    )

    # Initialize trainer
    trainer = DPOTrainer(
        model=CONFIG["model_name"],
        peft_config=peft_config,
        args=training_args,
        train_dataset=dataset,
        processing_class=None,  # Will be auto-loaded from model
    )

    print("Starting training...")
    trainer.train()
    print("Training completed!")

    # Save the trained model
    trainer.save_model("dpo_phi4_model")
    print("Model saved as dpo_phi4_model")

if __name__ == "__main__":
    main()