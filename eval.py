import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data_loader import DatasetProcessor
from src.reward_funcs import (
    format_structure_reward,
    f1_score_intent_reward,
    f1_score_emotion_reward,
    accuracy_intent_reward,
    accuracy_emotion_reward,
    category_validity_intent_reward,
    category_validity_emotion_reward,
    squared_match_intent_reward,
    squared_match_emotion_reward,
    thinking_efficiency_reward,
    set_global_params,
)
import yaml
from loguru import logger
from tqdm import tqdm

# Define which reward functions to use - comment out to disable
REWARD_FUNCTIONS = [
    ("format_structure_reward", format_structure_reward),
    ("f1_score_intent_reward", f1_score_intent_reward),
    ("f1_score_emotion_reward", f1_score_emotion_reward),
    ("accuracy_intent_reward", accuracy_intent_reward),
    ("accuracy_emotion_reward", accuracy_emotion_reward),
    ("category_validity_intent_reward", category_validity_intent_reward),
    ("category_validity_emotion_reward", category_validity_emotion_reward),
    ("squared_match_intent_reward", squared_match_intent_reward),
    ("squared_match_emotion_reward", squared_match_emotion_reward),
    ("thinking_efficiency_reward", thinking_efficiency_reward),
]


def generate_response(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.1):
    """Generate response from model with given temperature."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


def evaluate_model(
        model_path,
        config_path="train_config.yaml",
        categories_path="categories.yaml",
        prompt_path="prompt.md",
        output_csv="evaluation_results.csv",
):
    """Evaluate model on test dataset and save results to CSV."""

    # Load config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    train_config = config["training"]
    max_completion_length = train_config.get("max_completion_length", 1024)

    logger.info(f"Loading model from {model_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Load test dataset
    processor = DatasetProcessor(categories_path, prompt_path, tokenizer)
    _, test_dataset, intent_categories, emotion_categories = (
        processor.load_and_process_dataset(test_size=0.1)
    )

    # Set global params for reward functions
    set_global_params(intent_categories, emotion_categories, max_completion_length)

    logger.info(f"Evaluating on {len(test_dataset)} test samples")
    logger.info(f"Active reward functions: {[name for name, _ in REWARD_FUNCTIONS]}")

    # Prepare results storage
    results = []

    # Evaluate each sample
    for idx, sample in enumerate(tqdm(test_dataset, desc="Evaluating")):
        prompt = sample["prompt"]
        ground_truth = sample["answer"]

        # Generate response
        generation = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_completion_length,
            temperature=0.1
        )

        # Store basic info
        result = {
            "speech": prompt,
            "generation": generation,
            "ground_truth": ground_truth,
        }

        # Calculate each enabled reward
        completions = [generation]
        prompts = [prompt]
        answers = [ground_truth]

        for reward_name, reward_func in REWARD_FUNCTIONS:
            score = reward_func(prompts=prompts, completions=completions, answer=answers)[0]
            result[reward_name] = score

        results.append(result)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # Calculate and log average scores
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)

    score_columns = [col for col in df.columns if col not in ["speech", "generation", "ground_truth"]]
    for col in score_columns:
        avg_score = df[col].mean()
        logger.info(f"{col}: {avg_score:.4f}")

    logger.info(f"\nResults saved to {output_csv}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained model on test dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model on HuggingFace Hub or local path"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="train_config.yaml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--categories_path",
        type=str,
        default="/workspace/proactive-ai/categories.yaml",
        help="Path to categories file"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="/workspace/proactive-ai/prompt.md",
        help="Path to prompt file"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="evaluation_results.csv",
        help="Output CSV file path"
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        config_path=args.config_path,
        categories_path=args.categories_path,
        prompt_path=args.prompt_path,
        output_csv=args.output_csv,
    )