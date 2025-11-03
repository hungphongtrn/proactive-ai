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
    parse_structured_response,
)
import yaml
from loguru import logger
from tqdm import tqdm
from vllm import SamplingParams, LLM
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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


def generate_responses_batch(model, prompts, max_new_tokens=1024, temperature=0.1, use_vllm=False, use_tqdm=True):
    """Generate responses for multiple prompts efficiently using vLLM batching."""
    if use_vllm:
        # vLLM handles batching automatically
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)

        if use_tqdm:
            # Use smaller chunks for better progress tracking
            # This allows vLLM to optimize while still showing progress
            chunk_size = 32  # Process 32 prompts at a time for progress updates
            all_outputs = []

            total_chunks = (len(prompts) + chunk_size - 1) // chunk_size
            pbar = tqdm(total=len(prompts), desc="Generating with vLLM")

            for i in range(0, len(prompts), chunk_size):
                chunk_prompts = prompts[i:i + chunk_size]
                chunk_outputs = model.generate(chunk_prompts, sampling_params)
                all_outputs.extend(chunk_outputs)
                pbar.update(len(chunk_prompts))

            pbar.close()
            return [output.outputs[0].text for output in all_outputs]
        else:
            outputs = model.generate(prompts, sampling_params)
            return [output.outputs[0].text for output in outputs]
    else:
        # For non-vLLM case, process individually (no batching optimization)
        responses = []
        iterator = tqdm(prompts, desc="Generating responses") if use_tqdm else prompts
        for prompt in iterator:
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
            responses.append(response)
        return responses

def generate_response_api(prompt, model="gpt-5", max_tokens=1024, temperature=0.1):
    """Generate response using OpenAI API."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response.choices[0].message.content


def evaluate_model(
        model_path=None,
        config_path="train_config.yaml",
        categories_path="categories.yaml",
        prompt_path="prompt.md",
        output_csv="evaluation_results.csv",
        use_vllm=False,
        api_model=None,
        tokenizer_name=None,
        use_tqdm=True
):
    """Evaluate model on test dataset and save results to CSV."""

    # Load config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    trainer_config = config["trainer"]
    max_completion_length = trainer_config.get("max_completion_length", 1024)

    # Initialize model based on mode
    model = None
    tokenizer = None

    if api_model is None:
        logger.info(f"Loading model from {model_path}")

        if use_vllm:
            model = LLM(model=model_path, dtype="bfloat16")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            model.eval()
    else:
        logger.info(f"Using API model: {api_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_path
        )

    # Load test dataset
    processor = DatasetProcessor(categories_path, prompt_path, tokenizer)
    _, test_dataset, intent_categories, emotion_categories = (
        processor.load_and_process_dataset(test_size=0.1)
    )

    # Set global params for reward functions
    set_global_params(intent_categories, emotion_categories, max_completion_length)

    logger.info(f"Evaluating on {len(test_dataset)} test samples")
    logger.info(f"Active reward functions: {[name for name, _ in REWARD_FUNCTIONS]}")
    if use_vllm:
        logger.info(f"Using vLLM with continuous batching (sending all prompts at once)")

    # Prepare results storage
    results = []

    # Process all prompts at once with vLLM
    if use_vllm and api_model is None:
        # vLLM handles its own optimal batching, send all prompts at once
        logger.info(f"Sending all {len(test_dataset)} prompts to vLLM at once")

        # Prepare all data
        all_prompts = [sample["prompt"] for sample in test_dataset]
        all_ground_truths = [sample["answer"] for sample in test_dataset]

        # Generate all responses at once
        all_generations = generate_responses_batch(
            model,
            all_prompts,
            max_new_tokens=max_completion_length,
            temperature=0.1,
            use_vllm=True,
            use_tqdm=use_tqdm
        )

        # Process each result
        for sample_idx, (sample, generation, ground_truth) in enumerate(tqdm(zip(test_dataset, all_generations, all_ground_truths), total=len(test_dataset), desc="Processing Results")):
            # Parse prediction and ground truth
            parsed = parse_structured_response(generation)
            pred_intent = parsed["intent"]
            pred_emotion = parsed["emotion"]

            gt_intent, gt_emotion = ground_truth.split("|") if "|" in ground_truth else (ground_truth, "")

            result = {
                "speech": all_prompts[sample_idx],
                "generation": generation,
                "ground_truth": ground_truth,
                "predict_intent": pred_intent.strip(),
                "predict_emotion": pred_emotion.strip(),
                "groundtruth_intent": gt_intent.strip(),
                "groundtruth_emotion": gt_emotion.strip(),
            }

            # Calculate each enabled reward (still individual for reward functions)
            completions = [generation]
            prompts = [all_prompts[sample_idx]]
            answers = [ground_truth]

            for reward_name, reward_func in REWARD_FUNCTIONS:
                score = reward_func(prompts=prompts, completions=completions, answer=answers)[0]
                result[reward_name] = score

            results.append(result)
    else:
        # Original individual processing for non-vLLM or API models
        for idx, sample in enumerate(tqdm(test_dataset, desc="Evaluating")):
            prompt = sample["prompt"]
            ground_truth = sample["answer"]

            # Generate response
            if api_model is not None:
                generation = generate_response_api(
                    prompt,
                    model=api_model,
                    max_tokens=max_completion_length,
                    temperature=0.1
                )
            else:
                generation = generate_responses_batch(
                    model,
                    [prompt],
                    max_new_tokens=max_completion_length,
                    temperature=0.1,
                    use_vllm=False,
                    use_tqdm=use_tqdm
                )[0]

            # Parse prediction and ground truth
            parsed = parse_structured_response(generation)
            pred_intent = parsed["intent"]
            pred_emotion = parsed["emotion"]

            gt_intent, gt_emotion = ground_truth.split("|") if "|" in ground_truth else (ground_truth, "")

            result = {
                "speech": prompt,
                "generation": generation,
                "ground_truth": ground_truth,
                "predict_intent": pred_intent.strip(),
                "predict_emotion": pred_emotion.strip(),
                "groundtruth_intent": gt_intent.strip(),
                "groundtruth_emotion": gt_emotion.strip(),
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

    score_columns = [col for col in df.columns if
                     col not in ["speech", "generation", "ground_truth", "predict_intent", "predict_emotion",
                                 "groundtruth_intent", "groundtruth_emotion"]]
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
        default="categories.yaml",
        help="Path to categories file"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompt.md",
        help="Path to prompt file"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="evaluation_results.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--api_model",
        type=str,
        default=None
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--use_tqdm",
        action="store_true",
        default=True,
        help="Show progress bars during generation and evaluation"
    )
    parser.add_argument(
        "--no_tqdm",
        action="store_true",
        help="Disable progress bars"
    )
    args = parser.parse_args()

    # Handle tqdm arguments
    use_tqdm = not args.no_tqdm and args.use_tqdm

    evaluate_model(
        model_path=args.model_path,
        config_path=args.config_path,
        categories_path=args.categories_path,
        prompt_path=args.prompt_path,
        output_csv=args.output_csv,
        use_vllm=True,
        api_model=None,
        tokenizer_name='Qwen/Qwen3-1.7B',
        use_tqdm=use_tqdm
    )