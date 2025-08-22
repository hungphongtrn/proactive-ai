from datasets import load_dataset, Dataset
from typing import List, Tuple


def load_and_process_dataset(output_col: str) -> Tuple[Dataset, List[str]]:
    """Load the dataset and extract unique categories for the specified column."""
    dataset = load_dataset("richelle05/maxims_50_golden_samples")["train"]
    all_categories = set()
    for item in dataset:
        if item[output_col]:
            categories_list = [cat.strip().lower() for cat in item[output_col].split(",")]
            all_categories.update(categories_list)

    global_categories = sorted(list(all_categories))
    print(f"Found {len(global_categories)} unique categories: {global_categories}")

    # Create category definitions (placeholder - should be updated with real definitions)
    category_definitions = "\n".join([f"{cat.upper()}: Definition for {cat}" for cat in global_categories])

    # Create the prompt template. NEED TO UPDATE THIS WITH REAL DEFINITIONS
    def create_prompt(example):
        category_list = ", ".join(global_categories)
        prompt_text = f"""You are a linguistics expert. Given a dialogue between user and agent, classify the speech acts of user into one or more of the following categories:
{category_list}

Use the following definitions to classify the speech acts:
Label Definition
{category_definitions}

Note that this is a multi-label classification task. Return the possible labels in a comma-separated string in lowercase (e.g., suggest, offer, promise).

User: {example['user1']}"""

        return {
            'prompt': [
                {'role': 'system', 'content': 'You are a helpful assistant that classifies speech acts.'},
                {'role': 'user', 'content': prompt_text}
            ],
            'answer': example[output_col].lower() if example[output_col] else ""
        }

    # Process the dataset
    processed_dataset = dataset.map(create_prompt)

    return processed_dataset, global_categories


def extract_answer(text: str) -> str:
    """Extract the answer from the model's response."""
    text = text.strip().lower()

    # Take the last non-empty line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        return lines[-1]

    return text


def normalize_categories(text: str, valid_categories: List[str]) -> str:
    """Normalize and validate categories in the response."""
    if not text:
        return ""

    predicted_cates = [cate.strip().lower() for cate in text.split(",")]

    # Filter only valid categories
    valid_predicted = [cate for cate in predicted_cates if cate in valid_categories]

    return ", ".join(valid_predicted)