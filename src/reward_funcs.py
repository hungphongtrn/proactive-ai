from typing import List
from .data_loader import extract_answer

# Global variable to store categories - will be set from main training script
categories = []


def multi_label_accuracy_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """
    TODO: Implement proper multi-label accuracy reward function.
    This is a placeholder that should be replaced with a proper implementation.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]

    # Simple placeholder: reward if any category matches
    rewards = []
    for extracted, true_answer in zip(extracted_responses, answer):
        if extracted and true_answer:
            extracted_cats = set(cat.strip().lower() for cat in extracted.split(","))
            true_cats = set(cat.strip().lower() for cat in true_answer.split(","))

            # Jaccard similarity as reward
            intersection = len(extracted_cats.intersection(true_cats))
            union = len(extracted_cats.union(true_cats))
            reward = intersection / union if union > 0 else 0.0
            rewards.append(reward * 2.0)  # Scale to 0-2 range
        else:
            rewards.append(0.0)

    print(f"Sample - True: {answer[0]}, Predicted: {extracted_responses[0]}, Reward: {rewards[0]}")
    return rewards


def format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the response is in comma-separated format."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]

    rewards = []
    for response in extracted_responses:
        if not response:
            rewards.append(0.0)
            continue

        # Check if it's comma-separated and lowercase
        is_comma_separated = "," in response or len(response.split()) == 1
        is_lowercase = response.islower()

        reward = 0.0
        if is_comma_separated:
            reward += 0.3
        if is_lowercase:
            reward += 0.2

        rewards.append(reward)

    return rewards


def category_validity_reward(completions, **kwargs) -> List[float]:
    """Reward function that checks if predicted categories are valid."""
    global categories
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]

    rewards = []
    for response in extracted_responses:
        if not response:
            rewards.append(0.0)
            continue

        predicted_cats = [cat.strip().lower() for cat in response.split(",")]
        valid_count = sum(1 for cat in predicted_cats if cat in categories)
        total_count = len(predicted_cats)

        if total_count > 0:
            validity_ratio = valid_count / total_count
            rewards.append(validity_ratio * 0.5)  # Scale to 0-0.5 range
        else:
            rewards.append(0.0)

    return rewards


def set_categories(global_categories: List[str]):
    """Set the global categories list for use in reward functions."""
    global categories
    categories = global_categories