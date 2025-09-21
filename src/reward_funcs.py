from typing import List
from .data_loader import extract_answer

# Global variable to store categories - will be set from main training script
categories = []


def _get_responses(completions):
    """Helper function to extract responses from completions."""
    if isinstance(completions[0], str):
        return completions
    else:
        return [completion[0]['content'] for completion in completions]


def hamming_loss_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Reward function based on Hamming Loss (converted to reward).
    Hamming Loss = (FP + FN) / Total_Labels
    Reward = 1 - Hamming Loss
    """
    global categories
    responses = completions
    extracted_responses = [extract_answer(r) for r in responses]

    rewards = []
    for extracted, true_answer in zip(extracted_responses, answer):
        if not extracted or not true_answer:
            rewards.append(0.0)
            continue

        # Convert to sets of lowercase categories
        predicted_cats = set(cat.strip().lower() for cat in extracted.split(","))
        true_cats = set(cat.strip().lower() for cat in true_answer.split(","))

        # Calculate on all possible categories
        total_labels = len(categories)

        # False positives and false negatives
        fp = len(predicted_cats - true_cats)
        fn = len(true_cats - predicted_cats)

        # Hamming loss
        hamming_loss = (fp + fn) / total_labels if total_labels > 0 else 1.0

        # Convert to reward (1 - loss)
        reward = 1.0 - hamming_loss
        rewards.append(reward)

    return rewards


def f1_score_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Reward function based on F1-score for multi-label classification.
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    responses = completions
    extracted_responses = [extract_answer(r) for r in responses]

    rewards = []
    for extracted, true_answer in zip(extracted_responses, answer):
        if not extracted or not true_answer:
            rewards.append(0.0)
            continue

        # Convert to sets of lowercase categories
        predicted_cats = set(cat.strip().lower() for cat in extracted.split(","))
        true_cats = set(cat.strip().lower() for cat in true_answer.split(","))

        # Calculate precision and recall
        tp = len(predicted_cats.intersection(true_cats))

        precision = tp / len(predicted_cats) if len(predicted_cats) > 0 else 0.0
        recall = tp / len(true_cats) if len(true_cats) > 0 else 0.0

        # F1-score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        rewards.append(f1)

    return rewards


def multi_label_accuracy_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """
    TODO: Implement proper multi-label accuracy reward function.
    This is a placeholder that should be replaced with a proper implementation.
    """
    responses = _get_responses(completions)
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
    responses = _get_responses(completions)
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
    responses = _get_responses(completions)
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