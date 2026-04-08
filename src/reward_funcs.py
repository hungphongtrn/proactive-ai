from typing import List, Dict
from loguru import logger
import re
import wandb

# Global variables to store categories - will be set from main training script
intent_categories = []
emotion_categories = []
max_new_tokens = 1024

def set_global_params(intent_cats: List[str], emotion_cats: List[str], max_tokens: int = 1024):
    """Set the global categories lists for use in reward functions."""
    global intent_categories, emotion_categories, max_new_tokens
    intent_categories = [cat.lower() for cat in intent_cats]
    emotion_categories = [cat.lower() for cat in emotion_cats]
    max_new_tokens = max_tokens

def _get_responses(completions):
    """Helper function to extract responses from completions."""
    if isinstance(completions[0], str):
        return completions
    else:
        return [completion[0]["content"] for completion in completions]


def parse_structured_response(text: str) -> Dict[str, str]:
    """Parse the structured response with reasoning, intent, emotion, and response tags."""

    # Only extract the result after </think>
    if "</think>" in text:
        text = text.split("</think>")[-1]

    result = {"intent": "", "emotion": "", "response": ""}

    # Define patterns for each tag
    patterns = {
        "intent": r"<intent>\s*(.*?)\s*</intent>",
        "emotion": r"<emotion>\s*(.*?)\s*</emotion>",
        "response": r"<response>\s*(.*?)\s*</response>",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    return result


def format_structure_reward(completions, **kwargs) -> List[float]:
    """Reward function that checks if the response follows the expected XML structure."""
    responses = _get_responses(completions)

    rewards = []
    for response in responses:
        if not response:
            rewards.append(0.0)
            continue

        parsed = parse_structured_response(response)

        # Check if all required tags are present and non-empty
        required_tags = ["intent", "emotion", "response"]
        score = 0.0

        for tag in required_tags:
            if parsed[tag]:
                score += 0.33  # Each tag worth 0.33, total ~= 1.0

        rewards.append(score)

    logger.info(f"Format Reward: {rewards}")
    return rewards


def f1_score_intent_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """Reward function based on F1-score for intent only."""
    responses = _get_responses(completions)

    rewards = []
    for response, true_answer in zip(responses, answer):
        parsed = parse_structured_response(response)

        if not parsed["intent"]:
            rewards.append(0.0)
            continue

        # Parse true answer
        try:
            true_intent_str, _ = true_answer.split("|")
        except ValueError:
            rewards.append(0.0)
            continue

        # Calculate F1 for intent
        predicted_intents = set(
            cat.strip().lower() for cat in parsed["intent"].split(",")
        )
        true_intents = set(cat.strip().lower() for cat in true_intent_str.split(","))

        tp_intent = len(predicted_intents.intersection(true_intents))
        precision_intent = (
            tp_intent / len(predicted_intents) if len(predicted_intents) > 0 else 0.0
        )
        recall_intent = tp_intent / len(true_intents) if len(true_intents) > 0 else 0.0

        if precision_intent + recall_intent > 0:
            f1_intent = (
                2
                * (precision_intent * recall_intent)
                / (precision_intent + recall_intent)
            )
        else:
            f1_intent = 0.0

        rewards.append(f1_intent)

    if rewards:
        logger.info(f"F1 Score Intent Reward: {rewards[0]}")
        if wandb.run:
            wandb.log({"f1_intent": rewards[0]})

    return rewards


def f1_score_emotion_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """Reward function based on F1-score for emotion only."""
    responses = _get_responses(completions)

    rewards = []
    for response, true_answer in zip(responses, answer):
        parsed = parse_structured_response(response)

        if not parsed["emotion"]:
            rewards.append(0.0)
            continue

        # Parse true answer
        try:
            _, true_emotion_str = true_answer.split("|")
        except ValueError:
            rewards.append(0.0)
            continue

        # Calculate F1 for emotion
        predicted_emotions = set(
            cat.strip().lower() for cat in parsed["emotion"].split(",")
        )
        true_emotions = set(cat.strip().lower() for cat in true_emotion_str.split(","))

        tp_emotion = len(predicted_emotions.intersection(true_emotions))
        precision_emotion = (
            tp_emotion / len(predicted_emotions) if len(predicted_emotions) > 0 else 0.0
        )
        recall_emotion = (
            tp_emotion / len(true_emotions) if len(true_emotions) > 0 else 0.0
        )

        if precision_emotion + recall_emotion > 0:
            f1_emotion = (
                2
                * (precision_emotion * recall_emotion)
                / (precision_emotion + recall_emotion)
            )
        else:
            f1_emotion = 0.0

        rewards.append(f1_emotion)

    if rewards:
        logger.info(f"F1 Score Emotion Reward: {rewards[0]}")
        logger.info(f"Groundtruth: {answer[0]}")
        logger.info(f"Prediction: {responses[0]}")
        if wandb.run:
            wandb.log({
                "f1_emotion": rewards[0],
                "groundtruth": answer[0],
                "prediction": responses[0],
            })

    return rewards


def accuracy_intent_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """Jaccard similarity based reward for intent only."""
    responses = _get_responses(completions)

    rewards = []
    for response, true_answer in zip(responses, answer):
        score = _calculate_jaccard_intent(response, true_answer)
        rewards.append(score)

    if rewards:
        logger.info(f"Accuracy Intent Reward: {rewards[0]}")

    return rewards


def accuracy_emotion_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """Jaccard similarity based reward for emotion only."""
    responses = _get_responses(completions)

    rewards = []
    for response, true_answer in zip(responses, answer):
        score = _calculate_jaccard_emotion(response, true_answer)
        rewards.append(score)

    if rewards:
        logger.info(f"Accuracy Emotion Reward: {rewards[0]}")

    return rewards


def category_validity_intent_reward(completions, **kwargs) -> List[float]:
    """Reward function that checks if predicted intent categories are valid."""
    global intent_categories
    responses = _get_responses(completions)

    rewards = []
    for response in responses:
        if not response:
            rewards.append(0.0)
            continue

        parsed = parse_structured_response(response)

        if not parsed["intent"]:
            rewards.append(0.0)
            continue

        # Check intent validity
        predicted_intents = [cat.strip().lower() for cat in parsed["intent"].split(",")]
        valid_intent_count = sum(
            1 for cat in predicted_intents if cat in intent_categories
        )
        total_intent_count = len(predicted_intents)

        intent_validity = (
            valid_intent_count / total_intent_count if total_intent_count > 0 else 0.0
        )

        rewards.append(intent_validity)

    logger.info(f"Category Validity Intent Reward: {rewards[0]}")
    return rewards


def category_validity_emotion_reward(completions, **kwargs) -> List[float]:
    """Reward function that checks if predicted emotion categories are valid."""
    global emotion_categories
    responses = _get_responses(completions)

    rewards = []
    for response in responses:
        if not response:
            rewards.append(0.0)
            continue

        parsed = parse_structured_response(response)

        if not parsed["emotion"]:
            rewards.append(0.0)
            continue

        # Check emotion validity
        predicted_emotions = [
            cat.strip().lower() for cat in parsed["emotion"].split(",")
        ]
        valid_emotion_count = sum(
            1 for cat in predicted_emotions if cat in emotion_categories
        )
        total_emotion_count = len(predicted_emotions)

        emotion_validity = (
            valid_emotion_count / total_emotion_count
            if total_emotion_count > 0
            else 0.0
        )

        rewards.append(emotion_validity)

    logger.info(f"Category Validity Emotion Reward: {rewards[0]}")
    return rewards


def squared_match_intent_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """Reward function where correct intent matches are squared."""
    responses = _get_responses(completions)

    rewards = []
    for response, true_answer in zip(responses, answer):
        parsed = parse_structured_response(response)

        if not parsed["intent"]:
            rewards.append(0.0)
            continue

        # Parse true answer
        try:
            true_intent_str, _ = true_answer.split("|")
        except ValueError:
            rewards.append(0.0)
            continue

        # Calculate squared match reward for intent
        predicted_intents = set(
            cat.strip().lower() for cat in parsed["intent"].split(",")
        )
        true_intents = set(cat.strip().lower() for cat in true_intent_str.split(","))

        correct_intent = len(predicted_intents.intersection(true_intents))
        total_intent = len(true_intents)
        reward_intent = (
            (correct_intent**2) / (total_intent**2) if total_intent > 0 else 0.0
        )

        rewards.append(reward_intent)

    if rewards:
        logger.info(f"Squared Match Intent Reward: {rewards[0]}")

    return rewards


def squared_match_emotion_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """Reward function where correct emotion matches are squared."""
    responses = _get_responses(completions)

    rewards = []
    for response, true_answer in zip(responses, answer):
        parsed = parse_structured_response(response)

        if not parsed["emotion"]:
            rewards.append(0.0)
            continue

        # Parse true answer
        try:
            _, true_emotion_str = true_answer.split("|")
        except ValueError:
            rewards.append(0.0)
            continue

        # Calculate squared match reward for emotion
        predicted_emotions = set(
            cat.strip().lower() for cat in parsed["emotion"].split(",")
        )
        true_emotions = set(cat.strip().lower() for cat in true_emotion_str.split(","))

        correct_emotion = len(predicted_emotions.intersection(true_emotions))
        total_emotion = len(true_emotions)
        reward_emotion = (
            (correct_emotion**2) / (total_emotion**2) if total_emotion > 0 else 0.0
        )

        rewards.append(reward_emotion)

    if rewards:
        logger.info(f"Squared Match Emotion Reward: {rewards[0]}")

    return rewards


def _calculate_jaccard_intent(response: str, true_answer: str) -> float:
    """Helper function to calculate Jaccard score for intent only."""
    parsed = parse_structured_response(response)

    if not parsed["intent"]:
        return 0.0

    try:
        true_intent_str, _ = true_answer.split("|")
    except ValueError:
        return 0.0

    # Calculate Jaccard for intent
    predicted_intents = set(cat.strip().lower() for cat in parsed["intent"].split(","))
    true_intents = set(cat.strip().lower() for cat in true_intent_str.split(","))

    intersection_intent = len(predicted_intents.intersection(true_intents))
    union_intent = len(predicted_intents.union(true_intents))
    jaccard_intent = intersection_intent / union_intent if union_intent > 0 else 0.0

    return jaccard_intent


def _calculate_jaccard_emotion(response: str, true_answer: str) -> float:
    """Helper function to calculate Jaccard score for emotion only."""
    parsed = parse_structured_response(response)

    if not parsed["emotion"]:
        return 0.0

    try:
        _, true_emotion_str = true_answer.split("|")
    except ValueError:
        return 0.0

    # Calculate Jaccard for emotion
    predicted_emotions = set(cat.strip().lower() for cat in parsed["emotion"].split(","))
    true_emotions = set(cat.strip().lower() for cat in true_emotion_str.split(","))

    intersection_emotion = len(predicted_emotions.intersection(true_emotions))
    union_emotion = len(predicted_emotions.union(true_emotions))
    jaccard_emotion = intersection_emotion / union_emotion if union_emotion > 0 else 0.0

    return jaccard_emotion


def thinking_efficiency_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Reward: short thinking when correct, long thinking when incorrect.
    Uses both intent and emotion accuracy.
    """
    responses = _get_responses(completions)

    rewards = []
    for response, true_answer in zip(responses, answer):
        # Extract thinking length
        thinking_length = 0
        if "<think>" in response and "</think>" in response:
            thinking_text = response.split("<think>")[1].split("</think>")[0]
            thinking_length = len(thinking_text.split())

        thinking_ratio = min(thinking_length / max_new_tokens, 1.0)

        # Calculate accuracy using both intent and emotion Jaccard
        intent_accuracy = _calculate_jaccard_intent(response, true_answer)
        emotion_accuracy = _calculate_jaccard_emotion(response, true_answer)
        accuracy = (intent_accuracy + emotion_accuracy) / 2.0

        # High accuracy + short thinking = high reward
        # Low accuracy + long thinking = high reward
        reward = accuracy * (1 - thinking_ratio) + (1 - accuracy) * thinking_ratio

        rewards.append(reward)

    if rewards:
        logger.info(f"Thinking Efficiency Reward: {rewards[0]}")

    return rewards

def set_global_params(intent_cats: List[str], emotion_cats: List[str], max_tokens: int = 1024):
    """Set the global categories lists for use in reward functions."""
    global intent_categories, emotion_categories, max_new_tokens
    intent_categories = [cat.lower() for cat in intent_cats]
    emotion_categories = [cat.lower() for cat in emotion_cats]
    max_new_tokens = max_tokens

