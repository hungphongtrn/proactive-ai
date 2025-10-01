from typing import List, Dict
from loguru import logger
import re
import wandb

# Global variables to store categories - will be set from main training script
intent_categories = []
emotion_categories = []


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
                score += 0.33  # Each tag worth 0.25, total = 1.0

        rewards.append(score)

    logger.info(f"Format Reward: {rewards}")
    return rewards


def hamming_loss_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Reward function based on Hamming Loss for both intent and emotion.
    Returns combined reward for both predictions.
    """
    global intent_categories, emotion_categories
    responses = _get_responses(completions)

    rewards = []
    for response, true_answer in zip(responses, answer):
        parsed = parse_structured_response(response)

        if not parsed["intent"] or not parsed["emotion"]:
            rewards.append(0.0)
            continue

        # Parse true answer (expecting format: "intent1,intent2|emotion1,emotion2")
        try:
            true_intent_str, true_emotion_str = true_answer.split("|")
        except ValueError:
            rewards.append(0.0)
            continue

        # Calculate reward for intent
        predicted_intents = set(
            cat.strip().lower() for cat in parsed["intent"].split(",")
        )
        true_intents = set(cat.strip().lower() for cat in true_intent_str.split(","))

        total_intent_labels = len(intent_categories)
        fp_intent = len(predicted_intents - true_intents)
        fn_intent = len(true_intents - predicted_intents)
        hamming_loss_intent = (
            (fp_intent + fn_intent) / total_intent_labels
            if total_intent_labels > 0
            else 1.0
        )
        reward_intent = 1.0 - hamming_loss_intent

        # Calculate reward for emotion
        predicted_emotions = set(
            cat.strip().lower() for cat in parsed["emotion"].split(",")
        )
        true_emotions = set(cat.strip().lower() for cat in true_emotion_str.split(","))

        total_emotion_labels = len(emotion_categories)
        fp_emotion = len(predicted_emotions - true_emotions)
        fn_emotion = len(true_emotions - predicted_emotions)
        hamming_loss_emotion = (
            (fp_emotion + fn_emotion) / total_emotion_labels
            if total_emotion_labels > 0
            else 1.0
        )
        reward_emotion = 1.0 - hamming_loss_emotion

        # Average of both rewards
        combined_reward = (reward_intent + reward_emotion) / 2.0
        rewards.append(combined_reward)

    if rewards:
        logger.info(f"Hamming Loss Reward: {rewards[0]}")
    return rewards


def f1_score_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Reward function based on F1-score for both intent and emotion.
    Returns combined F1 score for both predictions.
    """
    responses = _get_responses(completions)

    rewards = []
    for response, true_answer in zip(responses, answer):
        parsed = parse_structured_response(response)

        if not parsed["intent"] or not parsed["emotion"]:
            rewards.append(0.0)
            continue

        # Parse true answer
        try:
            true_intent_str, true_emotion_str = true_answer.split("|")
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

        # Average of both F1 scores
        combined_f1 = (f1_intent + f1_emotion) / 2.0
        rewards.append(combined_f1)

    if rewards:
        logger.info
        logger.info(f"Groundtruth: {answer[0]}")
        logger.info(f"Prediction: {responses[0]}")
        logger.info(f"F1 Score Reward: {rewards[0]}")
        if wandb.run:
            wandb.log(
                {
                    "groundtruth": answer[0],
                    "prediction": responses[0],
                }
            )
    return rewards


def accuracy_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Jaccard similarity based reward for both intent and emotion.
    """
    responses = _get_responses(completions)

    rewards = []
    for response, true_answer in zip(responses, answer):
        parsed = parse_structured_response(response)

        if not parsed["intent"] or not parsed["emotion"]:
            rewards.append(0.0)
            continue

        # Parse true answer
        try:
            true_intent_str, true_emotion_str = true_answer.split("|")
        except ValueError:
            rewards.append(0.0)
            continue

        # Calculate Jaccard for intent
        predicted_intents = set(
            cat.strip().lower() for cat in parsed["intent"].split(",")
        )
        true_intents = set(cat.strip().lower() for cat in true_intent_str.split(","))

        intersection_intent = len(predicted_intents.intersection(true_intents))
        union_intent = len(predicted_intents.union(true_intents))
        jaccard_intent = intersection_intent / union_intent if union_intent > 0 else 0.0

        # Calculate Jaccard for emotion
        predicted_emotions = set(
            cat.strip().lower() for cat in parsed["emotion"].split(",")
        )
        true_emotions = set(cat.strip().lower() for cat in true_emotion_str.split(","))

        intersection_emotion = len(predicted_emotions.intersection(true_emotions))
        union_emotion = len(predicted_emotions.union(true_emotions))
        jaccard_emotion = (
            intersection_emotion / union_emotion if union_emotion > 0 else 0.0
        )

        # Average of both Jaccard scores
        combined_jaccard = (jaccard_intent + jaccard_emotion) / 2.0
        rewards.append(combined_jaccard)

    if rewards:
        logger.info(f"Accuracy Reward: {rewards[0]}")

    return rewards


def category_validity_reward(completions, **kwargs) -> List[float]:
    """Reward function that checks if predicted categories are valid for both intent and emotion."""
    global intent_categories, emotion_categories
    responses = _get_responses(completions)

    rewards = []
    for response in responses:
        if not response:
            rewards.append(0.0)
            continue

        parsed = parse_structured_response(response)

        if not parsed["intent"] or not parsed["emotion"]:
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

        # Average validity of both
        combined_validity = (intent_validity + emotion_validity) / 2.0
        rewards.append(combined_validity)
    logger.info(f"Category Validity Reward: {rewards[0]}")
    return rewards


def squared_match_reward(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Reward function where correct matches are squared.
    Getting 2 correct gives 4x the reward of 1 correct (not 2x).
    """
    responses = _get_responses(completions)

    rewards = []
    for response, true_answer in zip(responses, answer):
        parsed = parse_structured_response(response)

        if not parsed["intent"] or not parsed["emotion"]:
            rewards.append(0.0)
            continue

        # Parse true answer
        try:
            true_intent_str, true_emotion_str = true_answer.split("|")
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

        # Average of both rewards
        combined_reward = (reward_intent + reward_emotion) / 2.0
        rewards.append(combined_reward)

    if rewards:
        logger.info(f"Squared Match Reward: {rewards[0]}")

    return rewards


def set_categories(intent_cats: List[str], emotion_cats: List[str]):
    """Set the global categories lists for use in reward functions."""
    global intent_categories, emotion_categories
    intent_categories = [cat.lower() for cat in intent_cats]
    emotion_categories = [cat.lower() for cat in emotion_cats]
