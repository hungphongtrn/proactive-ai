from datasets import load_dataset, Dataset
from typing import List, Tuple, Dict
import yaml
from loguru import logger


class DatasetProcessor:
    def __init__(self, categories_path: str = 'categories.yaml', prompt_path: str = 'prompt_template.md', tokenizer = None):
        self.categories_file = categories_path
        self.prompt_file = prompt_path
        self.categories_data = self._load_categories()
        self.prompt_template = self._load_prompt_template()
        self.tokenizer = tokenizer

    def _load_categories(self) -> Dict:
        with open(self.categories_file, 'r') as f:
            return yaml.safe_load(f)

    def _load_prompt_template(self) -> str:
        with open(self.prompt_file, 'r') as f:
            return f.read().strip()

    def _format_categories_for_prompt(self) -> Tuple[str, str]:
        intent_dict = "\n".join([f"- {k}: {v}" for k, v in self.categories_data['intents'].items()])
        emotion_dict = "\n".join([f"- {k}: {v}" for k, v in self.categories_data['emotions'].items()])
        return intent_dict, emotion_dict

    def load_and_process_dataset(self, test_size: float = 0.2) -> Tuple[Dataset, Dataset, List[str], List[str]]:
        """Load the dataset and process for multi-output training with train/test split."""
        dataset = load_dataset("hungphongtrn/proactive-ai-2000")["train"]

        intent_categories = list(self.categories_data['intents'].keys())
        emotion_categories = list(self.categories_data['emotions'].keys())
        intent_dict, emotion_dict = self._format_categories_for_prompt()

        logger.info(f"Found {len(intent_categories)} intent categories: {intent_categories}")
        logger.info(f"Found {len(emotion_categories)} emotion categories: {emotion_categories}")

        def create_prompt(example):
            if self.tokenizer:
                system_prompt = self.prompt_template.format(
                                        intent_dict=intent_dict,
                                        emotion_dict=emotion_dict,
                                    )
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': example['user1']},
                ]
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = self.prompt_template.format(
                    intent_dict=intent_dict,
                    emotion_dict=emotion_dict,
                ) + f"This is the user speech\n<speech>\n{example['user1']}\n</speech>"


            # Combine intent and emotion as "intent_str|emotion_str"
            intent_str = example['intent'].lower() if example['intent'] else ""
            emotion_str = example['emotion'].lower() if example['emotion'] else ""
            combined_answer = f"{intent_str}|{emotion_str}"

            return {
                'prompt': prompt_text,
                'answer': combined_answer
            }

        processed_dataset = dataset.map(create_prompt)

        # Split dataset with seed 42
        split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=42)
        train_dataset = split_dataset['train']
        test_dataset = split_dataset['test']

        logger.info(f"Dataset split - Train: {len(train_dataset)}, Test: {len(test_dataset)}")

        return train_dataset, test_dataset, intent_categories, emotion_categories


def normalize_categories(intent_text: str, emotion_text: str,
                         valid_intent_categories: List[str],
                         valid_emotion_categories: List[str]) -> str:
    """Normalize and validate categories in both intent and emotion responses."""

    # Normalize intent
    if intent_text:
        predicted_intents = [cate.strip().lower() for cate in intent_text.split(",")]
        valid_intents = [cate for cate in predicted_intents if cate in valid_intent_categories]
        intent_str = ", ".join(valid_intents)
    else:
        intent_str = ""

    # Normalize emotion
    if emotion_text:
        predicted_emotions = [cate.strip().lower() for cate in emotion_text.split(",")]
        valid_emotions = [cate for cate in predicted_emotions if cate in valid_emotion_categories]
        emotion_str = ", ".join(valid_emotions)
    else:
        emotion_str = ""

    return f"{intent_str}|{emotion_str}"