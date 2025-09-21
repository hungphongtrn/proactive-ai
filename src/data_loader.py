from datasets import load_dataset, Dataset
from typing import List, Tuple, Dict
import yaml


class DatasetProcessor:
    def __init__(self, categories_file: str = 'categories.yaml', prompt_file: str = 'prompt_template.md'):
        self.categories_file = categories_file
        self.prompt_file = prompt_file
        self.categories_data = self._load_categories()
        self.prompt_template = self._load_prompt_template()

    def _load_categories(self) -> Dict:
        with open(self.categories_file, 'r') as f:
            return yaml.safe_load(f)

    def _load_prompt_template(self) -> str:
        with open(self.prompt_file, 'r') as f:
            return f.read().strip()

    def _get_categories_for_column(self, output_col: str) -> List[str]:
        if output_col == 'intent':
            return list(self.categories_data['intents'].keys())
        elif output_col == 'emotion':
            return list(self.categories_data['emotions'].keys())
        else:
            raise ValueError(f"Unsupported output_col: {output_col}. Must be 'intent' or 'emotion'")

    def _format_categories_for_prompt(self) -> Tuple[str, str]:
        intent_dict = "\n".join([f"- {k}: {v}" for k, v in self.categories_data['intents'].items()])
        emotion_dict = "\n".join([f"- {k}: {v}" for k, v in self.categories_data['emotions'].items()])
        return intent_dict, emotion_dict

    def load_and_process_dataset(self, output_col: str) -> Tuple[Dataset, List[str]]:
        """Load the dataset and extract unique categories for the specified column."""
        dataset = load_dataset("richelle05/maxims_50_golden_samples")["train"]

        global_categories = self._get_categories_for_column(output_col)
        intent_dict, emotion_dict = self._format_categories_for_prompt()

        print(f"Found {len(global_categories)} unique categories: {global_categories}")

        def create_prompt(example):
            prompt_text = self.prompt_template.format(
                intent_dict=intent_dict,
                emotion_dict=emotion_dict,
                speech=example['user1']
            )

            return {
                'prompt': prompt_text,
                'answer': example[output_col].lower() if example[output_col] else ""
            }

        processed_dataset = dataset.map(create_prompt)
        return processed_dataset, global_categories


def extract_answer(text: str) -> str:
    """Extract the answer from the model's response."""
    text = text.strip().lower()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        return lines[-1]
    return text


def normalize_categories(text: str, valid_categories: List[str]) -> str:
    """Normalize and validate categories in the response."""
    if not text:
        return ""
    predicted_cates = [cate.strip().lower() for cate in text.split(",")]
    valid_predicted = [cate for cate in predicted_cates if cate in valid_categories]
    return ", ".join(valid_predicted)