import pandas as pd
from datasets import Dataset
from huggingface_hub import login

# Login to HuggingFace (run: huggingface-cli login first, or use token)
# login(token="your_token")

# Read both sheets
dialogue_df = pd.read_excel("merged_output.xlsx", sheet_name="Dialogue")
annotation_df = pd.read_excel("merged_output.xlsx", sheet_name="Annotation")

# Merge on dialog_id
merged_df = pd.merge(dialogue_df, annotation_df, on="dialog_id", how="inner")

# Create HuggingFace dataset
dataset = Dataset.from_pandas(merged_df, preserve_index=False)

# Push to hub
dataset.push_to_hub("VietMedTeam/proactive-ai-dataset-2000")

print(f"Pushed {len(dataset)} rows")