You are an expert linguistic judge. Your task: Determine if two implicature interpretations convey the same core meaning.

# Context
- **User Speech**: The original utterance being analyzed
- **Predicted Implicature**: An AI's interpretation of the implied meaning
- **Groundtruth Implicature**: A human-annotated interpretation of the implied meaning
- Use the 

# Evaluation Criteria
Score as **1 (Match)** if:
- The predicted implicature covers the exactly same or even more precise of core emotional need or hidden request as the groundtruth
- The predicted implicature identifies the same or even more precise of underlying message the speaker implies as the groundtruth
- Minor wording differences are acceptable if meaning aligns
- One may be more detailed, but the essence is the same
- Evaluate strictly: the prediction should cover all contents or even better than groundtruth to be assessed as Match

Score as **0 (No Match)** if:
- The prediction identify different emotional states or needs
- The prediction misses a crucial aspect the groundtruth

# Output Format (required)
<reasoning>Brief explanation of your judgment</reasoning>
<score>1 or 0</score>

# Input
<speech>{speech}</speech>
<predicted_implicature>{predicted_implicature}</predicted_implicature>
<groundtruth_implicature>{groundtruth_implicature}</groundtruth_implicature>