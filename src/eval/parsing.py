import re


def extract_tag_content(text: str, tag: str) -> str:
    """Extract content from a specific XML tag."""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def parse_all_outputs(output: str) -> dict:
    """Parse all predicted fields from model generation."""
    return {
        "pred_speech_act": extract_tag_content(output, "speech_act"),
        "pred_intent": extract_tag_content(output, "intent"),
        "pred_emotion": extract_tag_content(output, "emotion"),
        "pred_maxim_quality": extract_tag_content(output, "quality"),
        "pred_maxim_quantity": extract_tag_content(output, "quantity"),
        "pred_maxim_relevance": extract_tag_content(output, "relevance"),
        "pred_maxim_manner": extract_tag_content(output, "manner"),
        "pred_implicature": extract_tag_content(output, "implicature"),
    }


def get_groundtruth(sample: dict) -> dict:
    """Get all groundtruth values from a dataset sample."""
    return {
        "gt_speech_act": sample.get("speech_act", ""),
        "gt_intent": sample.get("intent", ""),
        "gt_emotion": sample.get("emotion", ""),
        "gt_maxim_quality": sample.get("maxim_quality", ""),
        "gt_maxim_quantity": sample.get("maxim_quantity", ""),
        "gt_maxim_relevance": sample.get("maxim_relevance", ""),
        "gt_maxim_manner": sample.get("maxim_manner", ""),
        "gt_implicature": sample.get("implicature_text", ""),
    }
