def compare_values(pred: str, gt: str) -> int:
    """Compare prediction and groundtruth (case-insensitive exact match)."""
    return 1 if pred.strip().lower() == gt.strip().lower() else 0


def calculate_f1_multilabel(pred: str, gt: str) -> dict:
    """Calculate precision, recall, F1 for multi-label (comma-separated) values."""
    pred_set = set(p.strip().lower() for p in pred.split(",") if p.strip())
    gt_set = set(g.strip().lower() for g in gt.split(",") if g.strip())

    if not pred_set and not gt_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not gt_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate using dynamic programming."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compare_all(parsed: dict, groundtruth: dict) -> dict:
    """Compare all predictions with groundtruth using appropriate metrics per column."""
    result = {}

    # Multi-label F1 for speech_act, intent, emotion
    for field in ["speech_act", "intent", "emotion"]:
        metrics = calculate_f1_multilabel(parsed[f"pred_{field}"], groundtruth[f"gt_{field}"])
        result[f"{field}_precision"] = metrics["precision"]
        result[f"{field}_recall"] = metrics["recall"]
        result[f"{field}_f1"] = metrics["f1"]

    # Binary accuracy for maxims
    for maxim in ["quality", "quantity", "relevance", "manner"]:
        result[f"compare_maxim_{maxim}"] = compare_values(
            parsed[f"pred_maxim_{maxim}"], groundtruth[f"gt_maxim_{maxim}"]
        )

    return result
