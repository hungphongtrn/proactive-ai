"""Compute pass@N, majority@N, and joint_pass@N from judged JSONL files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.eval.metrics import compare_values


def compute_sample_metrics(rec: dict) -> dict:
    """Compute per-sample metrics from a judged record.

    Expected fields:
        quantity_preds: list[str]  — N quantity predictions
        gt_quantity: str           — ground truth
        implicature_scores: list[int]  — N judge scores (1=match, 0=no)
        routing_k, temperature, mode, idx
    """
    qty_preds = rec.get("quantity_preds", [])
    gt_qty = rec.get("gt_quantity", "")
    impl_scores = rec.get("implicature_scores", [])

    n = max(len(qty_preds), len(impl_scores))
    if n == 0:
        return {"idx": rec["idx"], "routing_k": rec["routing_k"],
                "temperature": rec["temperature"], "n": 0,
                "pass_n_quantity": 0, "pass_n_implicature": 0,
                "joint_pass_n": 0, "majority_n_quantity": 0,
                "majority_n_implicature": 0}

    # pass@N: any sample correct
    qty_matches = [compare_values(p, gt_qty) for p in qty_preds]
    pass_n_qty = 1 if any(m == 1 for m in qty_matches) else 0
    pass_n_impl = 1 if any(s == 1 for s in impl_scores) else 0

    # joint_pass@N: any index i where BOTH are correct
    # For separate mode, pair by index
    min_n = min(len(qty_matches), len(impl_scores))
    joint = 1 if any(qty_matches[i] == 1 and impl_scores[i] == 1 for i in range(min_n)) else 0

    # majority@N: most common prediction is correct
    if qty_preds:
        qty_counter = Counter(p.strip().lower() for p in qty_preds if p.strip())
        majority_qty_pred = qty_counter.most_common(1)[0][0] if qty_counter else ""
        majority_n_qty = compare_values(majority_qty_pred, gt_qty)
    else:
        majority_n_qty = 0

    # majority@N for implicature: majority of judge scores are 1
    if impl_scores:
        majority_n_impl = 1 if sum(impl_scores) > len(impl_scores) / 2 else 0
    else:
        majority_n_impl = 0

    return {
        "idx": rec["idx"],
        "routing_k": rec["routing_k"],
        "temperature": rec["temperature"],
        "n": n,
        "pass_n_quantity": pass_n_qty,
        "pass_n_implicature": pass_n_impl,
        "joint_pass_n": joint,
        "majority_n_quantity": majority_n_qty,
        "majority_n_implicature": majority_n_impl,
        "gt_quantity": gt_qty,
        "quantity_pred_distribution": dict(Counter(p.strip().lower() for p in qty_preds if p.strip())),
    }


def compute_sweep_metrics(output_dir: str) -> pd.DataFrame:
    """Compute aggregate metrics across all judged JSONL files.

    Returns DataFrame with one row per (routing_k, temperature) config.
    """
    judged_files = sorted(Path(output_dir).glob("*_judged.jsonl"))
    if not judged_files:
        print(f"No judged files found in {output_dir}")
        return pd.DataFrame()

    all_sample_metrics = []

    for jf in judged_files:
        with open(jf) as f:
            for line in f:
                rec = json.loads(line)
                m = compute_sample_metrics(rec)
                all_sample_metrics.append(m)

    df = pd.DataFrame(all_sample_metrics)

    # Aggregate per (routing_k, temperature)
    agg = df.groupby(["routing_k", "temperature"]).agg(
        n_samples=("idx", "count"),
        n_per_sample=("n", "first"),
        pass_n_quantity=("pass_n_quantity", "mean"),
        pass_n_implicature=("pass_n_implicature", "mean"),
        joint_pass_n=("joint_pass_n", "mean"),
        majority_n_quantity=("majority_n_quantity", "mean"),
        majority_n_implicature=("majority_n_implicature", "mean"),
    ).reset_index()

    return agg


def print_sweep_results(df: pd.DataFrame) -> None:
    """Print sweep results table."""
    if df.empty:
        print("No results to display.")
        return

    print("\n" + "=" * 90)
    print("MOE ROUTING-K SWEEP RESULTS")
    print("=" * 90)
    print(f"{'k':>4} {'temp':>5} {'n':>4} {'samples':>7} | "
          f"{'pass@N_qty':>10} {'pass@N_impl':>11} {'joint':>7} | "
          f"{'maj@N_qty':>9} {'maj@N_impl':>10}")
    print("-" * 90)

    for _, row in df.sort_values(["routing_k", "temperature"]).iterrows():
        print(f"{int(row['routing_k']):>4} {row['temperature']:>5.1f} "
              f"{int(row['n_per_sample']):>4} {int(row['n_samples']):>7} | "
              f"{row['pass_n_quantity']:>10.3f} {row['pass_n_implicature']:>11.3f} "
              f"{row['joint_pass_n']:>7.3f} | "
              f"{row['majority_n_quantity']:>9.3f} {row['majority_n_implicature']:>10.3f}")

    print("=" * 90)
