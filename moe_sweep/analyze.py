"""Analysis and summary of sweep results."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .metrics import compute_sweep_metrics, print_sweep_results


def analyze(output_dir: str) -> None:
    """Run full analysis on sweep outputs."""
    df = compute_sweep_metrics(output_dir)
    if df.empty:
        return

    # Save aggregate results
    results_path = Path(output_dir) / "sweep_results.csv"
    df.to_csv(results_path, index=False)
    print(f"Saved aggregate results to {results_path}")

    # Print table
    print_sweep_results(df)

    # Key comparisons
    print("\n--- KEY COMPARISONS ---")

    # Best config for each metric
    for metric in ["pass_n_quantity", "pass_n_implicature", "joint_pass_n",
                    "majority_n_quantity", "majority_n_implicature"]:
        best = df.loc[df[metric].idxmax()]
        print(f"Best {metric}: k={int(best['routing_k'])} t={best['temperature']:.1f} "
              f"= {best[metric]:.3f}")

    # Training k (4) baseline
    baseline = df[df["routing_k"] == 4]
    if not baseline.empty:
        print("\n--- BASELINE (k=4, training config) ---")
        for _, row in baseline.iterrows():
            print(f"  t={row['temperature']:.1f}: "
                  f"pass@N_qty={row['pass_n_quantity']:.3f} "
                  f"pass@N_impl={row['pass_n_implicature']:.3f} "
                  f"joint={row['joint_pass_n']:.3f}")

    # Quantity prediction distribution analysis
    print("\n--- QUANTITY PREDICTION DISTRIBUTION ---")
    _analyze_quantity_distribution(output_dir)


def _analyze_quantity_distribution(output_dir: str) -> None:
    """Analyze what quantity labels the model predicts across configs."""
    judged_files = sorted(Path(output_dir).glob("*_judged.jsonl"))

    for jf in judged_files:
        all_preds = Counter()
        gt_dist = Counter()
        with open(jf) as f:
            for line in f:
                rec = json.loads(line)
                for p in rec.get("quantity_preds", []):
                    if p.strip():
                        all_preds[p.strip().lower()] += 1
                gt_dist[rec.get("gt_quantity", "").strip().lower()] += 1

        k = jf.stem.split("_")[0]  # e.g. "k4"
        t = jf.stem.split("_")[1]  # e.g. "t1.0"
        total = sum(all_preds.values())
        if total == 0:
            continue
        print(f"\n  {k} {t}:")
        print(f"    GT distribution: {dict(gt_dist)}")
        print(f"    Pred distribution ({total} total):")
        for label, count in all_preds.most_common():
            print(f"      {label}: {count} ({count/total:.1%})")
