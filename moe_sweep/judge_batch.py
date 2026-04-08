"""Batch implicature judging as a separate step after generation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from src.eval.judge import judge_implicature, load_judge_prompt

from .config import SweepConfig


async def judge_all_outputs(config: SweepConfig, file_filter: str | None = None) -> None:
    """Judge all implicature predictions across all raw JSONL files.

    Reads raw generation files, judges each implicature prediction,
    and writes results to *_judged.jsonl files with judge scores cached.

    Args:
        config: Sweep configuration.
        file_filter: Optional glob pattern to filter raw files (e.g. "k4_*_raw.jsonl").
    """
    output_dir = Path(config.experiment_output_dir)
    pattern = file_filter or "*_raw.jsonl"
    raw_files = sorted(output_dir.glob(pattern))

    if not raw_files:
        print(f"No raw JSONL files found in {output_dir}")
        return

    judge_prompt = load_judge_prompt(config.judge_prompt_path)
    semaphore = asyncio.Semaphore(20)  # judge concurrency

    for raw_path in raw_files:
        judged_path = raw_path.with_name(raw_path.name.replace("_raw.jsonl", "_judged.jsonl"))

        # Load existing judged results for caching
        cache = {}
        if judged_path.exists():
            with open(judged_path) as f:
                for line in f:
                    rec = json.loads(line)
                    cache[rec["idx"]] = rec

        # Load raw records
        records = []
        with open(raw_path) as f:
            for line in f:
                records.append(json.loads(line))

        # Collect all judge tasks
        tasks = []
        task_keys = []  # (record_idx_in_list, sample_n)

        for ri, rec in enumerate(records):
            idx = rec["idx"]
            cached = cache.get(idx)
            impl_preds = rec.get("implicature_preds", [])
            gt_impl = rec.get("gt_implicature", "")

            if not gt_impl or not gt_impl.strip():
                continue

            for n, pred in enumerate(impl_preds):
                # Check cache
                if cached and "implicature_scores" in cached:
                    scores = cached["implicature_scores"]
                    if n < len(scores) and scores[n] is not None:
                        continue  # already judged

                if not pred or not pred.strip():
                    task_keys.append((ri, n, None))  # will be score 0
                    continue

                tasks.append((ri, n, rec["speech"], pred, gt_impl))

        print(f"Judging {raw_path.name}: {len(tasks)} predictions to judge ({len(records)} samples)")

        # Run judge calls
        results_map: dict[tuple[int, int], int] = {}

        async def _judge(ri: int, n: int, speech: str, pred: str, gt: str):
            try:
                result = await judge_implicature(
                    speech=speech,
                    pred_implicature=pred,
                    gt_implicature=gt,
                    model=config.judge_model,
                    prompt_template=judge_prompt,
                    semaphore=semaphore,
                )
                results_map[(ri, n)] = result["implicature_score"]
            except Exception as e:
                print(f"Judge error (record {ri}, sample {n}): {e}")
                results_map[(ri, n)] = 0

        judge_tasks = [_judge(ri, n, sp, pr, gt) for ri, n, sp, pr, gt in tasks]

        # Handle empty predictions
        for ri, n, _ in task_keys:
            if _ is None:
                results_map[(ri, n)] = 0

        if judge_tasks:
            await tqdm_asyncio.gather(*judge_tasks, desc=f"Judging {raw_path.stem}")

        # Build judged records
        with open(judged_path, "w") as f:
            for ri, rec in enumerate(records):
                impl_preds = rec.get("implicature_preds", [])

                # Merge with cache
                cached = cache.get(rec["idx"])
                if cached and "implicature_scores" in cached:
                    scores = list(cached["implicature_scores"])
                else:
                    scores = [None] * len(impl_preds)

                # Fill in new results
                for n in range(len(impl_preds)):
                    if (ri, n) in results_map:
                        scores[n] = results_map[(ri, n)]
                    elif scores[n] is None:
                        scores[n] = 0  # default for missing

                # Pad if needed
                while len(scores) < len(impl_preds):
                    scores.append(0)

                judged_rec = {**rec, "implicature_scores": scores}
                f.write(json.dumps(judged_rec) + "\n")

        print(f"Saved {judged_path}")
