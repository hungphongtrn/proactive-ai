from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from .providers import ProviderResult, create_generate_fn
from .config import EvalConfig
from .judge import judge_implicature, load_judge_prompt
from .metrics import calculate_wer, compare_all
from .parsing import get_groundtruth, parse_all_outputs


def build_output_path(config: EvalConfig) -> str:
    """Generate output CSV path from config.

    Structure: {output_dir}/{model}_{DDMM}/{tag}.csv
    """
    if config.output.custom_name:
        return str(Path(config.output.dir) / config.output.custom_name)

    date_str = datetime.now().strftime("%d%m")
    backend = config.model.backend
    if backend == "cascade":
        model_part = f"cascade-{config.model.cascade.stt_model}-{config.model.cascade.llm_model}"
    elif backend == "gemini":
        model_part = config.model.gemini_model
    else:
        model_part = config.model.audio_model if config.input.type == "voice" else config.model.name

    # Sanitize model name for filesystem (e.g. "arcee-ai/trinity-large-preview:free" -> "arcee-ai_trinity-large-preview_free")
    safe_model = model_part.replace("/", "_").replace(":", "_")
    folder = f"{safe_model}_{date_str}"
    name = f"{config.output.tag}.csv"
    return str(Path(config.output.dir) / folder / name)


def load_prompt_template(config: EvalConfig) -> str:
    """Load and fill the prompt template with category definitions."""
    template = Path(config.prompt.template_path).read_text()

    categories = yaml.safe_load(Path(config.prompt.categories_path).read_text())
    intent_definitions = "\n".join(
        f"- {k}: {v}" for k, v in categories["intents"].items()
    )
    emotion_definitions = "\n".join(
        f"- {k}: {v}" for k, v in categories["emotions"].items()
    )

    # Fill category placeholders (speech placeholder filled per-sample)
    return template.replace("{intent_definitions}", intent_definitions).replace(
        "{emotion_definitions}", emotion_definitions
    )


def fill_speech(template: str, speech: str) -> str:
    """Fill the {speech} placeholder in the prompt template."""
    return template.replace("{speech}", speech)


def print_summary(df: pd.DataFrame, num_skipped: int = 0) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Processed: {len(df)}, Skipped: {num_skipped}")

    for field_name in ["speech_act", "intent", "emotion"]:
        f1_col = f"{field_name}_f1"
        if f1_col in df.columns:
            print(f"\n{field_name.upper()}:")
            print(f"  Precision: {df[f'{field_name}_precision'].mean():.4f}")
            print(f"  Recall: {df[f'{field_name}_recall'].mean():.4f}")
            print(f"  F1: {df[f1_col].mean():.4f}")

    print("\nMAXIMS:")
    for maxim in ["quality", "quantity", "relevance", "manner"]:
        col = f"compare_maxim_{maxim}"
        if col in df.columns:
            acc = df[col].mean()
            print(f"  {maxim}: {acc:.4f} ({int(df[col].sum())}/{len(df)})")

    if "implicature_score" in df.columns:
        valid = df["implicature_score"].dropna()
        if len(valid) > 0:
            print(f"\nIMPLICATURE (LLM Judge):")
            print(f"  Score: {int(valid.sum())}/{len(valid)} ({valid.mean():.2%})")

    if "wer" in df.columns:
        print(f"\nWER: {df['wer'].mean():.4f}")


async def run_eval(config: EvalConfig, repair_path: str | None = None) -> pd.DataFrame:
    """Run the full evaluation pipeline.

    If repair_path is given, only retry the skipped samples from that run
    and merge results back into the existing CSV.
    """
    # Load dataset
    dataset = load_dataset(config.dataset.name)["train"]
    if config.dataset.split == "all":
        test_dataset = dataset
        print(f"Loaded all {len(test_dataset)} records")
    else:
        split = dataset.train_test_split(test_size=config.dataset.test_size, seed=config.dataset.seed)
        test_dataset = split["test"]
        print(f"Loaded {len(test_dataset)} test records")

    if config.dataset.test_num is not None and config.dataset.test_num < len(test_dataset):
        test_dataset = test_dataset.select(range(config.dataset.test_num))
        print(f"Selected first {config.dataset.test_num} records")

    # Repair mode: determine which indices to retry
    repair_indices = None
    if repair_path:
        skipped_path = repair_path.replace(".csv", "_skipped.csv")
        if not Path(skipped_path).exists():
            print(f"No skipped file found at {skipped_path}, nothing to repair.")
            return pd.DataFrame()
        skipped_df = pd.read_csv(skipped_path)
        repair_indices = set(skipped_df["idx"].tolist())
        print(f"Repair mode: retrying {len(repair_indices)} skipped samples from {repair_path}")

    # Load prompt template with categories filled
    prompt_template = load_prompt_template(config)
    print(f"Loaded prompt from {config.prompt.template_path}")

    # Load judge prompt if enabled
    judge_prompt = None
    if config.judge.enabled:
        judge_prompt = load_judge_prompt(config.judge.prompt_path)
        print(f"Judge enabled (model: {config.judge.model})")

    # Create backend and semaphores
    generate_fn = create_generate_fn(config)
    semaphore = asyncio.Semaphore(config.concurrency.max_workers)
    judge_semaphore = asyncio.Semaphore(config.concurrency.judge_workers)

    is_cascade = config.model.backend == "cascade"

    async def process_sample(idx: int, sample: dict) -> dict:
        speech = sample["user1"]
        prompt = fill_speech(prompt_template, speech)

        # Generate
        try:
            result: ProviderResult = await generate_fn(prompt, sample, semaphore)
        except Exception as e:
            print(f"ERROR on sample {idx}: {e}")
            return {"__skipped__": True, "idx": idx, "reason": str(e)}

        if not result.generation:
            return {"__skipped__": True, "idx": idx, "reason": "empty_generation"}

        # Parse + metrics
        parsed = parse_all_outputs(result.generation)
        groundtruth = get_groundtruth(sample)
        comparison = compare_all(parsed, groundtruth)

        record = {
            "original_index": idx,
            "speech": speech,
            "output": result.generation,
            **parsed,
            **groundtruth,
            **comparison,
        }

        # Cascade WER
        if is_cascade and result.transcription is not None:
            record["transcription"] = result.transcription
            record["groundtruth_text"] = speech
            record["wer"] = calculate_wer(speech, result.transcription)

        # LLM judge for implicature
        if (
            config.judge.enabled
            and judge_prompt
            and groundtruth["gt_implicature"]
            and groundtruth["gt_implicature"].strip()
        ):
            try:
                judge_result = await judge_implicature(
                    speech=speech,
                    pred_implicature=parsed["pred_implicature"],
                    gt_implicature=groundtruth["gt_implicature"],
                    model=config.judge.model,
                    prompt_template=judge_prompt,
                    semaphore=judge_semaphore,
                )
                record.update(judge_result)
            except Exception as e:
                print(f"Judge error on sample {idx}: {e}")

        return record

    # Build task list — filter to repair indices if in repair mode
    if repair_indices is not None:
        tasks = [
            process_sample(idx, test_dataset[idx])
            for idx in sorted(repair_indices)
            if idx < len(test_dataset)
        ]
    else:
        tasks = [process_sample(idx, sample) for idx, sample in enumerate(test_dataset)]

    all_results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")

    # Separate results and skipped
    results = []
    skipped = []
    for r in all_results:
        if r.get("__skipped__"):
            skipped.append(r)
        else:
            results.append(r)

    # Save / merge
    if repair_path:
        output_path = repair_path
        skipped_path = repair_path.replace(".csv", "_skipped.csv")

        # Merge with existing results
        existing_df = pd.read_csv(output_path)
        new_df = pd.DataFrame(results)
        if not new_df.empty:
            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
            merged_df = merged_df.sort_values("original_index").reset_index(drop=True)
        else:
            merged_df = existing_df

        merged_df.to_csv(output_path, index=False)

        recovered = len(results)
        still_failed = len(skipped)
        print(f"\nRepair complete: {recovered} recovered, {still_failed} still failed")
        print(f"Total rows now: {len(merged_df)} (was {len(existing_df)})")
        print(f"Updated {output_path}")

        if skipped:
            pd.DataFrame(skipped).to_csv(skipped_path, index=False)
            print(f"Updated skipped: {skipped_path}")
        elif Path(skipped_path).exists():
            Path(skipped_path).unlink()
            print(f"All samples recovered! Removed {skipped_path}")

        print_summary(merged_df, num_skipped=still_failed)
        return merged_df
    else:
        output_path = build_output_path(config)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

        if skipped:
            skipped_path = output_path.replace(".csv", "_skipped.csv")
            pd.DataFrame(skipped).to_csv(skipped_path, index=False)
            print(f"Skipped records saved to {skipped_path}")

        print_summary(results_df, num_skipped=len(skipped))
        return results_df
