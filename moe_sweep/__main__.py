"""MoE Routing-k Sweep CLI.

Usage:
    python -m moe_sweep generate --experiment my_exp --ks 4,8 --no-vllm
    python -m moe_sweep judge --experiment my_exp
    python -m moe_sweep analyze --experiment my_exp
    python -m moe_sweep run --experiment my_exp --ks 4 --no-vllm
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from dotenv import load_dotenv

from .config import load_sweep_config


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="MoE Routing-k Sweep")
    parser.add_argument("command", choices=["generate", "judge", "analyze", "run"])
    parser.add_argument("--config", default="moe_sweep/sweep_config.yaml", help="Sweep config YAML")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name (scopes output directory)")
    parser.add_argument("--ks", type=str, default=None, help="Comma-separated routing-k values to run (default: all from config)")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs (one vLLM per k)")
    parser.add_argument("--port", type=int, default=None, help="Override vLLM base port")
    parser.add_argument("--mode", choices=["separate", "implicature_first", "quantity_first"], default="separate")
    parser.add_argument("--model-name", type=str, default=None, help="Model name for vLLM API (auto-detected from config if not set)")
    parser.add_argument("--vllm-args", type=str, default=None, help="Extra vLLM args (space-separated)")
    parser.add_argument("-n", type=int, default=None, help="Override n_samples")
    parser.add_argument("--no-vllm", action="store_true", help="Skip vLLM start/stop (assume already running)")

    args = parser.parse_args()
    config = load_sweep_config(args.config)

    # CLI overrides
    if args.experiment is not None:
        config.experiment_name = args.experiment
    if args.n is not None:
        config.n_samples = args.n
    if args.port is not None:
        config.vllm_base_port = args.port

    ks = config.routing_ks
    if args.ks:
        ks = [int(k) for k in args.ks.split(",")]

    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]

    vllm_extra = args.vllm_args.split() if args.vllm_args else None

    if args.command == "generate":
        _run_generate(config, ks, gpu_ids, args, vllm_extra)

    elif args.command == "judge":
        from .judge_batch import judge_all_outputs
        asyncio.run(judge_all_outputs(config))

    elif args.command == "analyze":
        from .analyze import analyze
        analyze(config.experiment_output_dir)

    elif args.command == "run":
        from .judge_batch import judge_all_outputs
        from .analyze import analyze

        _run_generate(config, ks, gpu_ids, args, vllm_extra)

        print(f"\n[run] judge")
        asyncio.run(judge_all_outputs(config))

        print(f"\n[run] analyze")
        analyze(config.experiment_output_dir)


def _run_generate(config, ks, gpu_ids, args, vllm_extra):
    """Run generation for all specified k values."""
    from pathlib import Path
    from .runner import run_sweep_for_k

    for i, k in enumerate(ks):
        variant_path = str(Path(config.model_base_path) / "variants" / f"ept{k}")
        port = config.vllm_base_port + i
        model_name = args.model_name or variant_path

        k_gpus = None
        if gpu_ids:
            k_gpus = [gpu_ids[i % len(gpu_ids)]]

        print(f"\n{'='*50}")
        print(f"Running k={k} on port {port} (GPUs: {k_gpus})")
        print(f"Experiment: {config.experiment_name} -> {config.experiment_output_dir}")
        print(f"{'='*50}")

        asyncio.run(run_sweep_for_k(
            config, k, variant_path, port, model_name,
            gpu_ids=k_gpus, mode=args.mode, vllm_extra_args=vllm_extra,
            no_vllm=args.no_vllm,
        ))


if __name__ == "__main__":
    main()
