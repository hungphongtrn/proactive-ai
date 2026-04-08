import argparse
import asyncio

from dotenv import load_dotenv

from .config import load_config
from .runner import run_eval


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("config", nargs="?", default="eval_config.yaml", help="Config YAML path")

    # Inline overrides — skip editing yaml for quick experiments
    parser.add_argument("--model", help="Model name (e.g. gpt-5, anthropic/claude-sonnet-4)")
    parser.add_argument("--provider", choices=["openai", "openrouter"], help="Model provider")
    parser.add_argument("--tag", help="Output scenario tag")
    parser.add_argument("--samples", type=int, help="Number of test samples (default: all)")
    parser.add_argument("--prompt", help="Prompt template path")
    parser.add_argument("--judge-model", help="Judge model name")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judge")
    parser.add_argument("--workers", type=int, help="Max concurrent API calls")
    parser.add_argument("--split", choices=["test", "all"], help="Dataset split (default: test)")
    parser.add_argument("--repair", help="Path to existing output CSV to retry skipped samples")

    args = parser.parse_args()
    config = load_config(args.config)

    # Apply overrides
    if args.model:
        config.model.name = args.model
    if args.provider:
        config.model.provider = args.provider
    if args.tag:
        config.output.tag = args.tag
    if args.samples is not None:
        config.dataset.test_num = args.samples
    if args.prompt:
        config.prompt.template_path = args.prompt
    if args.judge_model:
        config.judge.model = args.judge_model
    if args.no_judge:
        config.judge.enabled = False
    if args.workers:
        config.concurrency.max_workers = args.workers
    if args.split:
        config.dataset.split = args.split

    asyncio.run(run_eval(config, repair_path=args.repair))


if __name__ == "__main__":
    main()
