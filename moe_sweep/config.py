from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SweepConfig:
    experiment_name: str = "default"
    model_base_path: str = ""
    routing_ks: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    temperatures: list[float] = field(default_factory=lambda: [1.0, 1.3, 1.6])
    n_samples: int = 16
    dataset_name: str = "VietMedTeam/proactive-ai-dataset-2000"
    prompt_quantity_path: str = "prompts/prompt_eval_maxims.md"
    prompt_implicature_path: str = "prompts/prompt_eval_implicature_only.md"
    judge_model: str = "gpt-5"
    judge_prompt_path: str = "prompts/prompt_judge.md"
    output_dir: str = "moe_sweep/outputs"
    vllm_base_port: int = 8000
    max_concurrent_requests: int = 32
    test_size: float = 0.1
    seed: int = 42

    @property
    def experiment_output_dir(self) -> str:
        """Output directory scoped to the experiment name."""
        return str(Path(self.output_dir) / self.experiment_name)


def load_sweep_config(path: str = "moe_sweep/sweep_config.yaml") -> SweepConfig:
    p = Path(path)
    if not p.exists():
        return SweepConfig()
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    return SweepConfig(**{k: v for k, v in data.items() if k in SweepConfig.__dataclass_fields__})
