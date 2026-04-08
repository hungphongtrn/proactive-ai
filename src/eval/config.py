from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    name: str = "hungphongtrn/proactive-ai-2000"
    test_size: float = 0.1
    seed: int = 42
    test_num: int | None = None
    split: str = "test"  # "test" | "all"


@dataclass
class VoiceConfig:
    data_dir: str = "voice_data/noisy_out_music/snr_0.5db"
    file_pattern: str = "voice_{dialog_id}.wav"


@dataclass
class InputConfig:
    type: str = "text"  # "text" | "voice"
    voice: VoiceConfig = field(default_factory=VoiceConfig)


@dataclass
class CascadeConfig:
    stt_model: str = "nova-3"
    llm_model: str = "gpt-5"


@dataclass
class ModelConfig:
    provider: str = "openai"  # "openai" | "openrouter"
    backend: str = "openai"  # "openai" | "gemini" | "cascade"
    name: str = "gpt-5"
    audio_model: str = "gpt-4o-audio-preview"
    gemini_model: str = "gemini-2.5-flash"
    cascade: CascadeConfig = field(default_factory=CascadeConfig)


@dataclass
class JudgeConfig:
    enabled: bool = True
    model: str = "gpt-5"
    prompt_path: str = "prompts/prompt_judge.md"


@dataclass
class ConcurrencyConfig:
    max_workers: int = 10
    judge_workers: int = 20


@dataclass
class PromptConfig:
    template_path: str = "prompts/prompt_eval_all.md"
    categories_path: str = "categories.yaml"


@dataclass
class OutputConfig:
    dir: str = "outputs/eval"
    tag: str = "full"
    custom_name: str | None = None


@dataclass
class EvalConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    input: InputConfig = field(default_factory=InputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a dict to a nested dataclass."""
    if not isinstance(data, dict):
        return data
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, val in data.items():
        if key in cls.__dataclass_fields__:
            ft = cls.__dataclass_fields__[key].type
            # Resolve string annotations
            if isinstance(ft, str):
                ft = eval(ft, {k: v for k, v in globals().items()}, locals())
            if isinstance(val, dict) and hasattr(ft, "__dataclass_fields__"):
                kwargs[key] = _dict_to_dataclass(ft, val)
            else:
                kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> EvalConfig:
    """Load evaluation config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _dict_to_dataclass(EvalConfig, raw)
