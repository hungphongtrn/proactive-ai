"""
GRPO Multi-label Classification Package

A simple implementation for training language models with GRPO
(Group Relative Policy Optimization) for multi-label classification tasks.
"""

__version__ = "0.1.0"

from .data_loader import MultiLabelDataLoader
from .model_setup import ModelSetup
from .reward_funcs import get_reward_functions
from .trainer import MultiLabelGRPOTrainer, prepare_dataset_for_grpo
from .inference import MultiLabelInference

__all__ = [
    "MultiLabelDataLoader",
    "ModelSetup",
    "get_reward_functions",
    "MultiLabelGRPOTrainer",
    "prepare_dataset_for_grpo",
    "MultiLabelInference"
]