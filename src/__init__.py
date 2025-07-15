"""
L1 - Large Language Model Package

A PyTorch-based implementation of a transformer language model with
comprehensive training and inference capabilities.
"""

__version__ = "0.1.0"
__author__ = "L1 Development Team"
__email__ = "l1@example.com"

from .models import L1Model, L1Config
from .training import Trainer, TrainingConfig
from .inference import InferenceEngine
from .data import TextDataset, Tokenizer

__all__ = [
    "L1Model",
    "L1Config", 
    "Trainer",
    "TrainingConfig",
    "InferenceEngine",
    "TextDataset",
    "Tokenizer",
]
