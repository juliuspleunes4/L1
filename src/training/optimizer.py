"""
Optimizer and learning rate scheduler utilities.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LinearLR, 
    ConstantLR,
    SequentialLR
)
from typing import Union, Optional

from .config import TrainingConfig


def get_optimizer(
    model: nn.Module, 
    config: TrainingConfig
) -> torch.optim.Optimizer:
    """
    Get optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Training configuration
        
    Returns:
        Configured optimizer
    """
    # Get model parameters (excluding embeddings if weight decay is applied)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer
    if config.optimizer_type.lower() == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    elif config.optimizer_type.lower() == "adam":
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    elif config.optimizer_type.lower() == "sgd":
        optimizer = SGD(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")
    
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    num_training_steps: int
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Get learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Training configuration
        num_training_steps: Total number of training steps
        
    Returns:
        Configured scheduler or None
    """
    if config.scheduler_type.lower() == "constant":
        return None
    
    # Calculate warmup steps
    if config.warmup_ratio is not None:
        warmup_steps = int(config.warmup_ratio * num_training_steps)
    else:
        warmup_steps = config.warmup_steps
    
    warmup_steps = min(warmup_steps, num_training_steps)
    
    if config.scheduler_type.lower() == "cosine":
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps - warmup_steps,
                eta_min=0.1 * config.learning_rate
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,
                eta_min=0.1 * config.learning_rate
            )
    
    elif config.scheduler_type.lower() == "linear":
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            decay_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_training_steps - warmup_steps
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_training_steps
            )
    
    else:
        raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")
    
    return scheduler


class GradientClipping:
    """Gradient clipping utility."""
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients and return the gradient norm.
        
        Args:
            model: Model with gradients to clip
            
        Returns:
            Gradient norm before clipping
        """
        if self.max_norm <= 0:
            return 0.0
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.max_norm
        )
        
        return grad_norm.item()


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(
        self, 
        patience: int = 7, 
        min_delta: float = 0.0,
        greater_is_better: bool = False
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.greater_is_better = greater_is_better
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should be stopped early.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should be stopped
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.greater_is_better:
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta
