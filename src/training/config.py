"""
Training configuration for L1 model.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for training L1 model.
    
    Args:
        # Training parameters
        num_epochs: Number of training epochs
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        max_grad_norm: Maximum gradient norm for clipping
        
        # Optimizer settings
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
        beta1: Beta1 parameter for Adam optimizers
        beta2: Beta2 parameter for Adam optimizers
        eps: Epsilon parameter for optimizers
        
        # Learning rate scheduling
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
        warmup_steps: Number of warmup steps
        warmup_ratio: Warmup ratio (alternative to warmup_steps)
        
        # Data settings
        max_seq_length: Maximum sequence length for training
        dataloader_num_workers: Number of workers for data loading
        dataloader_pin_memory: Whether to pin memory for data loading
        
        # Training loop settings
        eval_steps: Evaluation frequency (in steps)
        save_steps: Checkpoint saving frequency (in steps)
        logging_steps: Logging frequency (in steps)
        max_steps: Maximum number of training steps (overrides epochs)
        
        # Checkpointing
        output_dir: Directory to save checkpoints and logs
        save_total_limit: Maximum number of checkpoints to keep
        load_best_model_at_end: Whether to load best model at end of training
        
        # Evaluation
        eval_strategy: Evaluation strategy ('no', 'steps', 'epoch')
        metric_for_best_model: Metric to use for best model selection
        greater_is_better: Whether higher metric values are better
        
        # Hardware settings
        device: Device to use for training ('auto', 'cpu', 'cuda')
        fp16: Whether to use mixed precision training
        gradient_checkpointing: Whether to use gradient checkpointing
        
        # Logging and monitoring
        wandb_project: Weights & Biases project name
        wandb_run_name: Weights & Biases run name
        report_to: List of loggers to use
        
        # Reproducibility
        seed: Random seed for reproducibility
    """
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 8
    eval_batch_size: Optional[int] = None
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimizer settings
    optimizer_type: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Learning rate scheduling
    scheduler_type: str = "cosine"
    warmup_steps: int = 500
    warmup_ratio: Optional[float] = None
    
    # Data settings
    max_seq_length: int = 1024
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Training loop settings
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10
    max_steps: Optional[int] = None
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Evaluation
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Hardware settings
    device: str = "auto"
    fp16: bool = False
    gradient_checkpointing: bool = False
    
    # Logging and monitoring
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    report_to: list = None
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
            
        if self.report_to is None:
            self.report_to = ["tensorboard"]
            
        if self.warmup_ratio is not None and self.warmup_steps > 0:
            raise ValueError("Cannot specify both warmup_ratio and warmup_steps")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'eval_batch_size': self.eval_batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
            'optimizer_type': self.optimizer_type,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'scheduler_type': self.scheduler_type,
            'warmup_steps': self.warmup_steps,
            'warmup_ratio': self.warmup_ratio,
            'max_seq_length': self.max_seq_length,
            'dataloader_num_workers': self.dataloader_num_workers,
            'dataloader_pin_memory': self.dataloader_pin_memory,
            'eval_steps': self.eval_steps,
            'save_steps': self.save_steps,
            'logging_steps': self.logging_steps,
            'max_steps': self.max_steps,
            'output_dir': self.output_dir,
            'save_total_limit': self.save_total_limit,
            'load_best_model_at_end': self.load_best_model_at_end,
            'eval_strategy': self.eval_strategy,
            'metric_for_best_model': self.metric_for_best_model,
            'greater_is_better': self.greater_is_better,
            'device': self.device,
            'fp16': self.fp16,
            'gradient_checkpointing': self.gradient_checkpointing,
            'wandb_project': self.wandb_project,
            'wandb_run_name': self.wandb_run_name,
            'report_to': self.report_to,
            'seed': self.seed,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
