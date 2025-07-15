"""
Loss functions for L1 model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LanguageModelingLoss(nn.Module):
    """Language modeling loss with optional label smoothing.
    
    Args:
        vocab_size: Size of the vocabulary
        ignore_index: Index to ignore in loss calculation (usually padding token)
        label_smoothing: Label smoothing factor (0.0 = no smoothing)
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
        if label_smoothing > 0.0:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_index,
                label_smoothing=label_smoothing
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute language modeling loss.
        
        Args:
            logits: Model logits of shape (batch_size, seq_length, vocab_size)
            labels: Target labels of shape (batch_size, seq_length)
            
        Returns:
            Cross-entropy loss
        """
        # Shift labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for cross-entropy computation
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Compute loss
        loss = self.criterion(shift_logits, shift_labels)
        
        return loss


class PerplexityMetric:
    """Perplexity metric for language modeling evaluation."""
    
    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index
        self.total_loss = 0.0
        self.total_tokens = 0
    
    def update(self, loss: torch.Tensor, num_tokens: int):
        """Update perplexity with new loss value."""
        self.total_loss += loss.item() * num_tokens
        self.total_tokens += num_tokens
    
    def compute(self) -> float:
        """Compute perplexity."""
        if self.total_tokens == 0:
            return float('inf')
        avg_loss = self.total_loss / self.total_tokens
        return torch.exp(torch.tensor(avg_loss)).item()
    
    def reset(self):
        """Reset metric state."""
        self.total_loss = 0.0
        self.total_tokens = 0
