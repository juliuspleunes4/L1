"""
Embedding layers for L1 transformer model.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class TokenEmbedding(nn.Module):
    """Token embedding layer with optional weight tying.
    
    Args:
        vocab_size: Size of the vocabulary
        embed_dim: Embedding dimension
        padding_idx: Index of padding token (optional)
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            vocab_size, 
            embed_dim, 
            padding_idx=padding_idx
        )
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            
        Returns:
            Token embeddings of shape (batch_size, seq_length, embed_dim)
        """
        return self.embedding(input_ids)


class PositionalEmbedding(nn.Module):
    """Learnable positional embedding layer.
    
    Args:
        max_seq_length: Maximum sequence length
        embed_dim: Embedding dimension
    """
    
    def __init__(self, max_seq_length: int, embed_dim: int):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(max_seq_length, embed_dim)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize positional embeddings."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            
        Returns:
            Positional embeddings of shape (batch_size, seq_length, embed_dim)
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(
            seq_length, 
            dtype=torch.long, 
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        return self.embedding(position_ids)


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding (fixed, not learnable).
    
    This implementation follows the original Transformer paper.
    
    Args:
        max_seq_length: Maximum sequence length
        embed_dim: Embedding dimension
    """
    
    def __init__(self, max_seq_length: int, embed_dim: int):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        
        # Create sinusoidal embeddings
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            -(math.log(10000.0) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            
        Returns:
            Positional embeddings of shape (batch_size, seq_length, embed_dim)
        """
        seq_length = input_ids.size(1)
        return self.pe[:, :seq_length]
