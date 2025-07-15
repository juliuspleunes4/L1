"""
Configuration management for L1 model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class L1Config:
    """Configuration class for L1 transformer model.
    
    Args:
        vocab_size: Size of the vocabulary
        max_seq_length: Maximum sequence length
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        n_embd: Embedding dimension
        n_inner: Inner dimension of feed-forward network
        dropout: Dropout probability
        layer_norm_epsilon: Epsilon for layer normalization
        initializer_range: Range for weight initialization
        use_cache: Whether to use past key values for caching
        pad_token_id: Token ID for padding
        eos_token_id: Token ID for end of sequence
        bos_token_id: Token ID for beginning of sequence
    """
    
    # Model architecture
    vocab_size: int = 50257
    max_seq_length: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    n_inner: Optional[int] = None  # Default: 4 * n_embd
    
    # Regularization
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    
    # Initialization
    initializer_range: float = 0.02
    
    # Generation
    use_cache: bool = True
    
    # Special tokens
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
            
        # Validate configuration
        assert self.n_embd % self.n_heads == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_heads ({self.n_heads})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_length > 0, "max_seq_length must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
    
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.n_embd // self.n_heads
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'n_embd': self.n_embd,
            'n_inner': self.n_inner,
            'dropout': self.dropout,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'initializer_range': self.initializer_range,
            'use_cache': self.use_cache,
            'pad_token_id': self.pad_token_id,
            'eos_token_id': self.eos_token_id,
            'bos_token_id': self.bos_token_id,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'L1Config':
        """Create config from dictionary."""
        return cls(**config_dict)
