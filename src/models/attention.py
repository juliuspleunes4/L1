"""
@file       : attention.py
@package    : src.models
@author     : J.J.G. Pleunes
@date       : 12/2025
@brief      : Advanced attention mechanisms for L1 model.
@details    : Implements Flash Attention 2, Grouped Query Attention (GQA),
              and Rotary Position Embeddings (RoPE) for improved efficiency
              and longer context lengths.
@version    : 2.0

@license    : MIT License
Copyright (c) 2025 Julius Pleunes
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import L1Config

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("⚠️  Flash Attention not available. Install with: pip install flash-attn")


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from Su et al. 2021.
    
    RoPE encodes position information directly into query and key vectors
    using rotation matrices, enabling better extrapolation to longer sequences.
    
    Paper: https://arxiv.org/abs/2104.09864
    
    Args:
        dim: Dimension of the embedding (head_dim)
        max_position_embeddings: Maximum sequence length
        base: Base for the exponential decay (default: 10000)
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: int = 10000,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cos/sin cache
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device)
    
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device):
        """Build the cos/sin cache for efficient computation."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        """
        Args:
            x: Input tensor (not used directly, just for device/dtype inference)
            seq_len: Sequence length to return cos/sin for
            
        Returns:
            Tuple of (cos, sin) tensors for rotary embedding
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)
        
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: Cosine values from rotary embedding
        sin: Sine values from rotary embedding
        position_ids: Optional position IDs for selective positions
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    if position_ids is None:
        # Default: sequential positions
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        cos = cos[position_ids].unsqueeze(1)  # (batch, 1, seq_len, dim)
        sin = sin[position_ids].unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) for efficient inference.
    
    GQA reduces the number of key-value heads while keeping multiple query heads,
    providing a balance between Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
    
    Paper: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
    https://arxiv.org/abs/2305.13245
    
    Args:
        config: L1 configuration object
        use_flash_attention: Whether to use Flash Attention 2 if available
        use_rope: Whether to use Rotary Position Embeddings
    """
    
    def __init__(
        self,
        config: L1Config,
        use_flash_attention: bool = True,
        use_rope: bool = True
    ):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        
        # Grouped Query Attention: fewer KV heads than Q heads
        # Default: 4 KV heads for 12 Q heads (3:1 ratio)
        self.n_kv_heads = getattr(config, 'n_kv_heads', max(1, self.n_heads // 4))
        self.n_rep = self.n_heads // self.n_kv_heads  # Number of Q heads per KV head
        
        assert self.n_embd % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        # Separate projections for Q, K, V with GQA
        self.q_proj = nn.Linear(self.n_embd, self.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.n_embd, bias=config.bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # Rotary embeddings
        self.use_rope = use_rope
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=getattr(config, 'max_seq_length', 4096),
                base=getattr(config, 'rope_theta', 10000)
            )
        
        # Flash Attention support
        self.use_flash_attention = use_flash_attention and FLASH_ATTENTION_AVAILABLE
        if use_flash_attention and not FLASH_ATTENTION_AVAILABLE:
            print("⚠️  Flash Attention requested but not available, falling back to standard attention")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights with scaled initialization."""
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))
        
        if self.config.bias:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.o_proj.bias)
    
    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Repeat key/value tensors to match the number of query heads.
        
        Args:
            hidden_states: (batch, num_kv_heads, seq_len, head_dim)
            
        Returns:
            Repeated tensor of shape (batch, num_heads, seq_len, head_dim)
        """
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if self.n_rep == 1:
            return hidden_states
        
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * self.n_rep, slen, head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with Grouped Query Attention and optional Flash Attention.
        
        Args:
            x: Input tensor (batch_size, seq_length, n_embd)
            attention_mask: Attention mask (batch_size, seq_length) or (batch_size, 1, seq_length, seq_length)
            past_key_value: Cached (key, value) for efficient generation
            use_cache: Whether to return updated cache
            position_ids: Position IDs for RoPE
            
        Returns:
            (output, present_key_value)
        """
        batch_size, seq_length, _ = x.shape
        
        # Project to Q, K, V
        query = self.q_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch_size, seq_length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, seq_length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rotary_emb(value, seq_len=seq_length)
            query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
        
        # Handle KV cache for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        
        present_key_value = (key, value) if use_cache else None
        
        # Repeat K and V for GQA
        key = self._repeat_kv(key)
        value = self._repeat_kv(value)
        
        # Choose attention implementation
        if self.use_flash_attention and self.training:
            # Flash Attention 2 (only in training mode for now)
            attn_output = self._flash_attention(query, key, value, attention_mask)
        else:
            # Standard attention with optimizations
            attn_output = self._standard_attention(query, key, value, attention_mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.n_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present_key_value
    
    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Flash Attention implementation for memory-efficient computation."""
        # Flash Attention expects (batch, seq_len, num_heads, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Flash Attention with causal masking
        attn_output = flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=True,  # Causal mask for autoregressive models
        )
        
        return attn_output.transpose(1, 2)
    
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Standard scaled dot-product attention with causal masking."""
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        seq_length = query.size(2)
        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), float('-inf'), device=query.device),
            diagonal=1
        )
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand (batch, seq_len) -> (batch, 1, 1, seq_len)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output


# Convenience function to create attention layers
def create_attention_layer(config: L1Config, layer_idx: Optional[int] = None) -> nn.Module:
    """Factory function to create the appropriate attention layer.
    
    Args:
        config: L1 configuration object
        layer_idx: Layer index (for potential layer-specific configurations)
        
    Returns:
        Attention module (GQA if configured, otherwise standard MHA)
    """
    use_gqa = getattr(config, 'use_gqa', False)
    use_flash_attention = getattr(config, 'use_flash_attention', True)
    use_rope = getattr(config, 'use_rope', True)
    
    if use_gqa:
        return GroupedQueryAttention(
            config,
            use_flash_attention=use_flash_attention,
            use_rope=use_rope
        )
    else:
        # Fallback to standard MultiHeadAttention (would need to be updated with RoPE)
        from .transformer import MultiHeadAttention
        return MultiHeadAttention(config)
