"""
@file       : transformer.py
@package    : src.models
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Transformer architecture implementation for L1 model.
@details    : This module implements the transformer architecture used in the L1 model,
              including multi-head attention, feed-forward networks, and the overall model structure.
              It is designed to be flexible and extensible for various configurations and tasks.
@version    : 1.0

@license    : MIT License
Copyright (c) 2025 Julius Pleunes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .config import L1Config
from .embeddings import TokenEmbedding, PositionalEmbedding


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.
    
    Args:
        config: L1 configuration object
    """
    
    def __init__(self, config: L1Config):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        
        assert self.n_embd % self.n_heads == 0
        
        # Combined linear projection for queries, keys, and values
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        nn.init.normal_(self.c_attn.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.c_attn.bias)
        nn.init.zeros_(self.c_proj.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_embd)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            past_key_value: Cached key and value tensors
            use_cache: Whether to return key and value for caching
            
        Returns:
            Tuple of (output_tensor, present_key_value)
        """
        batch_size, seq_length, _ = x.shape
        
        # Linear projection to get queries, keys, and values
        qkv = self.c_attn(x)  # (batch_size, seq_length, 3 * n_embd)
        
        # Split into queries, keys, and values
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key-value cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        present_key_value = (k, v) if use_cache else None
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Apply causal mask for autoregressive generation
        causal_mask = self._get_causal_mask(seq_length, x.device)
        attn_scores = attn_scores + causal_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.n_embd
        )
        
        # Final linear projection
        output = self.c_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output, present_key_value
    
    def _get_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive attention."""
        mask = torch.triu(
            torch.full((seq_length, seq_length), float('-inf'), device=device),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, seq_length)


class FeedForward(nn.Module):
    """Position-wise feed-forward network.
    
    Args:
        config: L1 configuration object
    """
    
    def __init__(self, config: L1Config):
        super().__init__()
        self.config = config
        
        self.c_fc = nn.Linear(config.n_embd, config.n_inner, bias=True)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize feed-forward weights."""
        nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.c_fc.bias)
        nn.init.zeros_(self.c_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_length, n_embd)
        """
        x = self.c_fc(x)
        x = F.gelu(x)  # Using GELU activation
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers.
    
    Args:
        config: L1 configuration object
    """
    
    def __init__(self, config: L1Config):
        super().__init__()
        self.config = config
        
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(config)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_embd)
            attention_mask: Attention mask
            past_key_value: Cached key and value tensors
            use_cache: Whether to return key and value for caching
            
        Returns:
            Tuple of (output_tensor, present_key_value)
        """
        # Pre-norm attention
        attn_output, present_key_value = self.attn(
            self.ln_1(x), 
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        # Residual connection
        x = x + attn_output
        
        # Pre-norm feed-forward
        mlp_output = self.mlp(self.ln_2(x))
        
        # Residual connection
        x = x + mlp_output
        
        return x, present_key_value


class L1Model(nn.Module):
    """L1 Transformer Language Model.
    
    Args:
        config: L1 configuration object
    """
    
    def __init__(self, config: L1Config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = TokenEmbedding(
            config.vocab_size, 
            config.n_embd,
            padding_idx=config.pad_token_id
        )
        self.position_embedding = PositionalEmbedding(
            config.max_seq_length, 
            config.n_embd
        )
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
        # Tie embeddings and language modeling head weights
        if hasattr(self, 'tie_weights'):
            self.tie_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
    
    def tie_weights(self):
        """Tie embedding and language modeling head weights."""
        self.lm_head.weight = self.token_embedding.embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass of L1 model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            past_key_values: Cached key-value pairs from previous forward passes
            use_cache: Whether to return key-value pairs for caching
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing logits and optionally past_key_values
        """
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(input_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Process attention mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_length, seq_length
            ).to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Apply transformer blocks
        present_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present_key_value = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                'logits': logits,
                'past_key_values': present_key_values if use_cache else None,
                'hidden_states': hidden_states
            }
        else:
            return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Starting token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token IDs
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        self.eval()
        device = input_ids.device
        batch_size, seq_length = input_ids.shape
        
        # Initialize output with input
        generated = input_ids.clone()
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    generated[:, -1:] if past_key_values is not None else generated,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs['logits'][:, -1, :]  # Get last token logits
                past_key_values = outputs['past_key_values']
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for end of sequence
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated
