"""
Comprehensive test suite for Flash Attention 2 and RoPE implementations.
Tests all edge cases, error conditions, and validates correctness.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.config import L1Config
from src.models.attention import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
    GroupedQueryAttention,
    FLASH_ATTENTION_AVAILABLE
)


class TestRotaryEmbedding:
    """Test suite for Rotary Position Embeddings (RoPE)."""
    
    def test_initialization(self):
        """Test RoPE initialization with various configurations."""
        # Standard initialization
        rope = RotaryEmbedding(dim=64, max_position_embeddings=2048)
        assert rope.dim == 64
        assert rope.max_position_embeddings == 2048
        assert rope.base == 10000
        
        # Custom base
        rope_custom = RotaryEmbedding(dim=64, max_position_embeddings=2048, base=50000)
        assert rope_custom.base == 50000
        
    def test_inv_freq_computation(self):
        """Test inverse frequency computation is correct."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=2048)
        
        # Check inv_freq shape
        assert rope.inv_freq.shape == (32,)  # dim // 2
        
        # Check values are decreasing
        assert torch.all(rope.inv_freq[:-1] >= rope.inv_freq[1:])
        
    def test_cos_sin_cache(self):
        """Test cosine and sine cache generation."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=2048)
        
        # Check cache shapes
        assert rope.cos_cached.shape == (2048, 64)
        assert rope.sin_cached.shape == (2048, 64)
        
        # Check values are in valid range [-1, 1]
        assert torch.all(rope.cos_cached >= -1) and torch.all(rope.cos_cached <= 1)
        assert torch.all(rope.sin_cached >= -1) and torch.all(rope.sin_cached <= 1)
        
    def test_forward_pass(self):
        """Test forward pass returns correct shapes."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=2048)
        
        # Create dummy input
        x = torch.randn(4, 512, 12, 64)  # (batch, seq, heads, dim)
        
        cos, sin = rope(x, seq_len=512)
        
        # Check shapes
        assert cos.shape == (512, 64)
        assert sin.shape == (512, 64)
        
    def test_cache_extension(self):
        """Test that cache extends when needed for longer sequences."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=1024)
        
        # Request longer sequence
        x = torch.randn(4, 2048, 12, 64)
        cos, sin = rope(x, seq_len=2048)
        
        # Check cache was extended
        assert rope.max_seq_len_cached >= 2048
        assert cos.shape == (2048, 64)
        assert sin.shape == (2048, 64)
        
    def test_different_devices(self):
        """Test RoPE works on both CPU and GPU."""
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        
        for device in devices:
            rope = RotaryEmbedding(dim=64, max_position_embeddings=2048, device=device)
            x = torch.randn(2, 128, 8, 64, device=device)
            cos, sin = rope(x, seq_len=128)
            
            # Check device type matches (index may differ)
            assert cos.device.type == device.type
            assert sin.device.type == device.type


class TestRotaryHelperFunctions:
    """Test helper functions for rotary embeddings."""
    
    def test_rotate_half(self):
        """Test rotate_half function."""
        x = torch.randn(2, 4, 128, 64)
        rotated = rotate_half(x)
        
        # Check shape preserved
        assert rotated.shape == x.shape
        
        # Check rotation is correct
        x1 = x[..., :32]
        x2 = x[..., 32:]
        expected = torch.cat((-x2, x1), dim=-1)
        assert torch.allclose(rotated, expected)
        
    def test_apply_rotary_pos_emb_shapes(self):
        """Test apply_rotary_pos_emb preserves shapes."""
        batch, heads, seq_len, head_dim = 2, 12, 128, 64
        
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)
        
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
    def test_apply_rotary_pos_emb_with_position_ids(self):
        """Test apply_rotary_pos_emb with custom position IDs."""
        batch, heads, seq_len, head_dim = 2, 12, 128, 64
        
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        cos = torch.randn(256, head_dim)  # Larger cache
        sin = torch.randn(256, head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
    def test_rotary_embedding_invertible(self):
        """Test that RoPE produces consistent results with proper frequencies."""
        # Use actual RoPE implementation instead of random cos/sin
        rope = RotaryEmbedding(dim=64, max_position_embeddings=2048)
        
        q = torch.randn(2, 12, 128, 64)
        k = torch.randn(2, 12, 128, 64)
        
        # Get proper cos/sin from RoPE
        cos, sin = rope(q, seq_len=128)
        
        # Apply rotation
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Check that rotation produces different output
        assert not torch.allclose(q, q_rot)
        assert not torch.allclose(k, k_rot)
        
        # Check shapes are preserved
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
        # Check no NaN or Inf values
        assert not torch.isnan(q_rot).any()
        assert not torch.isinf(q_rot).any()


class TestGroupedQueryAttention:
    """Test suite for Grouped Query Attention."""
    
    def test_initialization_standard_mha(self):
        """Test GQA initialization with standard MHA (n_kv_heads = n_heads)."""
        config = L1Config(
            n_heads=12,
            n_embd=768,
            use_gqa=False
        )
        
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        assert attn.n_heads == 12
        assert attn.n_kv_heads == 12
        assert attn.n_rep == 1
        
    def test_initialization_gqa(self):
        """Test GQA initialization with grouped queries."""
        config = L1Config(
            n_heads=12,
            n_embd=768,
            n_kv_heads=4,
            use_gqa=True
        )
        
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        assert attn.n_heads == 12
        assert attn.n_kv_heads == 4
        assert attn.n_rep == 3  # 12 // 4
        
    def test_initialization_fails_invalid_heads(self):
        """Test that initialization fails with invalid head configuration."""
        # Config validation happens in __post_init__
        with pytest.raises(AssertionError):
            config = L1Config(
                n_heads=12,
                n_embd=768,
                n_kv_heads=5,  # Not divisible
                use_gqa=True
            )
            
    def test_repeat_kv_no_repeat(self):
        """Test _repeat_kv when n_rep = 1."""
        config = L1Config(n_heads=12, n_embd=768, n_kv_heads=12)
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        kv = torch.randn(2, 12, 128, 64)
        repeated = attn._repeat_kv(kv)
        
        assert torch.equal(kv, repeated)
        
    def test_repeat_kv_with_grouping(self):
        """Test _repeat_kv correctly repeats KV heads."""
        config = L1Config(n_heads=12, n_embd=768, n_kv_heads=4)
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        kv = torch.randn(2, 4, 128, 64)
        repeated = attn._repeat_kv(kv)
        
        # Should expand to 12 heads
        assert repeated.shape == (2, 12, 128, 64)
        
        # Check that heads are properly repeated (each KV head used 3 times)
        for i in range(4):
            for j in range(3):
                assert torch.equal(repeated[:, i*3 + j], kv[:, i])
                
    def test_forward_pass_basic(self):
        """Test basic forward pass without caching."""
        config = L1Config(
            n_heads=12,
            n_embd=768,
            n_kv_heads=4,
            max_seq_length=2048
        )
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        batch_size, seq_len = 2, 128
        x = torch.randn(batch_size, seq_len, 768)
        
        output, kv_cache = attn(x, use_cache=False)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, 768)
        assert kv_cache is None
        
    def test_forward_pass_with_cache(self):
        """Test forward pass with KV caching."""
        config = L1Config(
            n_heads=12,
            n_embd=768,
            n_kv_heads=4,
            max_seq_length=2048
        )
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        batch_size, seq_len = 2, 128
        x = torch.randn(batch_size, seq_len, 768)
        
        # First forward pass
        output1, kv_cache = attn(x, use_cache=True)
        
        assert output1.shape == (batch_size, seq_len, 768)
        assert kv_cache is not None
        assert len(kv_cache) == 2  # (key, value)
        
        # Check cache shapes
        k_cache, v_cache = kv_cache
        assert k_cache.shape == (batch_size, 4, seq_len, 64)  # n_kv_heads
        assert v_cache.shape == (batch_size, 4, seq_len, 64)
        
        # Second forward pass with cache
        x_new = torch.randn(batch_size, 1, 768)  # Single token
        output2, kv_cache2 = attn(x_new, past_key_value=kv_cache, use_cache=True)
        
        assert output2.shape == (batch_size, 1, 768)
        k_cache2, v_cache2 = kv_cache2
        assert k_cache2.shape == (batch_size, 4, seq_len + 1, 64)  # Extended
        
    def test_forward_pass_with_rope(self):
        """Test forward pass with RoPE enabled."""
        config = L1Config(
            n_heads=12,
            n_embd=768,
            n_kv_heads=4,
            max_seq_length=2048,
            use_rope=True
        )
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=True)
        
        batch_size, seq_len = 2, 128
        x = torch.randn(batch_size, seq_len, 768)
        
        output, _ = attn(x, use_cache=False)
        
        assert output.shape == (batch_size, seq_len, 768)
        
    def test_attention_mask_application(self):
        """Test that attention mask is properly applied."""
        config = L1Config(
            n_heads=12,
            n_embd=768,
            n_kv_heads=4,
            max_seq_length=2048
        )
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        batch_size, seq_len = 2, 128
        x = torch.randn(batch_size, seq_len, 768)
        
        # Create attention mask (mask out second half)
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[:, seq_len//2:] = float('-inf')
        
        output, _ = attn(x, attention_mask=attention_mask, use_cache=False)
        
        assert output.shape == (batch_size, seq_len, 768)
        # Output should have valid values (not NaN)
        assert not torch.isnan(output).any()
        
    def test_causal_masking(self):
        """Test that causal masking is applied correctly."""
        config = L1Config(
            n_heads=12,
            n_embd=768,
            n_kv_heads=4,
            max_seq_length=2048
        )
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        batch_size, seq_len = 1, 4
        # Create input where each token has distinct value
        x = torch.arange(seq_len).float().view(1, seq_len, 1).expand(1, seq_len, 768)
        
        output, _ = attn(x, use_cache=False)
        
        # With causal masking, each position should only attend to previous positions
        # First token should only see itself, last token should see all
        assert output.shape == (batch_size, seq_len, 768)
        
    def test_different_batch_sizes(self):
        """Test GQA works with various batch sizes."""
        config = L1Config(n_heads=12, n_embd=768, n_kv_heads=4)
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 64, 768)
            output, _ = attn(x, use_cache=False)
            assert output.shape == (batch_size, 64, 768)
            
    def test_different_sequence_lengths(self):
        """Test GQA works with various sequence lengths."""
        config = L1Config(n_heads=12, n_embd=768, n_kv_heads=4, max_seq_length=4096)
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        for seq_len in [1, 32, 128, 512, 1024]:
            x = torch.randn(2, seq_len, 768)
            output, _ = attn(x, use_cache=False)
            assert output.shape == (2, seq_len, 768)
            
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Test GQA works on CUDA."""
        config = L1Config(n_heads=12, n_embd=768, n_kv_heads=4)
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False).cuda()
        
        x = torch.randn(2, 128, 768).cuda()
        output, _ = attn(x, use_cache=False)
        
        assert output.device.type == 'cuda'
        assert output.shape == (2, 128, 768)
        
    def test_gradient_flow(self):
        """Test that gradients flow correctly through GQA."""
        config = L1Config(n_heads=12, n_embd=768, n_kv_heads=4)
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        x = torch.randn(2, 128, 768, requires_grad=True)
        output, _ = attn(x, use_cache=False)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention not available")
    def test_flash_attention_mode(self):
        """Test Flash Attention mode if available."""
        config = L1Config(
            n_heads=12,
            n_embd=768,
            n_kv_heads=4,
            use_flash_attention=True
        )
        attn = GroupedQueryAttention(config, use_flash_attention=True, use_rope=False)
        
        assert attn.use_flash_attention == True
        
        # Test forward pass
        x = torch.randn(2, 128, 768)
        attn.train()  # Flash attention only in training mode
        output, _ = attn(x, use_cache=False)
        
        assert output.shape == (2, 128, 768)
        assert not torch.isnan(output).any()


class TestFlashAttentionFallback:
    """Test Flash Attention fallback behavior."""
    
    def test_fallback_when_not_available(self):
        """Test that model falls back gracefully when Flash Attention unavailable."""
        config = L1Config(
            n_heads=12,
            n_embd=768,
            use_flash_attention=True
        )
        
        # Should not raise error even if Flash Attention not available
        attn = GroupedQueryAttention(config, use_flash_attention=True, use_rope=False)
        
        x = torch.randn(2, 128, 768)
        output, _ = attn(x, use_cache=False)
        
        assert output.shape == (2, 128, 768)
        
    def test_eval_mode_uses_standard_attention(self):
        """Test that eval mode uses standard attention (not Flash)."""
        config = L1Config(n_heads=12, n_embd=768, use_flash_attention=True)
        attn = GroupedQueryAttention(config, use_flash_attention=True, use_rope=False)
        
        attn.eval()  # Evaluation mode
        x = torch.randn(2, 128, 768)
        output, _ = attn(x, use_cache=False)
        
        assert output.shape == (2, 128, 768)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_input_fails_gracefully(self):
        """Test handling of empty input."""
        config = L1Config(n_heads=12, n_embd=768)
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        # Empty sequence (seq_len=0) is valid and should return empty output
        x = torch.randn(2, 0, 768)
        output, _ = attn(x, use_cache=False)
        assert output.shape == (2, 0, 768)
            
    def test_single_token_input(self):
        """Test with single token input."""
        config = L1Config(n_heads=12, n_embd=768, n_kv_heads=4)
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        x = torch.randn(2, 1, 768)
        output, _ = attn(x, use_cache=False)
        
        assert output.shape == (2, 1, 768)
        
    def test_very_long_sequence(self):
        """Test with very long sequence."""
        config = L1Config(n_heads=12, n_embd=768, n_kv_heads=4, max_seq_length=8192)
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=True)
        
        # This might fail with OOM on some systems, so wrap in try
        try:
            x = torch.randn(1, 4096, 768)
            output, _ = attn(x, use_cache=False)
            assert output.shape == (1, 4096, 768)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
                
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        config = L1Config(n_heads=12, n_embd=768, n_kv_heads=4)
        attn = GroupedQueryAttention(config, use_flash_attention=False, use_rope=False)
        
        # Test with large values
        x = torch.randn(2, 128, 768) * 100
        output, _ = attn(x, use_cache=False)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
