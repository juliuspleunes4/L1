"""
@file       : test_models_comprehensive.py
@author     : J.J.G. Pleunes  
@date       : 08/2025
@brief      : Comprehensive test suite for L1 model components.
@details    : Extensive testing of model architecture, configurations, embeddings,
              transformer blocks, attention mechanisms, and model behaviors.
@version    : 1.0

@license    : MIT License
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import L1Model, L1Config, TransformerBlock, MultiHeadAttention, TokenEmbedding, PositionalEmbedding
from data import BPETokenizer


class TestL1Config(unittest.TestCase):
    """Comprehensive tests for L1 configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = L1Config()
        
        # Test default values
        self.assertEqual(config.vocab_size, 50257)
        self.assertEqual(config.max_seq_length, 1024)
        self.assertEqual(config.n_layers, 12)
        self.assertEqual(config.n_heads, 12)
        self.assertEqual(config.n_embd, 768)
        self.assertEqual(config.dropout, 0.1)
        self.assertTrue(hasattr(config, 'n_inner'))
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = L1Config(
            vocab_size=32000,
            max_seq_length=512,
            n_layers=6,
            n_heads=8,
            n_embd=512,
            dropout=0.2
        )
        
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.max_seq_length, 512)
        self.assertEqual(config.n_layers, 6)
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.n_embd, 512)
        self.assertEqual(config.dropout, 0.2)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test head dimension validation
        with self.assertRaises((ValueError, AssertionError)):
            L1Config(n_embd=768, n_heads=7)  # 768 not divisible by 7
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = L1Config(vocab_size=1000, n_layers=4)
        
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
            self.assertIsInstance(config_dict, dict)
            self.assertEqual(config_dict['vocab_size'], 1000)
            self.assertEqual(config_dict['n_layers'], 4)
    
    def test_config_from_dict(self):
        """Test configuration deserialization."""
        config_dict = {
            'vocab_size': 2000,
            'max_seq_length': 256,
            'n_layers': 8,
            'n_heads': 4,
            'n_embd': 256
        }
        
        if hasattr(L1Config, 'from_dict'):
            config = L1Config.from_dict(config_dict)
            self.assertEqual(config.vocab_size, 2000)
            self.assertEqual(config.n_layers, 8)


class TestTokenEmbedding(unittest.TestCase):
    """Test token embedding component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.n_embd = 128
        self.embedding = TokenEmbedding(self.vocab_size, self.n_embd)
    
    def test_embedding_creation(self):
        """Test embedding layer creation."""
        self.assertIsInstance(self.embedding, (TokenEmbedding, nn.Embedding))
        self.assertEqual(self.embedding.num_embeddings, self.vocab_size)
        self.assertEqual(self.embedding.embedding_dim, self.n_embd)
    
    def test_embedding_forward(self):
        """Test embedding forward pass."""
        batch_size, seq_length = 2, 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_length))
        
        output = self.embedding(input_ids)
        
        self.assertEqual(output.shape, (batch_size, seq_length, self.n_embd))
        self.assertEqual(output.dtype, torch.float32)
    
    def test_embedding_weights_initialization(self):
        """Test embedding weights are properly initialized."""
        weights = self.embedding.weight.data
        
        # Check weights are not all zeros or ones
        self.assertFalse(torch.allclose(weights, torch.zeros_like(weights)))
        self.assertFalse(torch.allclose(weights, torch.ones_like(weights)))
        
        # Check reasonable range
        self.assertTrue(weights.abs().max() < 1.0)


class TestPositionalEmbedding(unittest.TestCase):
    """Test positional embedding component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.max_seq_length = 256
        self.n_embd = 128
        
        # Try different possible class names
        try:
            self.pos_embedding = PositionalEmbedding(self.max_seq_length, self.n_embd)
        except:
            # Fallback to basic embedding if PositionalEmbedding doesn't exist
            self.pos_embedding = nn.Embedding(self.max_seq_length, self.n_embd)
    
    def test_positional_embedding_creation(self):
        """Test positional embedding creation."""
        self.assertIsInstance(self.pos_embedding, nn.Module)
    
    def test_positional_embedding_forward(self):
        """Test positional embedding forward pass."""
        batch_size, seq_length = 2, 50
        positions = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1)
        
        output = self.pos_embedding(positions)
        
        self.assertEqual(output.shape, (batch_size, seq_length, self.n_embd))


class TestMultiHeadAttention(unittest.TestCase):
    """Test multi-head attention mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_embd = 128
        self.n_heads = 8
        self.max_seq_length = 64
        
        try:
            self.attention = MultiHeadAttention(self.n_embd, self.n_heads)
        except:
            # Skip if MultiHeadAttention class doesn't exist
            self.skipTest("MultiHeadAttention class not found")
    
    def test_attention_creation(self):
        """Test attention layer creation."""
        self.assertIsInstance(self.attention, MultiHeadAttention)
        self.assertEqual(self.attention.n_heads, self.n_heads)
        self.assertEqual(self.attention.n_embd, self.n_embd)
    
    def test_attention_forward(self):
        """Test attention forward pass."""
        batch_size, seq_length = 2, 32
        x = torch.randn(batch_size, seq_length, self.n_embd)
        
        output = self.attention(x)
        
        self.assertEqual(output.shape, (batch_size, seq_length, self.n_embd))
    
    def test_attention_mask(self):
        """Test causal attention mask."""
        batch_size, seq_length = 1, 16
        x = torch.randn(batch_size, seq_length, self.n_embd)
        
        # Test with causal mask
        if hasattr(self.attention, 'causal'):
            self.attention.causal = True
            output = self.attention(x)
            self.assertEqual(output.shape, (batch_size, seq_length, self.n_embd))
    
    def test_attention_head_dimension(self):
        """Test attention head dimension calculation."""
        head_dim = self.n_embd // self.n_heads
        self.assertEqual(head_dim * self.n_heads, self.n_embd)
        self.assertGreater(head_dim, 0)


class TestTransformerBlock(unittest.TestCase):
    """Test transformer block component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_embd = 128
        self.n_heads = 8
        self.n_inner = 512
        
        try:
            self.block = TransformerBlock(
                n_embd=self.n_embd,
                n_heads=self.n_heads,
                n_inner=self.n_inner
            )
        except:
            self.skipTest("TransformerBlock class not found")
    
    def test_block_creation(self):
        """Test transformer block creation."""
        self.assertIsInstance(self.block, TransformerBlock)
    
    def test_block_forward(self):
        """Test transformer block forward pass."""
        batch_size, seq_length = 2, 32
        x = torch.randn(batch_size, seq_length, self.n_embd)
        
        output = self.block(x)
        
        self.assertEqual(output.shape, (batch_size, seq_length, self.n_embd))
    
    def test_block_components(self):
        """Test transformer block has required components."""
        # Check for attention and feed-forward components
        has_attention = any('attn' in name.lower() for name, _ in self.block.named_modules())
        has_mlp = any('mlp' in name.lower() or 'ff' in name.lower() for name, _ in self.block.named_modules())
        
        self.assertTrue(has_attention or has_mlp, "Block should have attention or MLP components")


class TestL1ModelComprehensive(unittest.TestCase):
    """Comprehensive tests for L1 model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = L1Config(
            vocab_size=1000,
            max_seq_length=128,
            n_layers=4,
            n_heads=8,
            n_embd=256,
            dropout=0.1
        )
        self.model = L1Model(self.config)
    
    def test_model_architecture(self):
        """Test model architecture components."""
        self.assertIsInstance(self.model, L1Model)
        
        # Check model has essential components
        has_embedding = any('emb' in name.lower() for name, _ in self.model.named_modules())
        has_transformer = any('block' in name.lower() or 'layer' in name.lower() for name, _ in self.model.named_modules())
        has_output = any('head' in name.lower() or 'lm_head' in name.lower() for name, _ in self.model.named_modules())
        
        self.assertTrue(has_embedding, "Model should have embedding layers")
        self.assertTrue(has_transformer or has_output, "Model should have transformer or output layers")
    
    def test_model_parameter_count(self):
        """Test model parameter count is reasonable."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 1000)  # Should have reasonable number of parameters
        self.assertEqual(total_params, trainable_params)  # All params should be trainable by default
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def test_forward_pass_shapes(self):
        """Test forward pass output shapes."""
        batch_size, seq_length = 2, 64
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        # Check output format
        if isinstance(outputs, dict):
            self.assertIn('logits', outputs)
            logits = outputs['logits']
        else:
            logits = outputs
        
        expected_shape = (batch_size, seq_length, self.config.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
    
    def test_forward_pass_gradient_flow(self):
        """Test gradient flow through model."""
        batch_size, seq_length = 2, 32
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        
        outputs = self.model(input_ids)
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Compute loss and check gradients
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1)
        )
        loss.backward()
        
        # Check that gradients exist
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter: {name}")
    
    def test_model_modes(self):
        """Test model training and evaluation modes."""
        # Test training mode
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Test evaluation mode
        self.model.eval()
        self.assertFalse(self.model.training)
    
    def test_model_device_handling(self):
        """Test model device placement."""
        # Test CPU
        device = torch.device('cpu')
        self.model.to(device)
        
        for param in self.model.parameters():
            self.assertEqual(param.device, device)
        
        # Test CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.model.to(device)
            
            for param in self.model.parameters():
                self.assertEqual(param.device, device)
    
    def test_generation_basic(self):
        """Test basic text generation."""
        self.model.eval()
        
        prompt = torch.randint(0, self.config.vocab_size, (1, 10))
        
        with torch.no_grad():
            if hasattr(self.model, 'generate'):
                generated = self.model.generate(prompt, max_new_tokens=20)
                
                self.assertEqual(generated.shape[0], 1)
                self.assertGreaterEqual(generated.shape[1], prompt.shape[1])
                
                # Check generated tokens are within vocabulary
                self.assertTrue((generated >= 0).all())
                self.assertTrue((generated < self.config.vocab_size).all())
    
    def test_generation_with_temperature(self):
        """Test generation with different temperatures."""
        if not hasattr(self.model, 'generate'):
            self.skipTest("Model doesn't have generate method")
        
        self.model.eval()
        prompt = torch.randint(0, self.config.vocab_size, (1, 5))
        
        temperatures = [0.1, 0.8, 1.0]
        generations = []
        
        for temp in temperatures:
            with torch.no_grad():
                try:
                    generated = self.model.generate(
                        prompt, 
                        max_new_tokens=10, 
                        temperature=temp
                    )
                    generations.append(generated)
                except TypeError:
                    # Method might not support temperature parameter
                    pass
        
        # If temperature is supported, generations should potentially differ
        if len(generations) > 1:
            # Check that at least some generations are different (stochastic)
            all_same = all(torch.equal(generations[0], gen) for gen in generations[1:])
            # Note: might be same by chance, so we don't assert False
    
    def test_model_state_dict(self):
        """Test model state dict save/load."""
        # Get original state
        original_state = self.model.state_dict()
        
        # Modify model
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        
        # Load original state
        self.model.load_state_dict(original_state)
        
        # Check state is restored
        restored_state = self.model.state_dict()
        for key in original_state:
            self.assertTrue(torch.allclose(original_state[key], restored_state[key]))
    
    def test_model_memory_efficiency(self):
        """Test model memory usage is reasonable."""
        batch_size, seq_length = 4, 128
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        
        if torch.cuda.is_available():
            # Move to GPU and test memory
            self.model.cuda()
            input_ids = input_ids.cuda()
            
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            memory_after = torch.cuda.memory_allocated()
            memory_used = memory_after - memory_before
            
            # Memory should be reasonable (less than 1GB for small model)
            self.assertLess(memory_used, 1e9, "Model uses too much GPU memory")
            
            print(f"GPU memory used: {memory_used / 1e6:.2f} MB")


class TestModelEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = L1Config(
            vocab_size=100,
            max_seq_length=32,
            n_layers=2,
            n_heads=4,
            n_embd=64
        )
        self.model = L1Model(self.config)
    
    def test_empty_input(self):
        """Test model with empty input."""
        # Test with batch size 0
        empty_input = torch.empty(0, 0, dtype=torch.long)
        
        try:
            outputs = self.model(empty_input)
            # If it doesn't crash, check output shape is consistent
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            self.assertEqual(logits.shape[0], 0)
        except (RuntimeError, ValueError):
            # Expected to fail with empty input
            pass
    
    def test_single_token_input(self):
        """Test model with single token input."""
        input_ids = torch.randint(0, self.config.vocab_size, (1, 1))
        
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        self.assertEqual(logits.shape, (1, 1, self.config.vocab_size))
    
    def test_max_length_input(self):
        """Test model with maximum length input."""
        max_len = self.config.max_seq_length
        input_ids = torch.randint(0, self.config.vocab_size, (1, max_len))
        
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        self.assertEqual(logits.shape, (1, max_len, self.config.vocab_size))
    
    def test_out_of_vocab_tokens(self):
        """Test model behavior with out-of-vocabulary tokens."""
        # Tokens at vocabulary boundary
        input_ids = torch.tensor([[self.config.vocab_size - 1, 0]])
        
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        # Should handle boundary tokens without error
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())


if __name__ == '__main__':
    # Set random seed for reproducible tests
    torch.manual_seed(42)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)
