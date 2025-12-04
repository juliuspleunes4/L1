"""
@file       : test_model_tokenizer_integration.py
@package    : tests
@author     : J.J.G. Pleunes
@date       : 12/2024
@brief      : Tests for model-tokenizer integration
@details    : Comprehensive tests ensuring model works correctly with different tokenizer vocab sizes
@version    : 1.0
"""

import pytest
import sys
import torch
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import BPETokenizer
from src.models.config import L1Config
from src.models.transformer import L1Model
from src.models.embeddings import TokenEmbedding


class TestTokenizerLoading:
    """Test tokenizer loading and basic properties."""
    
    def test_load_32k_tokenizer(self):
        """Test loading existing 32K tokenizer."""
        tokenizer_path = Path("data/processed/tokenizer.json")
        
        if not tokenizer_path.exists():
            pytest.skip("32K tokenizer not found")
        
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        
        assert tokenizer is not None
        assert len(tokenizer.vocab) > 0
        assert tokenizer.vocab_size > 0
        
    def test_load_50k_tokenizer(self):
        """Test loading new 50K tokenizer."""
        tokenizer_path = Path("data/processed/tokenizer_50000.json")
        
        if not tokenizer_path.exists():
            pytest.skip("50K tokenizer not found")
        
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        
        assert tokenizer is not None
        assert len(tokenizer.vocab) == 50000
        assert tokenizer.vocab_size == 50000
        
    def test_tokenizer_encode_decode(self):
        """Test basic encode/decode functionality."""
        tokenizer_path = Path("data/processed/tokenizer_50000.json")
        
        if not tokenizer_path.exists():
            pytest.skip("50K tokenizer not found")
        
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        
        test_text = "The quick brown fox jumps over the lazy dog."
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        assert all(0 <= token_id < tokenizer.vocab_size for token_id in encoded)
        assert isinstance(decoded, str)
        assert len(decoded) > 0


class TestConfigTokenizerCompatibility:
    """Test L1Config compatibility with different tokenizers."""
    
    def test_config_with_32k_vocab(self):
        """Test config with 32K vocabulary."""
        config = L1Config(vocab_size=32000)
        
        assert config.vocab_size == 32000
        assert config.n_embd > 0
        assert config.n_heads > 0
        
    def test_config_with_50k_vocab(self):
        """Test config with 50K vocabulary."""
        config = L1Config(vocab_size=50000)
        
        assert config.vocab_size == 50000
        assert config.n_embd > 0
        assert config.n_heads > 0
        
    def test_config_vocab_size_mismatch_detection(self):
        """Test that we can detect vocab size mismatches."""
        tokenizer_path = Path("data/processed/tokenizer_50000.json")
        
        if not tokenizer_path.exists():
            pytest.skip("50K tokenizer not found")
        
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        
        # Create config with wrong vocab size
        config = L1Config(vocab_size=32000)
        
        # This should be detected as a mismatch
        assert config.vocab_size != len(tokenizer.vocab)


class TestEmbeddingLayerCompatibility:
    """Test embedding layer with different vocab sizes."""
    
    def test_embedding_layer_32k(self):
        """Test embedding layer with 32K vocab."""
        vocab_size = 32000
        embed_dim = 240
        embedding = TokenEmbedding(vocab_size, embed_dim)
        
        assert embedding.vocab_size == 32000
        assert embedding.embed_dim == 240
        
    def test_embedding_layer_50k(self):
        """Test embedding layer with 50K vocab."""
        vocab_size = 50000
        embed_dim = 240
        embedding = TokenEmbedding(vocab_size, embed_dim)
        
        assert embedding.vocab_size == 50000
        assert embedding.embed_dim == 240
        
    def test_embedding_forward_pass_32k(self):
        """Test forward pass through 32K embedding."""
        vocab_size = 32000
        embed_dim = 120
        embedding = TokenEmbedding(vocab_size, embed_dim)
        
        # Create sample input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 32000, (batch_size, seq_len))
        
        output = embedding(input_ids)
        
        assert output.shape == (batch_size, seq_len, 120)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_embedding_forward_pass_50k(self):
        """Test forward pass through 50K embedding."""
        vocab_size = 50000
        embed_dim = 120
        embedding = TokenEmbedding(vocab_size, embed_dim)
        
        # Create sample input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))
        
        output = embedding(input_ids)
        
        assert output.shape == (batch_size, seq_len, 120)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_embedding_out_of_bounds(self):
        """Test that out-of-bounds tokens cause errors."""
        vocab_size = 1000
        embed_dim = 60
        embedding = TokenEmbedding(vocab_size, embed_dim)
        
        # Token ID beyond vocab size
        input_ids = torch.tensor([[1500]])
        
        with pytest.raises(IndexError):
            embedding(input_ids)
            
    def test_embedding_gradient_flow(self):
        """Test that gradients flow through embedding layer."""
        vocab_size = 1000
        embed_dim = 60
        embedding = TokenEmbedding(vocab_size, embed_dim)
        
        input_ids = torch.randint(0, 1000, (2, 5))
        output = embedding(input_ids)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert embedding.embedding.weight.grad is not None
        assert not torch.isnan(embedding.embedding.weight.grad).any()


class TestModelTokenizerIntegration:
    """Test full model with different tokenizers."""
    
    @pytest.fixture
    def small_config_32k(self):
        """Create small model config with 32K vocab."""
        return L1Config(
            vocab_size=32000,
            max_seq_length=128,
            n_layers=2,
            n_heads=4,
            n_embd=128,
            dropout=0.0
        )
    
    @pytest.fixture
    def small_config_50k(self):
        """Create small model config with 50K vocab."""
        return L1Config(
            vocab_size=50000,
            max_seq_length=128,
            n_layers=2,
            n_heads=4,
            n_embd=128,
            dropout=0.0
        )
    
    def test_model_initialization_32k(self, small_config_32k):
        """Test model initializes correctly with 32K vocab."""
        model = L1Model(small_config_32k)
        
        assert model.config.vocab_size == 32000
        assert model.token_embedding.embedding.num_embeddings == 32000
        
    def test_model_initialization_50k(self, small_config_50k):
        """Test model initializes correctly with 50K vocab."""
        model = L1Model(small_config_50k)
        
        assert model.config.vocab_size == 50000
        assert model.token_embedding.embedding.num_embeddings == 50000
        
    def test_model_forward_pass_32k(self, small_config_32k):
        """Test forward pass with 32K vocab."""
        model = L1Model(small_config_32k)
        model.eval()
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 32000, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
        
        logits = output['logits']
        assert logits.shape == (batch_size, seq_len, 32000)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        
    def test_model_forward_pass_50k(self, small_config_50k):
        """Test forward pass with 50K vocab."""
        model = L1Model(small_config_50k)
        model.eval()
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
        
        logits = output['logits']
        assert logits.shape == (batch_size, seq_len, 50000)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        
    def test_model_with_tokenizer_encode(self, small_config_50k):
        """Test model with actual tokenizer encoding."""
        tokenizer_path = Path("data/processed/tokenizer_50000.json")
        
        if not tokenizer_path.exists():
            pytest.skip("50K tokenizer not found")
        
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        model = L1Model(small_config_50k)
        model.eval()
        
        # Encode text
        text = "Hello, world! This is a test."
        input_ids = torch.tensor([tokenizer.encode(text)])
        
        # Forward pass
        with torch.no_grad():
            output = model(input_ids)
        
        logits = output['logits']
        assert logits.shape[0] == 1
        assert logits.shape[2] == 50000
        
    def test_model_training_step_32k(self, small_config_32k):
        """Test training step with 32K vocab."""
        model = L1Model(small_config_32k)
        model.train()
        
        input_ids = torch.randint(0, 32000, (2, 10))
        target_ids = torch.randint(0, 32000, (2, 10))
        
        output = model(input_ids)
        logits = output['logits']
        
        # Compute loss manually
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        assert loss.requires_grad
        assert not torch.isnan(loss)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert model.token_embedding.embedding.weight.grad is not None
        
    def test_model_training_step_50k(self, small_config_50k):
        """Test training step with 50K vocab."""
        model = L1Model(small_config_50k)
        model.train()
        
        input_ids = torch.randint(0, 50000, (2, 10))
        target_ids = torch.randint(0, 50000, (2, 10))
        
        output = model(input_ids)
        logits = output['logits']
        
        # Compute loss manually
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        assert loss.requires_grad
        assert not torch.isnan(loss)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert model.token_embedding.embedding.weight.grad is not None
        
    def test_model_save_load_32k(self, small_config_32k):
        """Test saving and loading model with 32K vocab."""
        model = L1Model(small_config_32k)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            
            # Save model and config
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': small_config_32k
            }, save_path)
            
            # Load model
            checkpoint = torch.load(save_path, weights_only=False)
            loaded_config = checkpoint['config']
            loaded_model = L1Model(loaded_config)
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Verify
            assert loaded_model.config.vocab_size == 32000
            
    def test_model_save_load_50k(self, small_config_50k):
        """Test saving and loading model with 50K vocab."""
        model = L1Model(small_config_50k)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            
            # Save model and config
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': small_config_50k
            }, save_path)
            
            # Load model
            checkpoint = torch.load(save_path, weights_only=False)
            loaded_config = checkpoint['config']
            loaded_model = L1Model(loaded_config)
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Verify
            assert loaded_model.config.vocab_size == 50000


class TestTextGeneration:
    """Test text generation with different tokenizers."""
    
    def test_greedy_generation_50k(self):
        """Test greedy generation with 50K tokenizer."""
        tokenizer_path = Path("data/processed/tokenizer_50000.json")
        
        if not tokenizer_path.exists():
            pytest.skip("50K tokenizer not found")
        
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        
        config = L1Config(
            vocab_size=50000,
            max_seq_length=128,
            n_layers=2,
            n_heads=4,
            n_embd=128,
            dropout=0.0
        )
        model = L1Model(config)
        model.eval()
        
        # Encode prompt
        prompt = "Hello"
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        
        # Generate (greedy)
        max_new_tokens = 10
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                output = model(generated_ids)
                logits = output['logits']
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop if max length reached
                if generated_ids.shape[1] >= config.max_seq_length:
                    break
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        
        assert isinstance(generated_text, str)
        assert len(generated_text) > len(prompt)
        
    def test_generation_token_ids_in_range(self):
        """Test that generated token IDs are within vocab range."""
        config = L1Config(
            vocab_size=50000,
            max_seq_length=128,
            n_layers=2,
            n_heads=4,
            n_embd=128,
            dropout=0.0
        )
        model = L1Model(config)
        model.eval()
        
        input_ids = torch.randint(0, 50000, (1, 5))
        
        with torch.no_grad():
            output = model(input_ids)
            logits = output['logits']
            next_token = logits[:, -1, :].argmax(dim=-1)
        
        assert 0 <= next_token.item() < 50000


class TestEdgeCases:
    """Test edge cases in model-tokenizer integration."""
    
    def test_empty_input(self):
        """Test model with empty input."""
        config = L1Config(vocab_size=1000, n_layers=1, n_heads=2, n_embd=64)
        model = L1Model(config)
        model.eval()
        
        # Empty input (batch_size=1, seq_len=0)
        input_ids = torch.empty((1, 0), dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_ids)
        
        assert output['logits'].shape == (1, 0, 1000)
        
    def test_single_token_input(self):
        """Test model with single token."""
        config = L1Config(vocab_size=1000, n_layers=1, n_heads=2, n_embd=64)
        model = L1Model(config)
        model.eval()
        
        input_ids = torch.tensor([[42]])
        
        with torch.no_grad():
            output = model(input_ids)
        
        assert output['logits'].shape == (1, 1, 1000)
        
    def test_max_length_input(self):
        """Test model with max length input."""
        max_len = 128
        config = L1Config(
            vocab_size=1000,
            max_seq_length=max_len,
            n_layers=1,
            n_heads=2,
            n_embd=64
        )
        model = L1Model(config)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (1, max_len))
        
        with torch.no_grad():
            output = model(input_ids)
        
        assert output['logits'].shape == (1, max_len, 1000)
        
    def test_batch_size_1_vs_many(self):
        """Test that batch_size=1 and batch_size>1 produce consistent results."""
        config = L1Config(vocab_size=1000, n_layers=1, n_heads=2, n_embd=64, dropout=0.0)
        model = L1Model(config)
        model.eval()
        
        input_ids_single = torch.tensor([[1, 2, 3, 4, 5]])
        input_ids_batch = input_ids_single.repeat(3, 1)  # batch_size=3
        
        with torch.no_grad():
            output_single = model(input_ids_single)
            output_batch = model(input_ids_batch)
        
        # First sample in batch should match single
        assert torch.allclose(output_single['logits'], output_batch['logits'][0:1], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
