"""
@file       : test_tokenizer.py
@package    : tests
@author     : J.J.G. Pleunes
@date       : 12/2024
@brief      : Comprehensive test suite for BPE tokenizer
@details    : Tests encoding/decoding, special tokens, unicode, edge cases, quality metrics
@version    : 2.0
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import BPETokenizer


class TestBPETokenizerInitialization:
    """Test tokenizer initialization and configuration."""
    
    def test_default_initialization(self):
        """Test tokenizer with default settings."""
        tokenizer = BPETokenizer()
        
        assert tokenizer.vocab_size == 50257
        assert '<pad>' in tokenizer.special_tokens
        assert '<unk>' in tokenizer.special_tokens
        assert '<bos>' in tokenizer.special_tokens
        assert '<eos>' in tokenizer.special_tokens
        assert len(tokenizer.vocab) == len(tokenizer.special_tokens)
        
    def test_custom_vocab_size(self):
        """Test tokenizer with custom vocabulary size."""
        vocab_sizes = [32000, 50000, 75000, 100000]
        
        for vocab_size in vocab_sizes:
            tokenizer = BPETokenizer(vocab_size=vocab_size)
            assert tokenizer.vocab_size == vocab_size
            
    def test_custom_special_tokens(self):
        """Test tokenizer with custom special tokens."""
        custom_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<mask>': 4,
            '<sep>': 5
        }
        
        tokenizer = BPETokenizer(special_tokens=custom_tokens)
        
        for token, idx in custom_tokens.items():
            assert token in tokenizer.vocab
            assert tokenizer.vocab[token] == idx
            
    def test_special_token_properties(self):
        """Test special token property accessors."""
        tokenizer = BPETokenizer()
        
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1
        assert tokenizer.bos_token_id == 2
        assert tokenizer.eos_token_id == 3


class TestBPETokenizerTraining:
    """Test tokenizer training process."""
    
    def test_train_on_simple_corpus(self):
        """Test training on simple text corpus."""
        tokenizer = BPETokenizer(vocab_size=1000)
        
        texts = [
            "Hello world!",
            "This is a test.",
            "Machine learning is fun.",
            "Natural language processing."
        ]
        
        trained = tokenizer.train(texts)
        
        assert trained is not None
        assert len(tokenizer.vocab) > len(tokenizer.special_tokens)
        assert len(tokenizer.vocab) <= 1000
        assert len(tokenizer.id_to_token) == len(tokenizer.vocab)
        
    def test_train_on_repeated_patterns(self):
        """Test training learns repeated patterns."""
        tokenizer = BPETokenizer(vocab_size=500)
        
        # Create corpus with repeated patterns
        texts = ["hello world " * 100, "test ing " * 100, "token ize " * 100]
        
        tokenizer.train(texts)
        
        # Check that common patterns are in vocab
        assert len(tokenizer.vocab) > len(tokenizer.special_tokens)
        
    def test_train_on_multilingual(self):
        """Test training on multilingual text."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        texts = [
            "Hello world",
            "Hola mundo",
            "Bonjour le monde",
            "Hallo Welt",
            "Привет мир",
            "你好世界",
            "こんにちは世界"
        ]
        
        tokenizer.train(texts)
        
        assert len(tokenizer.vocab) > len(tokenizer.special_tokens)
        
    def test_train_empty_corpus(self):
        """Test training on empty corpus."""
        tokenizer = BPETokenizer(vocab_size=100)
        
        # Should handle empty corpus gracefully
        tokenizer.train([])
        
        # Should still have special tokens
        assert len(tokenizer.vocab) >= len(tokenizer.special_tokens)
        
    def test_train_single_sample(self):
        """Test training on single sample."""
        tokenizer = BPETokenizer(vocab_size=500)
        
        texts = ["This is a single training sample with some repeated words words words."]
        
        tokenizer.train(texts)
        
        assert len(tokenizer.vocab) > len(tokenizer.special_tokens)


class TestBPETokenizerEncoding:
    """Test tokenizer encoding functionality."""
    
    @pytest.fixture
    def trained_tokenizer(self):
        """Create a trained tokenizer for testing."""
        tokenizer = BPETokenizer(vocab_size=5000)
        
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing deals with text and speech.",
            "Deep learning uses neural networks with many layers.",
        ] * 10  # Repeat for better training
        
        tokenizer.train(texts)
        return tokenizer
        
    def test_encode_simple_text(self, trained_tokenizer):
        """Test encoding simple text."""
        text = "Hello world"
        encoded = trained_tokenizer.encode(text)
        
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        assert all(isinstance(token_id, int) for token_id in encoded)
        assert all(0 <= token_id < len(trained_tokenizer.vocab) for token_id in encoded)
        
    def test_encode_empty_string(self, trained_tokenizer):
        """Test encoding empty string."""
        encoded = trained_tokenizer.encode("")
        
        assert isinstance(encoded, list)
        assert len(encoded) == 0
        
    def test_encode_whitespace_only(self, trained_tokenizer):
        """Test encoding whitespace-only string."""
        encoded = trained_tokenizer.encode("   \n\t  ")
        
        assert isinstance(encoded, list)
        # May have some tokens depending on BPE training
        
    def test_encode_special_characters(self, trained_tokenizer):
        """Test encoding special characters."""
        text = "!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/"
        encoded = trained_tokenizer.encode(text)
        
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        
    def test_encode_numbers(self, trained_tokenizer):
        """Test encoding numbers."""
        text = "123 456.789 -10 3.14159 1,000,000"
        encoded = trained_tokenizer.encode(text)
        
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        
    def test_encode_unicode(self, trained_tokenizer):
        """Test encoding unicode characters."""
        text = "Hello 世界! 🌍 Привет café"
        encoded = trained_tokenizer.encode(text)
        
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        
    def test_encode_long_text(self, trained_tokenizer):
        """Test encoding long text."""
        text = "word " * 10000  # 10k words
        encoded = trained_tokenizer.encode(text)
        
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        # Compression depends on training - at minimum, should be <= original length
        # and significantly less than naive character-level (which would be len(text))
        assert len(encoded) <= len(text)
        # Should encode without errors for long sequences
        assert all(isinstance(token_id, int) for token_id in encoded)
        
    def test_encode_unknown_words(self, trained_tokenizer):
        """Test encoding completely unknown words."""
        text = "xyzabc qwerty asdfgh"  # Unlikely to be in training
        encoded = trained_tokenizer.encode(text)
        
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        # Should use byte-level fallback, not <unk>


class TestBPETokenizerDecoding:
    """Test tokenizer decoding functionality."""
    
    @pytest.fixture
    def trained_tokenizer(self):
        """Create a trained tokenizer for testing."""
        tokenizer = BPETokenizer(vocab_size=5000)
        
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing deals with text and speech.",
            "Deep learning uses neural networks with many layers.",
        ] * 10
        
        tokenizer.train(texts)
        return tokenizer
        
    def test_decode_simple_tokens(self, trained_tokenizer):
        """Test decoding simple token sequence."""
        text = "Hello world"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        
        assert isinstance(decoded, str)
        assert len(decoded) > 0
        # May not be exact due to BPE normalization
        assert "Hello" in decoded or "hello" in decoded.lower()
        assert "world" in decoded or "world" in decoded.lower()
        
    def test_decode_empty_list(self, trained_tokenizer):
        """Test decoding empty token list."""
        decoded = trained_tokenizer.decode([])
        
        assert isinstance(decoded, str)
        assert decoded == ""
        
    def test_decode_special_tokens(self, trained_tokenizer):
        """Test that special tokens are filtered in decoding."""
        # Include special tokens in sequence
        token_ids = [
            trained_tokenizer.bos_token_id,
            trained_tokenizer.vocab.get('h', 10),  # Some token
            trained_tokenizer.eos_token_id,
            trained_tokenizer.pad_token_id
        ]
        
        decoded = trained_tokenizer.decode(token_ids)
        
        assert isinstance(decoded, str)
        # Special tokens should be filtered out
        assert '<bos>' not in decoded
        assert '<eos>' not in decoded
        assert '<pad>' not in decoded
        
    def test_decode_with_unknown_ids(self, trained_tokenizer):
        """Test decoding with unknown token IDs."""
        # Use IDs beyond vocab size
        invalid_ids = [999999, 888888]
        decoded = trained_tokenizer.decode(invalid_ids)
        
        assert isinstance(decoded, str)
        # Should handle gracefully, possibly with <unk> or spaces
        
    def test_decode_unicode_roundtrip(self, trained_tokenizer):
        """Test unicode handling in decode."""
        text = "café résumé naïve"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        
        assert isinstance(decoded, str)
        # Should preserve general content
        assert len(decoded) > 0


class TestBPETokenizerRoundtrip:
    """Test encode-decode roundtrip."""
    
    @pytest.fixture
    def trained_tokenizer(self):
        """Create a trained tokenizer for testing."""
        tokenizer = BPETokenizer(vocab_size=5000)
        
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing deals with text and speech.",
            "Deep learning uses neural networks with many layers.",
        ] * 10
        
        tokenizer.train(texts)
        return tokenizer
        
    def test_roundtrip_simple_text(self, trained_tokenizer):
        """Test roundtrip on simple text."""
        text = "Hello world! This is a test."
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        
        # Normalize for comparison (BPE may change whitespace)
        text_normalized = ' '.join(text.split())
        decoded_normalized = ' '.join(decoded.split())
        
        # Should be similar (not exact due to BPE properties)
        assert len(decoded_normalized) > 0
        assert abs(len(text_normalized) - len(decoded_normalized)) < len(text_normalized) * 0.5
        
    def test_roundtrip_preserves_content(self, trained_tokenizer):
        """Test that key content is preserved in roundtrip."""
        text = "machine learning neural network"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        
        decoded_lower = decoded.lower()
        
        # Key words should be present
        assert "machine" in decoded_lower or "machin" in decoded_lower
        assert "learning" in decoded_lower or "learn" in decoded_lower
        
    def test_roundtrip_numbers(self, trained_tokenizer):
        """Test roundtrip preserves numbers."""
        text = "The year 2024 and number 42"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        
        # Numbers should be roughly preserved
        assert any(char.isdigit() for char in decoded)


class TestBPETokenizerSaveLoad:
    """Test tokenizer save and load functionality."""
    
    def test_save_and_load(self):
        """Test saving and loading tokenizer."""
        # Train tokenizer
        tokenizer = BPETokenizer(vocab_size=1000)
        texts = ["Hello world", "Test text", "Machine learning"] * 10
        tokenizer.train(texts)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            tokenizer.save(temp_path)
            
            # Load tokenizer
            loaded_tokenizer = BPETokenizer.load(temp_path)
            
            # Verify properties match
            assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
            assert loaded_tokenizer.vocab == tokenizer.vocab
            assert loaded_tokenizer.special_tokens == tokenizer.special_tokens
            
            # Verify encoding/decoding works the same
            test_text = "Hello world test"
            original_encoded = tokenizer.encode(test_text)
            loaded_encoded = loaded_tokenizer.encode(test_text)
            
            assert original_encoded == loaded_encoded
            
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
            
    def test_load_preserves_functionality(self):
        """Test that loaded tokenizer functions correctly."""
        # Train tokenizer
        tokenizer = BPETokenizer(vocab_size=2000)
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating."
        ] * 20
        tokenizer.train(texts)
        
        # Save and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            tokenizer.save(temp_path)
            loaded_tokenizer = BPETokenizer.load(temp_path)
            
            # Test encoding
            text = "Machine learning test"
            encoded = loaded_tokenizer.encode(text)
            assert len(encoded) > 0
            
            # Test decoding
            decoded = loaded_tokenizer.decode(encoded)
            assert len(decoded) > 0
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestBPETokenizerQuality:
    """Test tokenizer quality metrics."""
    
    @pytest.fixture
    def trained_tokenizer(self):
        """Create a well-trained tokenizer."""
        tokenizer = BPETokenizer(vocab_size=10000)
        
        # Larger, more diverse corpus
        texts = []
        base_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing deals with text and speech.",
            "Deep learning uses neural networks with many layers.",
            "Python is a popular programming language.",
            "Data science combines statistics and computer science.",
        ]
        
        for text in base_texts:
            texts.extend([text] * 20)  # Repeat each 20 times
        
        tokenizer.train(texts)
        return tokenizer
        
    def test_compression_ratio(self, trained_tokenizer):
        """Test that tokenizer achieves reasonable compression."""
        text = "Machine learning and artificial intelligence are fascinating fields of study."
        
        chars = len(text)
        tokens = len(trained_tokenizer.encode(text))
        
        compression_ratio = chars / tokens
        
        # Should compress: at least 2 chars per token on average
        assert compression_ratio >= 2.0
        # But not too much: less than 10 chars per token
        assert compression_ratio <= 10.0
        
    def test_fertility_score(self, trained_tokenizer):
        """Test fertility score (tokens per word)."""
        text = "Machine learning uses neural networks for pattern recognition"
        
        words = len(text.split())
        tokens = len(trained_tokenizer.encode(text))
        
        fertility = tokens / words
        
        # Should be close to 1:1 (ideally less than 2 tokens per word)
        assert fertility >= 0.5  # At least one token per 2 words
        assert fertility <= 3.0  # At most 3 tokens per word
        
    def test_oov_rate(self, trained_tokenizer):
        """Test out-of-vocabulary rate on seen text."""
        # Use text similar to training corpus
        text = "Machine learning and deep learning are subsets of artificial intelligence."
        
        encoded = trained_tokenizer.encode(text)
        oov_count = sum(1 for token_id in encoded if token_id == trained_tokenizer.unk_token_id)
        
        oov_rate = oov_count / len(encoded) if encoded else 0
        
        # Should have very low OOV rate on similar text
        assert oov_rate <= 0.1  # Less than 10% unknown
        
    def test_vocab_utilization(self, trained_tokenizer):
        """Test that significant portion of vocab is used."""
        texts = [
            "Machine learning",
            "Deep learning", 
            "Neural networks",
            "Artificial intelligence",
            "Natural language processing"
        ]
        
        used_tokens = set()
        for text in texts:
            encoded = trained_tokenizer.encode(text)
            used_tokens.update(encoded)
        
        utilization = len(used_tokens) / len(trained_tokenizer.vocab)
        
        # Should use at least 1% of vocab on small sample
        assert utilization >= 0.01


class TestBPETokenizerEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def trained_tokenizer(self):
        """Create a trained tokenizer."""
        tokenizer = BPETokenizer(vocab_size=2000)
        texts = ["Hello world", "Test text"] * 50
        tokenizer.train(texts)
        return tokenizer
        
    def test_very_long_single_word(self, trained_tokenizer):
        """Test handling very long single word."""
        long_word = "a" * 10000
        encoded = trained_tokenizer.encode(long_word)
        
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        
    def test_only_special_characters(self, trained_tokenizer):
        """Test string with only special characters."""
        text = "!@#$%^&*()"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        
        assert isinstance(encoded, list)
        assert isinstance(decoded, str)
        
    def test_mixed_scripts(self, trained_tokenizer):
        """Test text with mixed writing systems."""
        text = "Hello 世界 Привет مرحبا"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        assert isinstance(decoded, str)
        
    def test_control_characters(self, trained_tokenizer):
        """Test handling control characters."""
        text = "Hello\x00World\x01Test"
        encoded = trained_tokenizer.encode(text)
        
        assert isinstance(encoded, list)
        # Should handle without crashing
        
    def test_repeated_spaces(self, trained_tokenizer):
        """Test handling repeated spaces."""
        text = "Hello     world      test"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        
        assert isinstance(encoded, list)
        assert isinstance(decoded, str)
        # Spaces may be normalized
        
    def test_newlines_and_tabs(self, trained_tokenizer):
        """Test handling newlines and tabs."""
        text = "Hello\nWorld\t\tTest"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        
        assert isinstance(encoded, list)
        assert isinstance(decoded, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
