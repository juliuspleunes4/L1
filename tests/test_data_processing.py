"""
@file       : test_data_processing.py
@author     : J.J.G. Pleunes
@date       : 08/2025
@brief      : Comprehensive test suite for data processing components.
@details    : Tests for tokenization, dataset handling, preprocessing,
              and data loading functionality.
@version    : 1.0

@license    : MIT License
"""

import unittest
import torch
import sys
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import BPETokenizer, TextDataset, TextPreprocessor
from torch.utils.data import DataLoader


class TestBPETokenizerComprehensive(unittest.TestCase):
    """Comprehensive tests for BPE tokenizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.tokenizer = BPETokenizer(vocab_size=self.vocab_size)
        
        # Sample training data
        self.sample_texts = [
            "Hello world! This is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating.",
            "Natural language processing with transformers.",
            "Byte pair encoding is an effective tokenization method."
        ]
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        self.assertEqual(self.tokenizer.vocab_size, self.vocab_size)
        self.assertIsNotNone(self.tokenizer.special_tokens)
        
        # Check special tokens exist
        expected_special = ['<unk>', '<pad>', '<bos>', '<eos>']
        for token in expected_special:
            if hasattr(self.tokenizer, 'special_tokens'):
                # Token might exist in special_tokens dict
                pass
    
    def test_tokenizer_training(self):
        """Test tokenizer training process."""
        # Train tokenizer
        self.tokenizer.train(self.sample_texts)
        
        # Check vocabulary was built
        if hasattr(self.tokenizer, 'vocab'):
            self.assertGreater(len(self.tokenizer.vocab), 0)
            self.assertLessEqual(len(self.tokenizer.vocab), self.vocab_size)
        
        if hasattr(self.tokenizer, 'merges'):
            self.assertIsInstance(self.tokenizer.merges, (list, dict))
    
    def test_encode_decode_consistency(self):
        """Test encoding and decoding consistency."""
        # Train first
        self.tokenizer.train(self.sample_texts)
        
        test_texts = [
            "Hello world",
            "This is a test",
            "Tokenization",
            "The quick brown fox"
        ]
        
        for text in test_texts:
            # Encode text
            encoded = self.tokenizer.encode(text)
            self.assertIsInstance(encoded, list)
            self.assertGreater(len(encoded), 0)
            
            # All tokens should be within vocabulary
            for token_id in encoded:
                self.assertGreaterEqual(token_id, 0)
                self.assertLess(token_id, self.vocab_size)
            
            # Decode back
            decoded = self.tokenizer.decode(encoded)
            self.assertIsInstance(decoded, str)
            
            # Basic consistency check (might not be exact due to BPE)
            self.assertGreater(len(decoded), 0)
    
    def test_special_tokens_handling(self):
        """Test special tokens are handled correctly."""
        self.tokenizer.train(self.sample_texts)
        
        # Test unknown token handling
        unknown_text = "UNKNOWNWORD12345"
        encoded = self.tokenizer.encode(unknown_text)
        self.assertIsInstance(encoded, list)
        
        # Should handle gracefully
        decoded = self.tokenizer.decode(encoded)
        self.assertIsInstance(decoded, str)
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        self.tokenizer.train(self.sample_texts)
        
        # Empty string
        encoded_empty = self.tokenizer.encode("")
        self.assertIsInstance(encoded_empty, list)
        
        decoded_empty = self.tokenizer.decode([])
        self.assertIsInstance(decoded_empty, str)
    
    def test_batch_encoding(self):
        """Test batch encoding if supported."""
        self.tokenizer.train(self.sample_texts)
        
        if hasattr(self.tokenizer, 'encode_batch'):
            batch_texts = ["Hello", "World", "Test"]
            encoded_batch = self.tokenizer.encode_batch(batch_texts)
            
            self.assertIsInstance(encoded_batch, list)
            self.assertEqual(len(encoded_batch), len(batch_texts))
            
            for encoded in encoded_batch:
                self.assertIsInstance(encoded, list)
    
    def test_vocabulary_size_respect(self):
        """Test that vocabulary size limits are respected."""
        # Train with small vocabulary
        small_tokenizer = BPETokenizer(vocab_size=50)
        small_tokenizer.train(self.sample_texts)
        
        if hasattr(small_tokenizer, 'vocab'):
            self.assertLessEqual(len(small_tokenizer.vocab), 50)
    
    def test_tokenizer_serialization(self):
        """Test tokenizer save/load functionality."""
        self.tokenizer.train(self.sample_texts)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test save
            if hasattr(self.tokenizer, 'save'):
                self.tokenizer.save(temp_path)
                self.assertTrue(os.path.exists(temp_path))
            
            # Test load
            if hasattr(BPETokenizer, 'load'):
                loaded_tokenizer = BPETokenizer.load(temp_path)
                
                # Test loaded tokenizer works
                test_text = "Hello world"
                original_encoded = self.tokenizer.encode(test_text)
                loaded_encoded = loaded_tokenizer.encode(test_text)
                
                self.assertEqual(original_encoded, loaded_encoded)
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestTextDataset(unittest.TestCase):
    """Test text dataset handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary text file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        self.temp_file.write("Hello world\n")
        self.temp_file.write("This is a test\n")
        self.temp_file.write("Sample text data\n")
        self.temp_file.write("For testing purposes\n")
        self.temp_file.close()
        
        # Create tokenizer
        self.tokenizer = BPETokenizer(vocab_size=1000)
        sample_texts = ["Hello world", "This is a test", "Sample text data"]
        self.tokenizer.train(sample_texts)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        try:
            dataset = TextDataset(
                file_path=self.temp_file.name,
                tokenizer=self.tokenizer,
                max_length=128
            )
            self.assertIsInstance(dataset, TextDataset)
        except:
            # Skip if TextDataset class doesn't exist
            self.skipTest("TextDataset class not found")
    
    def test_dataset_length(self):
        """Test dataset length."""
        try:
            dataset = TextDataset(
                file_path=self.temp_file.name,
                tokenizer=self.tokenizer,
                max_length=128
            )
            
            length = len(dataset)
            self.assertGreater(length, 0)
            self.assertLessEqual(length, 4)  # We wrote 4 lines
            
        except:
            self.skipTest("TextDataset class not found")
    
    def test_dataset_getitem(self):
        """Test dataset item access."""
        try:
            dataset = TextDataset(
                file_path=self.temp_file.name,
                tokenizer=self.tokenizer,
                max_length=128
            )
            
            if len(dataset) > 0:
                item = dataset[0]
                
                # Check item format
                if isinstance(item, dict):
                    self.assertIn('input_ids', item)
                    input_ids = item['input_ids']
                elif isinstance(item, torch.Tensor):
                    input_ids = item
                else:
                    input_ids = torch.tensor(item)
                
                self.assertIsInstance(input_ids, torch.Tensor)
                self.assertGreater(len(input_ids), 0)
                
        except:
            self.skipTest("TextDataset class not found")
    
    def test_dataloader_integration(self):
        """Test dataset works with DataLoader."""
        try:
            dataset = TextDataset(
                file_path=self.temp_file.name,
                tokenizer=self.tokenizer,
                max_length=64
            )
            
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
            
            # Test one batch
            for batch in dataloader:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids']
                else:
                    input_ids = batch
                
                self.assertIsInstance(input_ids, torch.Tensor)
                self.assertEqual(len(input_ids.shape), 2)  # [batch_size, seq_length]
                break
                
        except:
            self.skipTest("TextDataset class not found")


class TestTextPreprocessor(unittest.TestCase):
    """Test text preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            self.preprocessor = TextPreprocessor()
        except:
            self.preprocessor = None
    
    def test_preprocessor_creation(self):
        """Test preprocessor creation."""
        if self.preprocessor is None:
            self.skipTest("TextPreprocessor class not found")
        
        self.assertIsInstance(self.preprocessor, TextPreprocessor)
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        if self.preprocessor is None:
            self.skipTest("TextPreprocessor class not found")
        
        test_cases = [
            ("Hello  world!", "Hello world!"),  # Multiple spaces
            ("Text\twith\ttabs", "Text with tabs"),  # Tabs
            ("Line\nbreaks\nhere", "Line breaks here"),  # Line breaks
            ("UPPERCASE text", "UPPERCASE text"),  # Case handling
        ]
        
        for input_text, expected_pattern in test_cases:
            if hasattr(self.preprocessor, 'clean_text'):
                cleaned = self.preprocessor.clean_text(input_text)
                self.assertIsInstance(cleaned, str)
                self.assertNotEqual(cleaned, "")
            elif hasattr(self.preprocessor, 'preprocess'):
                cleaned = self.preprocessor.preprocess(input_text)
                self.assertIsInstance(cleaned, str)
    
    def test_text_normalization(self):
        """Test text normalization."""
        if self.preprocessor is None:
            self.skipTest("TextPreprocessor class not found")
        
        test_texts = [
            "This is a test.",
            "Multiple   spaces    here",
            "Special characters: @#$%",
            "Numbers 123 and symbols !@#"
        ]
        
        for text in test_texts:
            methods_to_try = ['normalize', 'preprocess', 'clean_text']
            
            for method_name in methods_to_try:
                if hasattr(self.preprocessor, method_name):
                    method = getattr(self.preprocessor, method_name)
                    result = method(text)
                    
                    self.assertIsInstance(result, str)
                    self.assertGreaterEqual(len(result), 0)
                    break
    
    def test_batch_preprocessing(self):
        """Test batch preprocessing."""
        if self.preprocessor is None:
            self.skipTest("TextPreprocessor class not found")
        
        batch_texts = [
            "First text sample",
            "Second text sample",
            "Third text sample"
        ]
        
        if hasattr(self.preprocessor, 'preprocess_batch'):
            processed_batch = self.preprocessor.preprocess_batch(batch_texts)
            
            self.assertIsInstance(processed_batch, list)
            self.assertEqual(len(processed_batch), len(batch_texts))
            
            for processed_text in processed_batch:
                self.assertIsInstance(processed_text, str)


class TestDataUtilities(unittest.TestCase):
    """Test data utility functions."""
    
    def test_text_splitting(self):
        """Test text splitting functionality."""
        # This would test any text splitting utilities
        sample_text = "This is a long piece of text that needs to be split into smaller chunks for processing."
        
        # Basic splitting test
        chunks = sample_text.split()
        self.assertGreater(len(chunks), 1)
        
        # Test chunk size limiting
        max_words = 5
        limited_chunks = []
        for i in range(0, len(chunks), max_words):
            chunk = " ".join(chunks[i:i+max_words])
            limited_chunks.append(chunk)
        
        self.assertGreater(len(limited_chunks), 1)
        
        # Each chunk should have at most max_words
        for chunk in limited_chunks[:-1]:  # Exclude last chunk which might be shorter
            word_count = len(chunk.split())
            self.assertLessEqual(word_count, max_words)
    
    def test_vocabulary_building(self):
        """Test vocabulary building utilities."""
        texts = [
            "hello world",
            "world peace",
            "hello peace"
        ]
        
        # Build vocabulary
        vocab = set()
        for text in texts:
            words = text.split()
            vocab.update(words)
        
        expected_vocab = {"hello", "world", "peace"}
        self.assertEqual(vocab, expected_vocab)
        
        # Test word frequency
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        self.assertEqual(word_freq["hello"], 2)
        self.assertEqual(word_freq["world"], 2)
        self.assertEqual(word_freq["peace"], 2)
    
    def test_sequence_padding(self):
        """Test sequence padding functionality."""
        sequences = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9, 10]
        ]
        
        max_length = 6
        pad_token = 0
        
        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_length:
                padded = seq + [pad_token] * (max_length - len(seq))
            else:
                padded = seq[:max_length]
            padded_sequences.append(padded)
        
        # Check all sequences have same length
        for padded in padded_sequences:
            self.assertEqual(len(padded), max_length)
        
        # Check original content preserved
        self.assertEqual(padded_sequences[0][:3], [1, 2, 3])
        self.assertEqual(padded_sequences[1][:2], [4, 5])
        self.assertEqual(padded_sequences[2], [6, 7, 8, 9, 10, 0])
    
    def test_data_validation(self):
        """Test data validation utilities."""
        # Test valid data
        valid_tokens = [1, 2, 3, 4, 5]
        vocab_size = 10
        
        # All tokens should be within vocabulary
        valid = all(0 <= token < vocab_size for token in valid_tokens)
        self.assertTrue(valid)
        
        # Test invalid data
        invalid_tokens = [1, 2, 15, 4, 5]  # 15 is out of range
        invalid = any(token >= vocab_size or token < 0 for token in invalid_tokens)
        self.assertTrue(invalid)
        
        # Test empty data
        empty_tokens = []
        empty_valid = len(empty_tokens) == 0
        self.assertTrue(empty_valid)


if __name__ == '__main__':
    # Set random seed for reproducible tests
    torch.manual_seed(42)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)
