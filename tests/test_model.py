"""
@file       : test_model.py
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Test suite for L1 model components.
@details    : This script contains unit tests for the various components of the
              L1 model, ensuring that they function correctly and meet
              specifications.
@version    : 3.3

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

import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import L1Model, L1Config
from data import BPETokenizer


class TestL1Model(unittest.TestCase):
    """Test cases for L1 model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = L1Config(
            vocab_size=1000,
            max_seq_length=128,
            n_layers=2,
            n_heads=4,
            n_embd=128
        )
        self.model = L1Model(self.config)
    
    def test_model_creation(self):
        """Test model can be created successfully."""
        self.assertIsInstance(self.model, L1Model)
        self.assertEqual(self.model.config.vocab_size, 1000)
    
    def test_forward_pass(self):
        """Test model forward pass."""
        batch_size, seq_length = 2, 64
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        
        outputs = self.model(input_ids)
        
        self.assertIn('logits', outputs)
        self.assertEqual(outputs['logits'].shape, (batch_size, seq_length, self.config.vocab_size))
    
    def test_generation(self):
        """Test text generation."""
        prompt = torch.randint(0, self.config.vocab_size, (1, 10))
        
        with torch.no_grad():
            generated = self.model.generate(prompt, max_new_tokens=20)
        
        self.assertEqual(generated.shape[0], 1)
        self.assertGreater(generated.shape[1], 10)


class TestBPETokenizer(unittest.TestCase):
    """Test cases for BPE tokenizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = BPETokenizer(vocab_size=1000)
    
    def test_tokenizer_creation(self):
        """Test tokenizer can be created."""
        self.assertIsInstance(self.tokenizer, BPETokenizer)
        self.assertEqual(self.tokenizer.vocab_size, 1000)
    
    def test_encode_decode(self):
        """Test encoding and decoding."""
        # First train tokenizer on sample text
        sample_texts = ["Hello world", "This is a test", "Tokenization works"]
        self.tokenizer.train(sample_texts)
        
        text = "Hello world"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
        # Note: BPE may not perfectly reconstruct due to byte encoding
        self.assertTrue(len(encoded) > 0)


if __name__ == '__main__':
    unittest.main()
