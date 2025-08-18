#!/usr/bin/env python3
"""
@file       : demo.py
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Quick demo script to test L1 components.
@details    : This script runs a series of tests to ensure that the L1 model,
              tokenizer, and other components are functioning as expected.
@version    : 3.3
@usage      : python demo.py

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

import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_creation():
    """Test creating L1 model."""
    print("Testing L1 model creation...")
    
    try:
        from src.models import L1Model, L1Config
        
        # Create small config for testing
        config = L1Config(
            vocab_size=1000,
            max_seq_length=128,
            n_layers=2,
            n_heads=4,
            n_embd=128
        )
        
        model = L1Model(config)
        
        print(f"✓ Model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Config: {config.n_layers} layers, {config.n_heads} heads, {config.n_embd} embedding dim")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test model forward pass."""
    print("\nTesting model forward pass...")
    
    try:
        from src.models import L1Model, L1Config
        
        config = L1Config(
            vocab_size=1000,
            max_seq_length=128,
            n_layers=2,
            n_heads=4,
            n_embd=128
        )
        
        model = L1Model(config)
        
        # Create dummy input
        batch_size, seq_length = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {input_ids.shape}")
        print(f"  - Output logits shape: {outputs['logits'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_tokenizer():
    """Test BPE tokenizer."""
    print("\nTesting BPE tokenizer...")
    
    try:
        from src.data import BPETokenizer
        
        # Create tokenizer
        tokenizer = BPETokenizer(vocab_size=1000)
        
        # Train on sample texts
        sample_texts = [
            "Hello world! How are you today?",
            "This is a test of the tokenizer.",
            "Machine learning is fascinating.",
            "Natural language processing with transformers."
        ]
        
        print("  - Training tokenizer...")
        tokenizer.train(sample_texts)
        
        # Test encoding/decoding
        test_text = "Hello world!"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"✓ Tokenizer working")
        print(f"  - Vocabulary size: {len(tokenizer.vocab)}")
        print(f"  - Test text: '{test_text}'")
        print(f"  - Encoded: {encoded}")
        print(f"  - Decoded: '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        return False


def test_data_preparation():
    """Test data preparation script."""
    print("\nTesting data preparation...")
    
    try:
        # Test with sample data
        sample_file = "data/raw/sample_text.txt"
        
        if os.path.exists(sample_file):
            print(f"✓ Sample data file found: {sample_file}")
            
            with open(sample_file, 'r') as f:
                lines = f.readlines()
            
            print(f"  - Sample data contains {len(lines)} lines")
            print(f"  - First line: '{lines[0].strip()}'")
            
            return True
        else:
            print(f"✗ Sample data file not found: {sample_file}")
            return False
            
    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("L1 LLM Project Demo")
    print("=" * 50)
    
    tests = [
        test_model_creation,
        test_forward_pass,
        test_tokenizer,
        test_data_preparation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Demo Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your L1 project is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your training data: python scripts/prepare_data.py --input your_data.txt --output data/processed/")
        print("2. Start training: python scripts/train.py --config configs/base_config.yaml")
        print("3. Generate text: python scripts/generate.py --model checkpoints/best_model.pt --prompt 'Hello, world!'")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    print("=" * 50)


if __name__ == '__main__':
    main()
