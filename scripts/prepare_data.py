#!/usr/bin/env python3
"""
Data preparation script for L1 model.

Usage:
    python scripts/prepare_data.py --input data/raw/text.txt --output data/processed/
"""

import os
import sys
import argparse
import random
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import BPETokenizer


def load_text_file(file_path: str) -> List[str]:
    """Load text data from file."""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def split_data(texts: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1) -> tuple:
    """Split data into train/validation/test sets."""
    random.shuffle(texts)
    
    n_total = len(texts)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_texts = texts[:n_train]
    val_texts = texts[n_train:n_train + n_val]
    test_texts = texts[n_train + n_val:]
    
    return train_texts, val_texts, test_texts


def save_texts(texts: List[str], file_path: str):
    """Save texts to file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')


def train_tokenizer(texts: List[str], vocab_size: int = 50257) -> BPETokenizer:
    """Train BPE tokenizer on texts."""
    print(f"Training tokenizer with vocab_size={vocab_size}...")
    
    # Use subset for faster training during development
    if len(texts) > 10000:
        sample_texts = random.sample(texts, 10000)
        print(f"Using sample of {len(sample_texts)} texts for tokenizer training")
    else:
        sample_texts = texts
    
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(sample_texts)
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description='Prepare data for L1 training')
    parser.add_argument('--input', type=str, required=True,
                       help='Input text file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--vocab-size', type=int, default=50257,
                       help='Tokenizer vocabulary size')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation data ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for data splitting')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load input data
    print(f"Loading data from {args.input}...")
    texts = load_text_file(args.input)
    print(f"Loaded {len(texts)} text examples")
    
    # Split data
    print("Splitting data...")
    train_texts, val_texts, test_texts = split_data(
        texts, args.train_ratio, args.val_ratio
    )
    
    print(f"Train: {len(train_texts)} examples")
    print(f"Validation: {len(val_texts)} examples") 
    print(f"Test: {len(test_texts)} examples")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save split data
    save_texts(train_texts, os.path.join(args.output, 'train.txt'))
    save_texts(val_texts, os.path.join(args.output, 'val.txt'))
    save_texts(test_texts, os.path.join(args.output, 'test.txt'))
    
    print(f"Data splits saved to {args.output}")
    
    # Train tokenizer
    tokenizer = train_tokenizer(train_texts, args.vocab_size)
    
    # Save tokenizer
    tokenizer_path = os.path.join(args.output, 'tokenizer.json')
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Test tokenizer
    sample_text = train_texts[0] if train_texts else "Hello, world!"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print("\nTokenizer test:")
    print(f"Original: {sample_text}")
    print(f"Encoded: {encoded[:20]}..." if len(encoded) > 20 else f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    print("\nData preparation completed!")


if __name__ == '__main__':
    main()
