#!/usr/bin/env python3
"""
@file       : prepare_large_dataset.py
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Dataset preparation script for large-scale training.
@details    : This script processes datasets in CSV, JSON, and plain text formats,
              cleaning and normalizing text data, creating a character-level vocabulary,
              and saving the processed dataset for training.
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

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Iterator
import re
from tqdm import tqdm

class DatasetProcessor:
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short texts (less than 10 characters)
        if len(text) < 10:
            return ""
            
        return text
    
    def process_csv_dataset(self, csv_path: str, text_column: str, max_samples: int = None) -> List[str]:
        """Process CSV dataset (common Kaggle format)"""
        print(f"Processing CSV dataset: {csv_path}")
        
        texts = []
        chunk_size = 10000
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            if text_column not in chunk.columns:
                raise ValueError(f"Column '{text_column}' not found. Available columns: {list(chunk.columns)}")
            
            for text in chunk[text_column].values:
                cleaned_text = self.clean_text(text)
                if cleaned_text:
                    texts.append(cleaned_text)
                    
                    if max_samples and len(texts) >= max_samples:
                        return texts
                        
            print(f"   Processed {len(texts):,} texts so far...")
        
        return texts
    
    def process_json_dataset(self, json_path: str, text_field: str, max_samples: int = None) -> List[str]:
        """Process JSONL or JSON dataset"""
        print(f"Processing JSON dataset: {json_path}")
        
        texts = []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            if json_path.endswith('.jsonl'):
                # JSONL format
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        if text_field in data:
                            cleaned_text = self.clean_text(data[text_field])
                            if cleaned_text:
                                texts.append(cleaned_text)
                        
                        if max_samples and len(texts) >= max_samples:
                            break
                            
                        if i % 10000 == 0 and i > 0:
                            print(f"   Processed {len(texts):,} texts so far...")
                            
                    except json.JSONDecodeError:
                        continue
            else:
                # Regular JSON format
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if text_field in item:
                            cleaned_text = self.clean_text(item[text_field])
                            if cleaned_text:
                                texts.append(cleaned_text)
                                
                            if max_samples and len(texts) >= max_samples:
                                break
        
        return texts
    
    def process_text_dataset(self, txt_path: str, max_samples: int = None) -> List[str]:
        """Process plain text dataset"""
        print(f"Processing text dataset: {txt_path}")
        
        texts = []
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                cleaned_text = self.clean_text(line)
                if cleaned_text:
                    texts.append(cleaned_text)
                    
                if max_samples and len(texts) >= max_samples:
                    break
                    
                if i % 10000 == 0 and i > 0:
                    print(f"   Processed {len(texts):,} texts so far...")
        
        return texts
    
    def create_vocabulary(self, texts: List[str], vocab_size: int = 10000) -> dict:
        """Create character-level vocabulary from texts"""
        print(f"Creating vocabulary (target size: {vocab_size:,})...")
        
        # Count character frequencies
        char_counts = {}
        for text in tqdm(texts, desc="Counting characters"):
            for char in text.lower():
                char_counts[char] = char_counts.get(char, 0) + 1
        
        # Sort by frequency and take top characters
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        top_chars = [char for char, count in sorted_chars[:vocab_size-4]]  # -4 for special tokens
        
        # Create vocabulary with special tokens
        vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3
        }
        
        for i, char in enumerate(top_chars):
            vocab[char] = i + 4
        
        print(f"Vocabulary created with {len(vocab):,} tokens")
        print(f"   Most common characters: {list(top_chars[:20])}")
        
        return vocab
    
    def save_dataset(self, texts: List[str], vocab: dict, split_ratio: float = 0.9):
        """Save processed dataset and vocabulary"""
        print(f"Saving dataset...")
        
        # Shuffle texts
        np.random.shuffle(texts)
        
        # Split into train/validation
        split_idx = int(len(texts) * split_ratio)
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        # Save training data
        train_path = self.output_dir / "train.txt"
        with open(train_path, 'w', encoding='utf-8') as f:
            for text in train_texts:
                f.write(text + '\n')
        
        # Save validation data
        val_path = self.output_dir / "val.txt"
        with open(val_path, 'w', encoding='utf-8') as f:
            for text in val_texts:
                f.write(text + '\n')
        
        # Save vocabulary
        vocab_data = {
            'vocab': vocab,
            'special_tokens': {
                '<pad>': 0,
                '<unk>': 1,
                '<bos>': 2,
                '<eos>': 3
            }
        }
        
        vocab_path = self.output_dir / "tokenizer.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"Dataset saved:")
        print(f"   - Training: {len(train_texts):,} samples -> {train_path}")
        print(f"   - Validation: {len(val_texts):,} samples -> {val_path}")
        print(f"   - Vocabulary: {len(vocab):,} tokens -> {vocab_path}")

def main():
    parser = argparse.ArgumentParser(description="Process large datasets for L1 training")
    parser.add_argument("input_path", help="Path to input dataset file")
    parser.add_argument("--format", choices=['csv', 'json', 'jsonl', 'txt'], 
                       help="Dataset format (auto-detected if not specified)")
    parser.add_argument("--text-column", default="text", 
                       help="Column name for text data (CSV/JSON)")
    parser.add_argument("--text-field", default="text", 
                       help="Field name for text data (JSON)")
    parser.add_argument("--max-samples", type=int, 
                       help="Maximum number of samples to process")
    parser.add_argument("--vocab-size", type=int, default=10000,
                       help="Vocabulary size")
    parser.add_argument("--output-dir", default="data/processed",
                       help="Output directory")
    parser.add_argument("--split-ratio", type=float, default=0.9,
                       help="Train/validation split ratio")
    
    args = parser.parse_args()
    
    # Auto-detect format if not specified
    if not args.format:
        input_path = Path(args.input_path)
        if input_path.suffix == '.csv':
            args.format = 'csv'
        elif input_path.suffix == '.jsonl':
            args.format = 'jsonl'
        elif input_path.suffix == '.json':
            args.format = 'json'
        elif input_path.suffix == '.txt':
            args.format = 'txt'
        else:
            raise ValueError(f"Could not auto-detect format for {input_path.suffix}")
    
    print("Starting dataset processing...")
    print(f"   Input: {args.input_path}")
    print(f"   Format: {args.format}")
    print(f"   Max samples: {args.max_samples or 'No limit'}")
    print(f"   Vocab size: {args.vocab_size:,}")
    print("="*50)
    
    # Process dataset
    processor = DatasetProcessor(args.output_dir)
    
    if args.format == 'csv':
        texts = processor.process_csv_dataset(args.input_path, args.text_column, args.max_samples)
    elif args.format in ['json', 'jsonl']:
        texts = processor.process_json_dataset(args.input_path, args.text_field, args.max_samples)
    elif args.format == 'txt':
        texts = processor.process_text_dataset(args.input_path, args.max_samples)
    else:
        raise ValueError(f"Unsupported format: {args.format}")
    
    if not texts:
        print("No valid texts found in dataset!")
        return
    
    print(f"\nDataset Statistics:")
    print(f"   - Total samples: {len(texts):,}")
    print(f"   - Average length: {np.mean([len(t) for t in texts]):.1f} characters")
    print(f"   - Min length: {min(len(t) for t in texts):,} characters")
    print(f"   - Max length: {max(len(t) for t in texts):,} characters")
    
    # Create vocabulary
    vocab = processor.create_vocabulary(texts, args.vocab_size)
    
    # Save dataset
    processor.save_dataset(texts, vocab, args.split_ratio)
    
    print(f"\nReady for training!")
    print(f"   Use: python train_gpu.py")

if __name__ == "__main__":
    main()
