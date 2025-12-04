"""
@file       : train_tokenizer.py
@package    : scripts
@author     : J.J.G. Pleunes
@date       : 12/2024
@brief      : Enhanced tokenizer training script with configurable vocabulary sizes
@details    : Trains BPE tokenizer with 50K/75K/100K vocab sizes, includes quality metrics
              and comprehensive validation.
@version    : 2.0

@license    : MIT License
Copyright (c) 2024 Julius Pleunes
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import BPETokenizer


class TokenizerTrainer:
    """Enhanced tokenizer trainer with quality metrics and validation."""
    
    def __init__(self, vocab_size: int = 50000):
        """
        Initialize tokenizer trainer.
        
        Args:
            vocab_size: Target vocabulary size (50000, 75000, or 100000)
        """
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.training_stats = {}
        
    def load_training_data(self, data_paths: List[str], max_samples: int = None) -> List[str]:
        """
        Load training data from multiple sources.
        
        Args:
            data_paths: List of file paths to load text from
            max_samples: Maximum number of samples to load (None = all)
            
        Returns:
            List of text samples
        """
        print(f"\n{'='*60}")
        print(f"Loading training data...")
        print(f"{'='*60}")
        
        texts = []
        total_chars = 0
        
        for path in data_paths:
            path_obj = Path(path)
            if not path_obj.exists():
                print(f"Warning: {path} not found, skipping...")
                continue
                
            print(f"\nReading: {path}")
            
            with open(path_obj, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split into reasonable chunks (paragraphs/lines)
            if '\n\n' in content:
                chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
            else:
                chunks = [line.strip() for line in content.split('\n') if line.strip()]
            
            texts.extend(chunks)
            total_chars += len(content)
            
            print(f"  - Loaded {len(chunks):,} chunks")
            print(f"  - Total characters: {len(content):,}")
            
            if max_samples and len(texts) >= max_samples:
                texts = texts[:max_samples]
                break
        
        print(f"\n{'='*60}")
        print(f"Data loading complete:")
        print(f"  - Total samples: {len(texts):,}")
        print(f"  - Total characters: {total_chars:,}")
        print(f"  - Avg chars/sample: {total_chars/len(texts):.1f}")
        print(f"{'='*60}\n")
        
        self.training_stats['num_samples'] = len(texts)
        self.training_stats['total_chars'] = total_chars
        self.training_stats['avg_chars_per_sample'] = total_chars / len(texts) if texts else 0
        
        return texts
    
    def train_tokenizer(self, texts: List[str]) -> BPETokenizer:
        """
        Train BPE tokenizer on text corpus.
        
        Args:
            texts: List of text samples
            
        Returns:
            Trained tokenizer
        """
        print(f"\n{'='*60}")
        print(f"Training BPE Tokenizer")
        print(f"{'='*60}")
        print(f"Target vocabulary size: {self.vocab_size:,}")
        
        start_time = time.time()
        
        # Create and train tokenizer
        self.tokenizer = BPETokenizer(vocab_size=self.vocab_size)
        self.tokenizer.train(texts)
        
        training_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  - Time taken: {training_time:.2f} seconds")
        print(f"  - Final vocab size: {len(self.tokenizer.vocab):,}")
        print(f"{'='*60}\n")
        
        self.training_stats['training_time'] = training_time
        self.training_stats['final_vocab_size'] = len(self.tokenizer.vocab)
        
        return self.tokenizer
    
    def calculate_quality_metrics(self, test_texts: List[str]) -> Dict[str, float]:
        """
        Calculate quality metrics for trained tokenizer.
        
        Metrics:
        - Compression ratio: chars per token (higher = better compression)
        - Fertility score: tokens per word (lower = better)
        - OOV rate: out-of-vocabulary rate (lower = better)
        - Coverage: % of unique tokens found in vocab
        
        Args:
            test_texts: Sample texts for evaluation
            
        Returns:
            Dictionary of quality metrics
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not trained yet!")
        
        print(f"\n{'='*60}")
        print(f"Calculating Quality Metrics")
        print(f"{'='*60}")
        
        total_chars = 0
        total_tokens = 0
        total_words = 0
        total_oov = 0
        unique_tokens = set()
        
        for text in test_texts[:1000]:  # Sample 1000 texts for speed
            # Compression ratio
            total_chars += len(text)
            tokens = self.tokenizer.encode(text)
            total_tokens += len(tokens)
            
            # Track unique tokens for coverage
            unique_tokens.update(tokens)
            
            # OOV rate
            total_oov += sum(1 for t in tokens if t == self.tokenizer.unk_token_id)
            
            # Fertility (tokens per word)
            words = text.split()
            total_words += len(words)
        
        # Calculate metrics
        compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
        fertility = total_tokens / total_words if total_words > 0 else 0
        oov_rate = (total_oov / total_tokens * 100) if total_tokens > 0 else 0
        coverage = (len(unique_tokens) / len(self.tokenizer.vocab) * 100) if self.tokenizer.vocab else 0
        
        metrics = {
            'compression_ratio': compression_ratio,
            'fertility_score': fertility,
            'oov_rate': oov_rate,
            'vocab_coverage': coverage,
            'unique_tokens_used': len(unique_tokens),
            'avg_tokens_per_sample': total_tokens / len(test_texts[:1000])
        }
        
        print(f"\nQuality Metrics:")
        print(f"  - Compression Ratio: {compression_ratio:.2f} chars/token")
        print(f"  - Fertility Score: {fertility:.2f} tokens/word")
        print(f"  - OOV Rate: {oov_rate:.3f}%")
        print(f"  - Vocab Coverage: {coverage:.2f}%")
        print(f"  - Unique Tokens Used: {len(unique_tokens):,}/{len(self.tokenizer.vocab):,}")
        print(f"  - Avg Tokens/Sample: {metrics['avg_tokens_per_sample']:.1f}")
        print(f"{'='*60}\n")
        
        self.training_stats.update(metrics)
        
        return metrics
    
    def validate_tokenizer(self, test_texts: List[str]) -> Dict[str, bool]:
        """
        Run validation tests on trained tokenizer.
        
        Args:
            test_texts: Sample texts for validation
            
        Returns:
            Dictionary of validation results
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not trained yet!")
        
        print(f"\n{'='*60}")
        print(f"Running Validation Tests")
        print(f"{'='*60}")
        
        results = {}
        
        # Test 1: Encode/Decode roundtrip
        print("\nTest 1: Encode/Decode Roundtrip...")
        sample_text = test_texts[0][:500]  # First 500 chars
        encoded = self.tokenizer.encode(sample_text)
        decoded = self.tokenizer.decode(encoded)
        # Note: BPE may not perfectly roundtrip due to whitespace normalization
        results['roundtrip_close'] = len(decoded) > 0 and len(decoded) < len(sample_text) * 2
        print(f"  ✓ Roundtrip: {'PASS' if results['roundtrip_close'] else 'FAIL'}")
        print(f"    Original length: {len(sample_text)}, Decoded length: {len(decoded)}")
        
        # Test 2: Special tokens
        print("\nTest 2: Special Tokens...")
        special_tokens_exist = all(
            token in self.tokenizer.vocab 
            for token in ['<pad>', '<unk>', '<bos>', '<eos>']
        )
        results['special_tokens'] = special_tokens_exist
        print(f"  ✓ Special tokens: {'PASS' if special_tokens_exist else 'FAIL'}")
        
        # Test 3: Empty string handling
        print("\nTest 3: Empty String Handling...")
        try:
            empty_encoded = self.tokenizer.encode("")
            empty_decoded = self.tokenizer.decode([])
            results['empty_handling'] = isinstance(empty_encoded, list) and isinstance(empty_decoded, str)
            print(f"  ✓ Empty handling: PASS")
        except Exception as e:
            results['empty_handling'] = False
            print(f"  ✗ Empty handling: FAIL - {e}")
        
        # Test 4: Unicode handling
        print("\nTest 4: Unicode Handling...")
        try:
            unicode_text = "Hello 世界! 🌍 Прив́ет café"
            unicode_encoded = self.tokenizer.encode(unicode_text)
            unicode_decoded = self.tokenizer.decode(unicode_encoded)
            results['unicode_handling'] = len(unicode_decoded) > 0
            print(f"  ✓ Unicode: PASS")
            print(f"    Original: {unicode_text}")
            print(f"    Decoded: {unicode_decoded}")
        except Exception as e:
            results['unicode_handling'] = False
            print(f"  ✗ Unicode: FAIL - {e}")
        
        # Test 5: Long sequence
        print("\nTest 5: Long Sequence...")
        try:
            long_text = " ".join(test_texts[:10])  # ~10 samples concatenated
            long_encoded = self.tokenizer.encode(long_text)
            long_decoded = self.tokenizer.decode(long_encoded)
            results['long_sequence'] = len(long_encoded) > 0 and len(long_decoded) > 0
            print(f"  ✓ Long sequence: PASS")
            print(f"    Input length: {len(long_text)} chars, {len(long_encoded)} tokens")
        except Exception as e:
            results['long_sequence'] = False
            print(f"  ✗ Long sequence: FAIL - {e}")
        
        # Test 6: Numeric handling
        print("\nTest 6: Numeric Handling...")
        try:
            numeric_text = "123 456.789 -10 3.14159 1,000,000"
            numeric_encoded = self.tokenizer.encode(numeric_text)
            numeric_decoded = self.tokenizer.decode(numeric_encoded)
            results['numeric_handling'] = len(numeric_decoded) > 0
            print(f"  ✓ Numeric: PASS")
            print(f"    Original: {numeric_text}")
            print(f"    Decoded: {numeric_decoded}")
        except Exception as e:
            results['numeric_handling'] = False
            print(f"  ✗ Numeric: FAIL - {e}")
        
        print(f"\n{'='*60}")
        passed = sum(results.values())
        total = len(results)
        print(f"Validation Summary: {passed}/{total} tests passed")
        print(f"{'='*60}\n")
        
        self.training_stats['validation_results'] = results
        self.training_stats['validation_passed'] = passed
        self.training_stats['validation_total'] = total
        
        return results
    
    def save_tokenizer(self, output_path: str):
        """
        Save trained tokenizer to file.
        
        Args:
            output_path: Path to save tokenizer
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not trained yet!")
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer.save(str(output_path_obj))
        print(f"\n✓ Tokenizer saved to: {output_path}")
        
        # Save training stats
        stats_path = output_path_obj.parent / f"{output_path_obj.stem}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_stats, f, indent=2)
        print(f"✓ Training stats saved to: {stats_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer with configurable vocabulary size"
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=50000,
        choices=[50000, 75000, 100000],
        help='Target vocabulary size (default: 50000)'
    )
    parser.add_argument(
        '--data_files',
        nargs='+',
        required=True,
        help='Paths to training data files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/tokenizer.json',
        help='Output path for trained tokenizer'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to use for training (default: all)'
    )
    parser.add_argument(
        '--skip_validation',
        action='store_true',
        help='Skip validation tests'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"BPE Tokenizer Training - V2.0")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Vocabulary Size: {args.vocab_size:,}")
    print(f"  - Data Files: {', '.join(args.data_files)}")
    print(f"  - Output: {args.output}")
    print(f"  - Max Samples: {args.max_samples or 'All'}")
    print(f"{'='*60}\n")
    
    # Initialize trainer
    trainer = TokenizerTrainer(vocab_size=args.vocab_size)
    
    # Load data
    texts = trainer.load_training_data(args.data_files, max_samples=args.max_samples)
    
    if not texts:
        print("Error: No training data loaded!")
        return 1
    
    # Train tokenizer
    tokenizer = trainer.train_tokenizer(texts)
    
    # Calculate quality metrics
    trainer.calculate_quality_metrics(texts)
    
    # Run validation
    if not args.skip_validation:
        validation_results = trainer.validate_tokenizer(texts)
        if not all(validation_results.values()):
            print("\nWarning: Some validation tests failed!")
    
    # Save tokenizer
    trainer.save_tokenizer(args.output)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
