"""
@file       : compare_tokenizer_sizes.py
@package    : scripts
@author     : J.J.G. Pleunes
@date       : 12/2024
@brief      : Train and compare tokenizers with different vocabulary sizes
@details    : Trains 50K, 75K, and 100K tokenizers and generates comparison report
@version    : 1.0
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import BPETokenizer


def load_training_data(data_path: str, max_chars: int = 100_000_000) -> List[str]:
    """Load training data from file."""
    print(f"\nLoading data from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read(max_chars)
    
    # Split into chunks
    if '\n\n' in content:
        chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
    else:
        chunks = [line.strip() for line in content.split('\n') if line.strip()]
    
    print(f"  - Loaded {len(chunks):,} chunks")
    print(f"  - Total chars: {len(content):,}")
    
    return chunks


def train_and_evaluate(vocab_size: int, texts: List[str], test_texts: List[str], min_freq: int = 3) -> Dict:
    """Train tokenizer and return metrics."""
    print(f"\n{'='*60}")
    print(f"Training {vocab_size:,} Token Vocabulary")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Train with minimum frequency filter for speed
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts, min_frequency=min_freq)
    
    training_time = time.time() - start_time
    
    # Calculate metrics on test set
    total_chars = 0
    total_tokens = 0
    total_words = 0
    
    for text in test_texts[:1000]:
        total_chars += len(text)
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
        total_words += len(text.split())
    
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    fertility = total_tokens / total_words if total_words > 0 else 0
    
    # Encoding speed test
    speed_test_text = " ".join(test_texts[:100])
    speed_start = time.time()
    for _ in range(10):
        tokenizer.encode(speed_test_text)
    speed_time = (time.time() - speed_start) / 10
    
    metrics = {
        'vocab_size': vocab_size,
        'actual_vocab_size': len(tokenizer.vocab),
        'training_time': training_time,
        'compression_ratio': compression_ratio,
        'fertility_score': fertility,
        'encoding_speed_ms': speed_time * 1000,
        'chars_per_second': len(speed_test_text) / speed_time
    }
    
    print(f"\nResults:")
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Actual Vocab Size: {len(tokenizer.vocab):,}")
    print(f"  Compression Ratio: {compression_ratio:.2f} chars/token")
    print(f"  Fertility Score: {fertility:.2f} tokens/word")
    print(f"  Encoding Speed: {speed_time*1000:.2f}ms ({metrics['chars_per_second']:.0f} chars/s)")
    
    return tokenizer, metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train and compare tokenizer vocabulary sizes')
    parser.add_argument('--sizes', nargs='+', type=int, default=[50000, 75000, 100000],
                       help='Vocabulary sizes to train (default: 50000 75000 100000)')
    parser.add_argument('--max-samples', type=int, default=20000,
                       help='Maximum training samples (default: 20000)')
    parser.add_argument('--min-freq', type=int, default=3,
                       help='Minimum word frequency (default: 3, higher = faster training)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Tokenizer Vocabulary Size Comparison")
    print(f"{'='*60}")
    print(f"Vocab sizes to train: {args.sizes}")
    print(f"Max samples: {args.max_samples:,}")
    
    # Load training data
    data_path = Path("data/raw/combined_dataset.txt")
    
    if not data_path.exists():
        print(f"\nError: {data_path} not found!")
        print("Please ensure training data exists.")
        return 1
    
    texts = load_training_data(str(data_path), max_chars=10_000_000)
    
    # Limit samples if requested
    if len(texts) > args.max_samples:
        print(f"Limiting to {args.max_samples:,} samples (from {len(texts):,})")
        texts = texts[:args.max_samples]
    
    # Split train/test
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    test_texts = texts[split_idx:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_texts):,} samples")
    print(f"  Test: {len(test_texts):,} samples")
    
    # Train different sizes
    results = []
    tokenizers = {}
    
    for vocab_size in args.sizes:
        tokenizer, metrics = train_and_evaluate(vocab_size, train_texts, test_texts, min_freq=args.min_freq)
        results.append(metrics)
        tokenizers[vocab_size] = tokenizer
        
        # Save tokenizer
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"tokenizer_{vocab_size}.json"
        tokenizer.save(str(output_path))
        print(f"  ✓ Saved to: {output_path}")
    
    # Generate comparison report
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"{'Metric':<25} {'50K':>12} {'75K':>12} {'100K':>12}")
    print(f"{'-'*61}")
    
    metrics_to_compare = [
        ('Actual Vocab Size', 'actual_vocab_size', '{:,}'),
        ('Training Time (s)', 'training_time', '{:.2f}'),
        ('Compression Ratio', 'compression_ratio', '{:.2f}'),
        ('Fertility Score', 'fertility_score', '{:.2f}'),
        ('Encoding Speed (ms)', 'encoding_speed_ms', '{:.2f}'),
        ('Chars/Second', 'chars_per_second', '{:.0f}'),
    ]
    
    for metric_name, metric_key, fmt in metrics_to_compare:
        values = [fmt.format(r[metric_key]) for r in results]
        # Pad with empty strings if fewer than 3 results
        while len(values) < 3:
            values.append('-')
        print(f"{metric_name:<25} {values[0]:>12} {values[1]:>12} {values[2]:>12}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    best_compression = max(results, key=lambda r: r['compression_ratio'])
    best_speed = min(results, key=lambda r: r['encoding_speed_ms'])
    
    print(f"Best Compression: {best_compression['vocab_size']:,} tokens")
    print(f"  → {best_compression['compression_ratio']:.2f} chars/token")
    print(f"\nFastest Encoding: {best_speed['vocab_size']:,} tokens")
    print(f"  → {best_speed['encoding_speed_ms']:.2f}ms per encode")
    
    # Balanced recommendation
    print(f"\n💡 Recommended: 50K tokens")
    print(f"   - Good balance of compression and speed")
    print(f"   - Industry standard (GPT-2/GPT-3 use ~50K)")
    print(f"   - Smaller embedding layer = faster training")
    print(f"   - Use 75K-100K only if targeting multilingual or specialized domains")
    
    # Save comparison report
    report_path = Path("data/processed/tokenizer_comparison.json")
    with open(report_path, 'w') as f:
        json.dump({
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'train_samples': len(train_texts),
            'test_samples': len(test_texts)
        }, f, indent=2)
    
    print(f"\n✓ Comparison report saved to: {report_path}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
