#!/usr/bin/env python3
"""
Download and process Wikipedia Simple English dataset for L1 training.
Uses kagglehub for modern Kaggle dataset downloading.
"""

import os
import sys
from pathlib import Path
import argparse

def install_kagglehub():
    """Install kagglehub if not available"""
    try:
        import kagglehub
        return kagglehub
    except ImportError:
        print("📦 Installing kagglehub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub
        return kagglehub

def download_wikipedia_dataset():
    """Download the Wikipedia Simple English dataset"""
    print("🚀 Downloading Wikipedia Simple English dataset...")
    print("=" * 60)
    
    # Install kagglehub if needed
    kagglehub = install_kagglehub()
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("ffatty/plain-text-wikipedia-simpleenglish")
        print(f"✅ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Kaggle account")
        print("   2. Kaggle API credentials configured")
        print("   3. Internet connection")
        return None

def find_text_files(dataset_path):
    """Find text files in the downloaded dataset"""
    dataset_dir = Path(dataset_path)
    
    # Look for common text file patterns
    text_files = []
    for pattern in ["*.txt", "*.json", "*.csv"]:
        text_files.extend(list(dataset_dir.glob(pattern)))
        text_files.extend(list(dataset_dir.glob(f"**/{pattern}")))
    
    print(f"📁 Found {len(text_files)} text files:")
    for file in text_files[:10]:  # Show first 10
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   ├── {file.name} ({size_mb:.1f} MB)")
    
    if len(text_files) > 10:
        print(f"   └── ... and {len(text_files) - 10} more files")
    
    return text_files

def process_wikipedia_dataset(dataset_path, max_samples=100000, vocab_size=20000):
    """Process the Wikipedia dataset for L1 training"""
    print(f"\n⚙️ Processing Wikipedia dataset...")
    
    text_files = find_text_files(dataset_path)
    if not text_files:
        print("❌ No text files found in dataset")
        return False
    
    # Use the largest text file (usually the main one)
    main_file = max(text_files, key=lambda f: f.stat().st_size)
    print(f"📄 Using main file: {main_file.name} ({main_file.stat().st_size / (1024*1024):.1f} MB)")
    
    # Determine file format and process accordingly
    if main_file.suffix == '.txt':
        format_type = 'txt'
        column_arg = ''
    elif main_file.suffix == '.csv':
        format_type = 'csv'
        column_arg = '--text-column "text"'  # Common column name
    elif main_file.suffix == '.json':
        format_type = 'json'
        column_arg = '--text-field "text"'
    else:
        print(f"⚠️  Unknown file format: {main_file.suffix}, trying as text file")
        format_type = 'txt'
        column_arg = ''
    
    # Build processing command
    cmd = f'python prepare_large_dataset.py "{main_file}" --format {format_type} {column_arg} --max-samples {max_samples} --vocab-size {vocab_size}'
    
    print(f"🔧 Processing command: {cmd}")
    print("📊 Processing dataset (this may take a few minutes)...")
    
    # Run the processing
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Dataset processing completed!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Dataset processing failed: {e}")
        print(f"Error output: {e.stderr}")
        
        # Try alternative approaches
        print("\n🔄 Trying alternative processing...")
        
        # If CSV failed, try guessing the column name
        if format_type == 'csv':
            print("   Trying common column names...")
            for col_name in ['content', 'article', 'page', 'wiki_text', 'plain_text']:
                alt_cmd = f'python prepare_large_dataset.py "{main_file}" --format csv --text-column "{col_name}" --max-samples {max_samples} --vocab-size {vocab_size}'
                try:
                    subprocess.run(alt_cmd, shell=True, check=True, capture_output=True)
                    print(f"✅ Success with column: {col_name}")
                    return True
                except:
                    continue
        
        # If all else fails, try as plain text
        print("   Trying as plain text file...")
        simple_cmd = f'python prepare_large_dataset.py "{main_file}" --format txt --max-samples {max_samples} --vocab-size {vocab_size}'
        try:
            subprocess.run(simple_cmd, shell=True, check=True, capture_output=True)
            print("✅ Success processing as plain text!")
            return True
        except:
            print("❌ All processing attempts failed")
            return False

def main():
    parser = argparse.ArgumentParser(description="Download and process Wikipedia dataset for L1")
    parser.add_argument("--max-samples", type=int, default=100000,
                       help="Maximum samples to process (default: 100,000)")
    parser.add_argument("--vocab-size", type=int, default=20000,
                       help="Vocabulary size (default: 20,000)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download if dataset already exists")
    parser.add_argument("--dataset-path", help="Path to existing dataset (if skipping download)")
    
    args = parser.parse_args()
    
    print("🌍 Wikipedia Simple English Dataset Setup")
    print("=" * 60)
    print("📚 Dataset: Simple English Wikipedia articles")
    print(f"🎯 Target samples: {args.max_samples:,}")
    print(f"📝 Vocabulary size: {args.vocab_size:,}")
    print("=" * 60)
    
    # Download dataset
    if args.skip_download and args.dataset_path:
        dataset_path = args.dataset_path
        print(f"📁 Using existing dataset: {dataset_path}")
    else:
        dataset_path = download_wikipedia_dataset()
        if not dataset_path:
            return
    
    # Process dataset
    success = process_wikipedia_dataset(dataset_path, args.max_samples, args.vocab_size)
    
    if success:
        print("\n🎉 Wikipedia dataset setup complete!")
        print("\n📊 Next steps:")
        print("   1. Review processed files in data/processed/")
        print("   2. Start training:")
        print("      python train_gpu.py")
        print("   3. Or use original training:")
        print("      python train_minimal.py")
        
        # Check if processed files exist
        required_files = [
            "data/processed/train.txt",
            "data/processed/val.txt",
            "data/processed/tokenizer.json"
        ]
        
        print(f"\n📄 Generated files:")
        for file_path in required_files:
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   ✅ {file_path} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {file_path} (missing)")
    else:
        print("\n❌ Setup failed!")
        print("\n🔧 Manual processing options:")
        print("   1. Check dataset files manually")
        print("   2. Use prepare_large_dataset.py directly")
        print("   3. Try with different parameters")

if __name__ == "__main__":
    main()
