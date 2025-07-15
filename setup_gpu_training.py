#!/usr/bin/env python3
"""
Automated setup script for L1 GPU training on new PC.
Downloads datasets, processes them, and starts training.
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    
    # Check PyTorch with CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch with CUDA detected: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("⚠️  PyTorch detected but no CUDA. Install with:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        print("❌ PyTorch not found. Install with:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    # Check other requirements
    required_packages = ['kaggle', 'pandas', 'numpy', 'tqdm', 'pyyaml']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All requirements satisfied")
    return True

def setup_kaggle():
    """Setup Kaggle API credentials"""
    print("\n🔑 Setting up Kaggle API...")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print("✅ Kaggle credentials found")
        return True
    
    print("❌ Kaggle credentials not found")
    print("\n📋 To setup Kaggle:")
    print("1. Go to https://kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Save kaggle.json to:", str(kaggle_json))
    print("4. Or set environment variables KAGGLE_USERNAME and KAGGLE_KEY")
    
    # Check environment variables
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        print("✅ Kaggle environment variables found")
        return True
    
    return False

def download_dataset(dataset_name, dataset_path):
    """Download a dataset from Kaggle"""
    print(f"\n📥 Downloading {dataset_name}...")
    
    # Create datasets directory
    os.makedirs("datasets", exist_ok=True)
    
    # Download dataset
    command = f"kaggle datasets download -d {dataset_path} -p ./datasets/ --unzip"
    return run_command(command, f"Download {dataset_name}")

def process_dataset(file_path, text_column, max_samples, vocab_size):
    """Process dataset for training"""
    print(f"\n⚙️ Processing dataset...")
    
    command = f"python prepare_large_dataset.py \"{file_path}\" --text-column \"{text_column}\" --max-samples {max_samples} --vocab-size {vocab_size}"
    return run_command(command, "Process dataset")

def start_training():
    """Start GPU training"""
    print(f"\n🚀 Starting GPU training...")
    
    command = "python train_gpu.py"
    print(f"Running: {command}")
    print("💡 Training will start in a new window. Check GPU usage with 'nvidia-smi'")
    
    # Start training (don't wait for completion)
    subprocess.Popen(command, shell=True)

def main():
    parser = argparse.ArgumentParser(description="L1 GPU Setup Assistant")
    parser.add_argument("--dataset", default="news", 
                       choices=["news", "wikipedia", "books", "custom"],
                       help="Dataset to download and use")
    parser.add_argument("--samples", type=int, default=100000,
                       help="Maximum samples to process")
    parser.add_argument("--vocab-size", type=int, default=20000,
                       help="Vocabulary size")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip dataset download")
    parser.add_argument("--custom-path", help="Path to custom dataset file")
    parser.add_argument("--custom-column", default="text", help="Text column for custom dataset")
    
    args = parser.parse_args()
    
    print("🎮 L1 GPU Training Setup Assistant")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup failed. Please install missing requirements.")
        return
    
    # Setup Kaggle
    if not args.skip_download and not setup_kaggle():
        print("\n❌ Setup failed. Please configure Kaggle API.")
        return
    
    # Dataset configurations
    datasets = {
        "news": {
            "kaggle_path": "snapcrack/all-the-news",
            "file_path": "datasets/articles1.csv",
            "text_column": "content",
            "description": "All The News (143k articles)"
        },
        "wikipedia": {
            "kaggle_path": "mikeortman/wikipedia-sentences", 
            "file_path": "datasets/wikipedia-sentences.csv",
            "text_column": "sentence",
            "description": "Wikipedia Sentences"
        },
        "books": {
            "kaggle_path": "alexandreparent/gutenberg-database",
            "file_path": "datasets/catalog.csv",
            "text_column": "text",
            "description": "Project Gutenberg Books"
        }
    }
    
    # Download and process dataset
    if args.dataset == "custom":
        if not args.custom_path:
            print("❌ Custom dataset requires --custom-path")
            return
        file_path = args.custom_path
        text_column = args.custom_column
        print(f"📁 Using custom dataset: {file_path}")
    else:
        dataset_config = datasets[args.dataset]
        print(f"📚 Selected dataset: {dataset_config['description']}")
        
        if not args.skip_download:
            # Download dataset
            if not download_dataset(dataset_config['description'], dataset_config['kaggle_path']):
                print("❌ Dataset download failed")
                return
        
        file_path = dataset_config['file_path']
        text_column = dataset_config['text_column']
    
    # Check if dataset file exists
    if not os.path.exists(file_path):
        print(f"❌ Dataset file not found: {file_path}")
        if not args.skip_download:
            print("   Try running with --skip-download if you already have the dataset")
        return
    
    # Process dataset
    if not process_dataset(file_path, text_column, args.samples, args.vocab_size):
        print("❌ Dataset processing failed")
        return
    
    # Check if processed files exist
    required_files = [
        "data/processed/train.txt",
        "data/processed/val.txt", 
        "data/processed/tokenizer.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing processed files: {missing_files}")
        return
    
    print("\n✅ Setup completed successfully!")
    print("\n📊 Ready to train:")
    print(f"   ├── Dataset: {args.samples:,} samples")
    print(f"   ├── Vocabulary: {args.vocab_size:,} tokens")
    print(f"   ├── Model: ~80M+ parameters (GPU config)")
    print(f"   └── Output: models/l1-gpu-v1/")
    
    # Ask if user wants to start training
    response = input("\n🚀 Start training now? (y/n): ").lower().strip()
    if response == 'y':
        start_training()
        print("\n🎯 Training started! Monitor progress with:")
        print("   ├── GPU usage: nvidia-smi")
        print("   ├── Training log: tail -f models/l1-gpu-v1/training.log")
        print("   └── Loss curves: Check the console output")
    else:
        print("\n💡 To start training later, run:")
        print("   python train_gpu.py")

if __name__ == "__main__":
    main()
