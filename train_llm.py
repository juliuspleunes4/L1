#!/usr/bin/env python3
"""
Quick training script for L1 LLM.
This script will train the L1 model and save all artifacts to the models/ directory.

All LLM files (model weights, tokenizer, config) will be in models/l1-v1/
The code stays separate in src/
"""

import os
import subprocess
import sys

def main():
    print("=" * 60)
    print("🚀 L1 LLM Training Script")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('configs/train_config.yaml'):
        print("❌ Error: Please run this script from the L1 project root directory")
        sys.exit(1)
    
    # Check if data is prepared
    if not os.path.exists('data/processed/train.txt'):
        print("📊 Preparing data first...")
        try:
            subprocess.run([sys.executable, 'scripts/prepare_data.py'], check=True)
        except subprocess.CalledProcessError:
            print("❌ Error: Failed to prepare data")
            sys.exit(1)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("🔥 Starting L1 LLM training...")
    print("📁 Model artifacts will be saved to: models/l1-v1/")
    print("📊 Training data: data/processed/")
    print("⚙️  Source code: src/")
    print()
    
    # Run training
    try:
        cmd = [sys.executable, 'scripts/train.py', '--config', 'configs/train_config.yaml']
        print(f"Running: {' '.join(cmd)}")
        print("-" * 60)
        subprocess.run(cmd, check=True)
        
        print("-" * 60)
        print("🎉 Training completed successfully!")
        print()
        print("📁 Your trained LLM is now available in:")
        print("   models/l1-v1/")
        print("   ├── pytorch_model.bin      (model weights)")
        print("   ├── config.json            (model config)")
        print("   ├── tokenizer.json         (tokenizer)")
        print("   ├── training_args.json     (training config)")
        print("   └── checkpoint-*/          (training checkpoints)")
        print()
        print("🚀 To generate text with your model:")
        print("   python scripts/generate.py --model_path models/l1-v1/")
        
    except subprocess.CalledProcessError:
        print("❌ Training failed. Check the output above for errors.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user.")
        sys.exit(1)

if __name__ == '__main__':
    main()
