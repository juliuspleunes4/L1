#!/usr/bin/env python3
"""
Resume training script - identical to train_gpu_compatible.py but shows resume info clearly.
"""

# Just import and run the main function from the updated training script
from train_gpu_compatible import main

if __name__ == "__main__":
    print("🔄 Resume Training Script")
    print("=" * 50)
    print("This script will:")
    print("✅ Check for existing checkpoints")
    print("✅ Resume from the latest checkpoint if found")
    print("✅ Start fresh training if no checkpoint exists")
    print("=" * 50)
    
    main()
