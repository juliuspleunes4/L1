# L1 Trained Models

This directory contains your trained L1 language models. Each model version is stored in its own subdirectory with comprehensive checkpointing and automatic resume capability.

## Directory Structure

```
models/
├── l1-gpu-compatible/         # Current active training (RTX 5060 Ti optimized)
│   ├── latest_checkpoint.pt   # Most recent checkpoint (auto-resume)
│   ├── best_checkpoint.pt     # Best loss checkpoint
│   ├── checkpoint_epoch_*_step_*.pt  # Regular checkpoints (auto-cleanup)
│   ├── tokenizer.json         # Tokenizer vocabulary and rules
│   ├── training.log           # Detailed training logs
│   └── config.json            # Model architecture configuration
├── l1-gpu-v1/                 # Previous GPU training version
│   ├── tokenizer.json         # Tokenizer from training
│   └── training.log           # Training history
├── l1-v1/                     # First CPU-based model (legacy)
│   ├── config.json            # Model architecture configuration
│   ├── tokenizer.json         # Tokenizer vocabulary and rules
│   ├── training_args.json     # Training configuration used
│   └── training.log           # Training history
└── README.md                  # This file
```

## Using Your Trained Models

### Generate Text (Current Method)
```bash
# Generate with current model using the simple generation script
python generate_simple.py --prompt "The future of AI is"

# Generate with specific model checkpoint
python generate_simple.py \
    --prompt "The future of artificial intelligence" \
    --max_tokens 100 \
    --temperature 0.8 \
    --model_path models/l1-gpu-compatible/best_checkpoint.pt

# Quick generation with default settings
python generate_simple.py --prompt "Once upon a time"
```

### Resume Training
```bash
# Training automatically resumes from latest checkpoint
python train_gpu_compatible.py

# Monitor training progress
tail -f models/l1-gpu-compatible/training.log
```

### Model Information

Each model directory contains:

- **latest_checkpoint.pt**: Most recent training state (used for automatic resume)
- **best_checkpoint.pt**: Checkpoint with the lowest loss (best performance)
- **checkpoint_epoch_X_step_Y.pt**: Regular training checkpoints (saved every 100 steps)
- **tokenizer.json**: Vocabulary and tokenization rules 
- **training.log**: Comprehensive training history and metrics
- **config.json**: Model architecture and configuration parameters (when available)

### Checkpoint Management

L1 uses advanced checkpointing:
- **Auto-save**: Every 100 steps (~18 minutes)
- **Auto-cleanup**: Keeps only 5 most recent regular checkpoints
- **Auto-resume**: Seamlessly continue training from interruptions
- **Best tracking**: Automatically saves best-performing model

## Generation Parameters

- **temperature**: Controls randomness (0.1 = conservative, 1.0 = creative)
- **max_tokens**: Maximum number of tokens to generate (default: 50)
- **model_path**: Path to specific checkpoint (defaults to best available)

## Model Versions

### l1-gpu-compatible (Current Active)
- **Architecture**: 12 layers, 16 heads, 1024 embedding dimension
- **Parameters**: ~155.8M parameters
- **Vocabulary Size**: ~20,000 tokens (BPE)
- **Training Data**: Wikipedia Simple English (90,000+ samples)
- **Optimization**: Mixed precision, gradient checkpointing, model compilation
- **Hardware**: Optimized for RTX 5060 Ti (16GB VRAM)
- **Status**: 🔄 Active training with automatic checkpointing
- **Use Case**: Production-ready model with excellent performance

### l1-gpu-v1 (Previous Version)
- **Architecture**: Earlier GPU training attempt
- **Status**: ✅ Completed training
- **Use Case**: Previous iteration, superseded by l1-gpu-compatible

### l1-v1 (Legacy)
- **Architecture**: 6 layers, 8 heads, 512 embedding dimension  
- **Parameters**: ~25M parameters
- **Vocabulary Size**: ~1000 tokens
- **Training Data**: Sample text data
- **Hardware**: CPU-based training
- **Status**: ✅ Proof of concept completed
- **Use Case**: Initial learning and experimentation

## Training Progress

Monitor your current training:
```bash
# View real-time training metrics
tail -f models/l1-gpu-compatible/training.log

# Check GPU utilization
nvidia-smi

# View checkpoint history
ls -la models/l1-gpu-compatible/checkpoint_*.pt
```

## Performance Notes

**Current model (l1-gpu-compatible):**
- Excellent loss reduction and convergence
- Advanced checkpointing every 100 steps
- Memory-optimized for 16GB VRAM
- Automatic resume capability
- Mixed precision training for 2x speed improvement
