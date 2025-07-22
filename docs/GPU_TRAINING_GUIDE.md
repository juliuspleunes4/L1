# 🚀 L1 GPU Training Guide - RTX 50.. Series Optimized (Also compatible with the 40 Series.)

## ✅ Current GPU Status 
L1 is fully GPU-optimized with sm_120 support!

**Current Setup:**
- ✅ **CUDA 12.8+** compatible
- ✅ **Mixed Precision Training** (AMP) enabled
- ✅ **Model Compilation** (PyTorch 2.0+) 
- ✅ **Smart Checkpointing** every 100 steps (~18 minutes)
- ✅ **Automatic Resume** from interruptions
- ✅ **Memory Optimization** for 16GB VRAM

## 🎮 GPU Training Commands

### Primary Training (Recommended)
```bash
# RTX 50.. Series optimized training
python train_gpu_compatible.py

# Monitor progress in real-time
tail -f models/l1-gpu-compatible/training.log
```

### Fallback Training
```bash
# Minimal CPU training (for testing)
python train_minimal.py
```

## 🔧 Current GPU Configuration

Your optimized setup in `configs/train_config_gpu.yaml`:
```yaml
model:
  vocab_size: 1783                # BPE tokenizer vocabulary
  max_seq_length: 1024            # Optimized for 16GB VRAM
  n_layers: 12                    # Production-ready depth
  n_heads: 16                     # Multi-head attention  
  n_embd: 1024                    # 1024-dimensional embeddings
  n_inner: 4096                   # 4x embedding dimension

training:
  batch_size: 8                   # Memory-optimized for RTX 5060 Ti
  mixed_precision: true           # AMP for 2x speed + memory saving
  checkpoint_every_steps: 100     # Save every ~18 minutes
  max_checkpoints_to_keep: 5      # Auto-cleanup old checkpoints
  gradient_accumulation_steps: 4  # Simulate larger batch sizes
```

**Model Stats:**
- **Parameters**: ~155.8M (production-ready size)
- **VRAM Usage**: ~8GB training, 16GB recommended
- **Training Speed**: ~100 steps in 18 minutes

## 📚 Dataset Management

### Current Dataset
Your L1 is currently using **Wikipedia Simple English**:
- **90,000+ samples** of high-quality text
- **BPE tokenization** with 1783 vocabulary size
- **Pre-processed** and ready for training

### Adding New Datasets
```bash
# Use pre-configured datasets
python add_dataset.py --preset beginner       # 50k samples (fast)
python add_dataset.py --preset intermediate   # 150k samples (balanced)  
python add_dataset.py --preset advanced       # 500k samples (comprehensive)

# Add specific datasets
python add_dataset.py wikipedia_full          # 500k Wikipedia samples
python add_dataset.py papers_arxiv            # Scientific papers
python add_dataset.py books_gutenberg         # Literature texts
```

### Custom Kaggle Datasets
```bash
# Method 1: Direct download
import kagglehub
dataset_path = kagglehub.dataset_download("username/dataset-name")
python prepare_large_dataset.py --custom_path dataset_path

# Method 2: Add to datasets.yaml (see README for details)
python add_dataset.py your_custom_dataset
```

## 🔧 Advanced GPU Features

### Current Optimizations
Your L1 includes cutting-edge optimizations:

**Memory Management:**
- **Mixed Precision (AMP)**: FP16 training for 2x memory + speed
- **Gradient Checkpointing**: Memory-efficient large model training
- **CUDA Cache Management**: Automatic GPU memory optimization
- **Pin Memory**: Faster CPU→GPU data transfers

**Training Safety:**
- **Smart Checkpointing**: Save every 100 steps with auto-cleanup
- **Resume Training**: Automatic recovery from interruptions  
- **Best Model Tracking**: Saves best checkpoint based on validation loss
- **Progress Monitoring**: Real-time training metrics and logging

**Performance:**
- **Model Compilation**: PyTorch 2.0+ graph optimization
- **Batch Size Optimization**: 8 for RTX 5060 Ti 16GB
- **Gradient Accumulation**: Simulate larger batch sizes (4 steps)

### Real-time Monitoring
```bash
# Training progress with ETA
🎓 Training Configuration:
   ├── Epochs: 10, Total steps: 11,250
   ├── Checkpoint every: 100 steps (~18 min)  
   ├── Keep checkpoints: 5 latest
   ├── Mixed precision: True
   └── Optimizer: AdamW

# Automatic checkpoint management  
💾 Saving progress checkpoint at step 1100...
🗑️ Cleaned up old checkpoint: checkpoint_epoch_1_step_600.pt
✅ Training resumed from epoch 2, step 1847, loss: 2.1432
```

## 🎯 Training Workflow

### Step 1: Verify GPU Setup
```bash
# Verify CUDA installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Expected output:
# CUDA Available: True  
# GPU: NVIDIA GeForce RTX 5060 Ti
```

### Step 2: Start Training
```bash
# Begin or resume training  
python train_gpu_compatible.py

# Training will automatically:
# ✅ Load latest checkpoint if exists
# ✅ Use mixed precision training
# ✅ Save checkpoints every 100 steps
# ✅ Monitor and log progress
```

### Step 3: Monitor Progress
```bash
# Watch training logs
tail -f models/l1-gpu-compatible/training.log

# Check GPU utilization 
nvidia-smi

# View saved checkpoints
ls models/l1-gpu-compatible/checkpoint_*.pt
```

### Step 4: Generate Text
```bash
# Use latest checkpoint for generation
python generate_simple.py \
    --prompt "The future of artificial intelligence is" \
    --model_path models/l1-gpu-compatible
```

## 💡 Hardware Requirements & Performance

### Minimum Requirements
- **GPU**: RTX 4060 or equivalent (8GB VRAM)
- **RAM**: 16GB system RAM  
- **Storage**: 10GB free space
- **CUDA**: 12.8+ (for RTX 50.. Series)

### Recommended Setup (Current)
- **GPU**: RTX 5060 Ti (16GB VRAM) ✅
- **RAM**: 32GB system RAM
- **Storage**: SSD with 50GB+ free space
- **CUDA**: 12.8+ with PyTorch 2.0+

### Performance Expectations
With RTX 5060 Ti 16GB:
- **Training Speed**: ~100 steps in 18 minutes
- **Model Size**: 155.8M parameters (production-ready)
- **Memory Usage**: ~8GB VRAM during training
- **Checkpoint Frequency**: Every 100 steps (automatic)

### Scaling Options
```yaml
# For more GPU memory, increase model size:
model:
  n_layers: 16          # Larger model (24 layers max)
  n_embd: 1536          # Higher dimension (up to 2048)
  max_seq_length: 2048  # Longer sequences

training:
  batch_size: 12        # Larger batches with more VRAM
```

## 📊 Model Scaling Guidelines

### Model Size vs GPU Memory
| Configuration | Parameters | VRAM | Training Time | Use Case |
|---------------|------------|------|---------------|----------|
| **Current L1** | **155.8M** | **8GB** | **18min/100 steps** | **Production** |
| Small         | 25M        | 4GB  | 8min/100 steps   | Experiments |
| Large         | 340M       | 16GB | 35min/100 steps  | Research |
| XL            | 1B+        | 24GB+| 2hr+/100 steps   | High-end |

### Vocabulary Size Impact
Your current vocab_size of **1783** is small. Consider increasing:
- **8,000-16,000**: Standard for most applications
- **32,000-50,000**: Production-quality models  
- **Tradeoff**: Larger vocab = better text quality but requires retraining

### Dataset Size Recommendations  
- **Current**: 90k samples (Wikipedia Simple) - Good baseline
- **Intermediate**: 150k-500k samples - Better performance
- **Production**: 1M+ samples - High-quality results

## 🚀 Ready for Production

Your L1 is GPU-optimized and production-ready:
✅ **RTX 50.. Series compatibility** 
✅ **155.8M parameter model**
✅ **Smart checkpointing system**
✅ **Mixed precision training**
✅ **Automatic resume capability**
✅ **Memory-optimized for 16GB VRAM**

**Next Steps:**
1. Continue current training to completion
2. Experiment with generation parameters
3. Consider increasing vocabulary size for better quality
4. Scale to larger datasets when ready

Your setup is optimized for the RTX 5060 Ti and ready for serious LLM development!
