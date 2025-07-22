# L1 Training Guide

## ✅ Current Training Status

**Your L1 is ready for GPU training with optimized setup:**
- ✅ **Dataset**: Wikipedia Simple English (90,000+ samples) ready
- ✅ **Model**: 155.8M parameters, 12 layers, 16 heads, 1024 embedding
- ✅ **GPU Optimization**: RTX 5060 Ti compatible with mixed precision
- ✅ **Checkpointing**: Smart saves every 100 steps with auto-resume
- ✅ **Training Scripts**: `train_gpu_compatible.py` ready to use

## 🚀 Quick Start Training

### Start Training (Recommended)
```bash
# Begin or resume GPU training
python train_gpu_compatible.py

# Training will automatically:
# ✅ Detect and resume from latest checkpoint
# ✅ Use mixed precision for speed/memory
# ✅ Save checkpoints every 100 steps (~18 minutes)
# ✅ Display real-time progress and metrics
```

### Monitor Training
```bash
# Watch training logs in real-time
tail -f models/l1-gpu-compatible/training.log

# Check GPU utilization
nvidia-smi

# View saved checkpoints
ls models/l1-gpu-compatible/checkpoint_*.pt
```

### Generate Text
```bash
# Test your model during or after training
python generate_simple.py \
    --prompt "The future of artificial intelligence is" \
    --model_path models/l1-gpu-compatible
```

## 📊 Current Training Configuration

Your optimized setup in `configs/train_config_gpu.yaml`:

### Model Architecture
```yaml
model:
  vocab_size: 1783           # BPE tokenizer vocabulary
  max_seq_length: 1024       # Memory-optimized for RTX 5060 Ti
  n_layers: 12               # Production depth
  n_heads: 16                # Multi-head attention
  n_embd: 1024               # High-dimensional embeddings
  n_inner: 4096              # Feed-forward dimension
  dropout: 0.1               # Regularization
```

### Training Settings
```yaml
training:
  num_epochs: 10                  # Complete training cycles
  batch_size: 8                   # Optimized for 16GB VRAM
  learning_rate: 0.0001           # Conservative for stability  
  mixed_precision: true           # AMP for 2x speed + memory
  checkpoint_every_steps: 100     # Save every ~18 minutes
  max_checkpoints_to_keep: 5      # Auto-cleanup old saves
  gradient_accumulation_steps: 4  # Simulate larger batches
```

**Model Stats:**
- **Parameters**: 155.8M (production-ready size)
- **Memory Usage**: ~8GB VRAM during training
- **Training Speed**: ~100 steps per 18 minutes on RTX 5060 Ti

## 🔧 Advanced Training Features

### Smart Checkpointing System
Your L1 includes advanced checkpoint management:

```bash
# Automatic checkpoint saving every 100 steps
💾 Saving progress checkpoint at step 1100...
🗑️ Cleaned up old checkpoint: checkpoint_epoch_1_step_600.pt
✅ Keeping: latest_checkpoint.pt, best_checkpoint.pt, checkpoint_epoch_1_step_1000.pt
```

**Features:**
- **Auto-resume**: Training continues from latest checkpoint automatically
- **Best model tracking**: Saves model with lowest validation loss
- **Storage management**: Keeps only 5 most recent checkpoints
- **Progress preservation**: Never lose training progress

### Mixed Precision Training (AMP)
Enabled by default for RTX 5060 Ti optimization:
- **2x speed improvement** with minimal quality loss
- **50% memory reduction** allowing larger batch sizes
- **Automatic loss scaling** prevents gradient underflow
- **Fallback protection** switches to FP32 if needed

### Memory Optimizations
```python
# Your training includes:
- Gradient checkpointing: Trade compute for memory
- Pin memory: Faster CPU→GPU transfers  
- CUDA cache clearing: Prevent memory leaks
- Model compilation: PyTorch 2.0+ graph optimization
```

## 📈 Training Monitoring

### Real-time Progress Display
```bash
🎓 Training Configuration:
   ├── Epochs: 10, Total steps: 11,250
   ├── Checkpoint every: 100 steps (~18 min)
   ├── Keep checkpoints: 5 latest
   ├── Mixed precision: True
   └── Optimizer: AdamW

📊 Training Progress:
   ├── Epoch 2/10, Step 1847/11250 (16.4%)
   ├── Loss: 2.1432 (↓ from 2.8941)  
   ├── Learning rate: 0.0001
   ├── GPU Memory: 7.8GB/16GB (49%)
   └── Speed: 5.4 steps/sec
```

### Key Metrics to Watch
- **Loss Reduction**: Should decrease steadily from ~4.0 to ~2.0
- **GPU Utilization**: Should stay near 95-100%  
- **Memory Usage**: ~8GB during training, stable
- **Speed**: ~100 steps per 18 minutes on RTX 5060 Ti

## 💻 Hardware Requirements & Performance

### Current Hardware (Optimal)
- ✅ **GPU**: RTX 5060 Ti (16GB VRAM)  
- ✅ **CUDA**: 12.8+ compatibility
- ✅ **System RAM**: 32GB recommended
- ✅ **Storage**: SSD for fast checkpointing

### Performance Expectations
| Hardware | Batch Size | Steps/Hour | Training Time |
|----------|------------|------------|---------------|
| **RTX 5060 Ti** | **8** | **~330** | **6-12 hours** |
| RTX 4070 | 6 | ~250 | 8-16 hours |
| RTX 3080 | 4 | ~200 | 12-24 hours |

### Scaling Options
If you want to increase model size with more VRAM:
```yaml
# For 24GB+ VRAM, edit configs/train_config_gpu.yaml:
model:
  n_layers: 16          # Larger model
  n_embd: 1536         # Higher dimension
  max_seq_length: 2048  # Longer sequences

training:
  batch_size: 12        # Larger batches
```

## 🔧 Training Management

### Resume Training (Automatic)
```bash
# Same command automatically detects and resumes
python train_gpu_compatible.py

# Output shows resumption:
📥 Loading checkpoint from models/l1-gpu-compatible/latest_checkpoint.pt
✅ Resumed from epoch 2, step 1847, loss: 2.1432
```

### Manual Checkpoint Selection
```bash
# Use specific checkpoint
python train_gpu_compatible.py --checkpoint models/l1-gpu-compatible/checkpoint_epoch_2_step_1500.pt

# Start fresh (ignore existing checkpoints)
python train_gpu_compatible.py --fresh-start
```

### View Training History
```bash
# Check training log
cat models/l1-gpu-compatible/training.log | tail -20

# View checkpoint files
ls -la models/l1-gpu-compatible/checkpoint_*.pt
```

## 🚨 Troubleshooting

### Common Issues & Solutions

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config
batch_size: 4  # Instead of 8

# Or enable gradient checkpointing
gradient_checkpointing: true
```

**2. Training Not Starting**
```bash
# Verify CUDA setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check data files exist
ls data/processed/train.txt data/processed/val.txt
```

**3. Loss Not Decreasing**
```bash
# Check if resuming from bad checkpoint
rm models/l1-gpu-compatible/latest_checkpoint.pt
python train_gpu_compatible.py  # Start fresh

# Verify data quality
head -5 data/processed/train.txt
```

**4. Slow Training Speed**
```bash
# Check GPU utilization
nvidia-smi

# Ensure mixed precision is enabled
mixed_precision: true  # In config

# Verify data loading isn't bottleneck
dataloader_num_workers: 2  # Try different values
```

### Performance Optimization Tips

**Memory Optimization:**
- Keep batch_size at 8 for RTX 5060 Ti 16GB
- Use gradient_accumulation_steps to simulate larger batches
- Enable gradient_checkpointing if memory tight

**Speed Optimization:**
- Ensure pin_memory: true in dataloader
- Use SSD storage for faster checkpoint saves
- Keep sequence length at 1024 (optimal for your setup)

**Quality Optimization:**
- Let training run for full 10 epochs
- Monitor validation loss - stop if it stops improving
- Use learning rate warmup for stability

## 📊 Training Results & Evaluation

### Expected Training Progression
```bash
# Typical loss progression on Wikipedia data:
Epoch 1: 4.2 → 3.1 (rapid initial learning)
Epoch 3: 3.1 → 2.4 (steady improvement) 
Epoch 6: 2.4 → 2.1 (fine-tuning)
Epoch 10: 2.1 → 1.9 (convergence)
```

### Model Quality Assessment
```bash
# Test generation quality during training
python generate_simple.py --prompt "Wikipedia describes artificial intelligence as"

# Expected improvement:
# Early: Random/incoherent text
# Middle: Grammatical but basic text  
# Final: Coherent, factual responses
```

### Best Practices
- **Patience**: Let training complete all 10 epochs
- **Monitoring**: Check progress every few hours
- **Validation**: Test generation quality periodically
- **Backup**: Checkpoints auto-saved, but backup important ones

## ✅ Next Steps After Training

### 1. Model Evaluation
```bash
# Generate comprehensive test outputs
python generate_simple.py --prompt "The history of" --max-tokens 100
python generate_simple.py --prompt "Science shows that" --max-tokens 100
python generate_simple.py --prompt "Technology will" --max-tokens 100
```

### 2. Fine-tuning (Optional)
If you want to specialize your model:
```bash
# Add domain-specific data and continue training
python add_dataset.py papers_arxiv  # For scientific knowledge
python train_gpu_compatible.py      # Continue from checkpoint
```

### 3. Production Use
```bash
# Use best model for generation
python generate_simple.py \
    --model_path models/l1-gpu-compatible/best_checkpoint.pt \
    --prompt "Your prompt here"
```

Your L1 training setup is optimized and ready for excellent results! 🚀
