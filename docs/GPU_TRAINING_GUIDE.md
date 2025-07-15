# 🚀 L1 GPU Training Setup Guide

## GPU Support Status
✅ **Your L1 project already has full GPU support!**

Your current implementation automatically:
- Detects CUDA availability 
- Moves model and data to GPU
- Handles device placement correctly

## 🎮 Enhanced GPU Training

### 1. Use the GPU-Optimized Script
```bash
# Use the enhanced training script
python train_gpu.py
```

### 2. GPU Configuration
The new `configs/train_config_gpu.yaml` includes:
- **Larger model**: 12 layers, 1024 embedding dimension  
- **Longer sequences**: 2048 tokens (4x longer)
- **Mixed precision**: Automatic Mixed Precision (AMP) for faster training
- **Gradient accumulation**: Simulate larger batch sizes
- **Memory optimizations**: Gradient checkpointing

### 3. Model Size Comparison
```
Current (CPU):     19.7M parameters
GPU Config:        ~80M+ parameters (4x larger!)
```

## 📚 Large Dataset Processing

### Download Kaggle Datasets
```bash
# Install Kaggle API
pip install kaggle

# Download popular text datasets
kaggle datasets download -d Cornell-University/arxiv
kaggle datasets download -d snapcrack/all-the-news
kaggle datasets download -d datasnaek/youtube-new
```

### Process Large Datasets
```bash
# CSV format (most common)
python prepare_large_dataset.py dataset.csv --text-column "content" --max-samples 1000000

# JSON format  
python prepare_large_dataset.py dataset.json --text-field "text" --vocab-size 50000

# Plain text
python prepare_large_dataset.py dataset.txt --max-samples 500000
```

## 🔧 GPU Optimizations Included

### Automatic Features:
- **Device Detection**: Auto-detects best available GPU
- **Memory Management**: Pin memory for faster transfers
- **Mixed Precision**: 16-bit training for speed + memory
- **Gradient Checkpointing**: Trade compute for memory
- **Model Compilation**: PyTorch 2.0 optimizations

### Performance Monitoring:
- GPU memory usage tracking
- Training speed metrics  
- Automatic checkpoint saving
- Detailed performance logs

## 🎯 Recommended GPU Training Workflow

### Step 1: Prepare Large Dataset
```bash
# Download a large text dataset from Kaggle
python prepare_large_dataset.py your_dataset.csv --max-samples 1000000 --vocab-size 50000
```

### Step 2: Train on GPU  
```bash
# Train with GPU optimizations
python train_gpu.py
```

### Step 3: Monitor Progress
- Watch GPU utilization: `nvidia-smi`
- Check training logs: `tail -f models/l1-gpu-v1/training.log`
- Monitor loss curves and memory usage

### Step 4: Scale Up
Once working, increase model size in `configs/train_config_gpu.yaml`:
```yaml
model:
  n_layers: 24        # Larger model
  n_embd: 2048        # Higher dimension
  max_seq_length: 4096 # Longer sequences
```

## 💡 Tips for Your New GPU PC

### Hardware Requirements
- **VRAM**: 8GB+ recommended for the GPU config
- **RAM**: 32GB+ for large dataset processing  
- **Storage**: SSD for faster data loading

### Environment Setup
```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Expected Performance
With a powerful GPU, you should see:
- **10-100x faster training** compared to CPU
- **Larger models** (50M-1B+ parameters possible)
- **Better quality** from more data and larger models

## 📊 Scaling Guidelines

### Model Size vs GPU Memory
| Model Size | Parameters | GPU Memory | Notes |
|------------|------------|------------|--------|
| Small      | 19M        | 2-4GB      | Current size |
| Medium     | 80M        | 6-8GB      | GPU config |
| Large      | 300M       | 12-16GB    | Scale up |
| XL         | 1B+        | 24GB+      | High-end |

### Dataset Size Recommendations
- **Small**: 100K-1M samples (quick experiments)
- **Medium**: 1M-10M samples (good performance)  
- **Large**: 10M+ samples (production quality)

## 🚀 Ready to Scale!

Your L1 project is already GPU-ready. Just:
1. Move to your new PC
2. Install PyTorch with CUDA
3. Run `python train_gpu.py`
4. Scale up with larger datasets and models!

The enhanced scripts will automatically utilize all your GPU's power for dramatically faster and more capable LLM training.
