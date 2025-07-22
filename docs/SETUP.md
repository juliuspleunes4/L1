# L1 - Large Language Model Setup Guide

## ✅ Current Status
L1 project is **production-ready** with:
- ✅ **155.8M parameter transformer** (12 layers, 16 heads, 1024 embedding)
- ✅ **Wikipedia Simple dataset** (90,000+ samples) pre-processed
- ✅ **RTX 5060 Ti optimization** with mixed precision training
- ✅ **Smart checkpointing** system with auto-resume

## 🛠️ Prerequisites

### Hardware Requirements
- **GPU**: RTX 4060+ (8GB VRAM minimum, 16GB recommended)
- **System RAM**: 16GB+ (32GB recommended)
- **Storage**: 10GB+ free space on SSD
- **CPU**: Modern multi-core processor

### Software Requirements
- **Python**: 3.8+ (3.9+ recommended)
- **CUDA**: 12.8+ for RTX 50.. Series compatibility
- **Git**: For repository management

## 🚀 Quick Setup

### 1. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/juliuspleunes4/L1
cd L1

# Create virtual environment
python -m venv l1_env

# Activate virtual environment
l1_env\Scripts\activate       # Windows
# source l1_env/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
# Install PyTorch with CUDA support (RTX 50.. Series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Verify GPU Setup
```bash
# Verify CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Expected output:
# CUDA Available: True
# GPU: NVIDIA GeForce RTX 5060 Ti
```

### 4. Quick Demo
```bash
# Run comprehensive demo (verifies everything works)
python demo.py
```

## 🎯 Ready to Train

Your L1 comes with **Wikipedia Simple dataset** already processed and ready:

### 1. Verify Data is Ready
```bash
# Check processed data exists
ls data/processed/
# Should show: train.txt, val.txt, tokenizer.json

# View dataset stats
python -c "
with open('data/processed/train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(f'Training samples: {len(lines):,}')
    print('✅ Ready for training!')
"
```

### 2. Start GPU Training (Recommended)
```bash
# Begin training with RTX 5060 Ti optimization
python train_gpu_compatible.py

# Training features:
# ✅ Mixed precision (AMP) for speed + memory
# ✅ Smart checkpointing every 100 steps
# ✅ Automatic resume from interruptions
# ✅ Real-time progress monitoring
```

### 3. Generate Text
```bash
# Test your trained model
python generate_simple.py \
    --prompt "The future of artificial intelligence is" \
    --model_path models/l1-gpu-compatible
```

### 4. Monitor Training
```bash
# Watch training logs in real-time
tail -f models/l1-gpu-compatible/training.log

# Check GPU utilization
nvidia-smi

# View saved checkpoints
ls models/l1-gpu-compatible/checkpoint_*.pt
```

## 📊 Alternative Dataset Options

If you want to experiment with different data:

### Use Pre-configured Datasets
```bash
# Beginner: 50k samples, fast training
python add_dataset.py --preset beginner

# Intermediate: 150k samples, balanced training
python add_dataset.py --preset intermediate

# Advanced: 500k samples, comprehensive training
python add_dataset.py --preset advanced
```

### Add Custom Kaggle Dataset
```bash
# Method 1: Direct download
import kagglehub
dataset_path = kagglehub.dataset_download("username/dataset-name")
python prepare_large_dataset.py --custom_path dataset_path

# Method 2: Add to datasets.yaml (see docs/EASY_DATASETS.md)
```

## 🔧 Development & Testing

### Run Tests
```bash
# Run unit tests
python -m pytest tests/ -v

# Test model functionality  
python test_model.py
```

### Code Quality
```bash
# Format code (if you modify source)
black src/ scripts/ tests/

# Type checking
mypy src/

# Lint code
flake8 src/ scripts/ tests/
```

## 📁 Current Project Structure

```
L1/
├── src/                          # Source code modules
│   ├── models/                   # Transformer architecture
│   ├── training/                 # Training pipeline  
│   ├── data/                     # Data processing
│   └── utils/                    # Utilities
├── configs/                      # YAML configuration files
│   ├── train_config_gpu.yaml     # GPU training config
│   └── base_config.yaml          # Base configuration
├── scripts/                      # Training and utility scripts
├── data/                         # Dataset storage
│   ├── processed/                # ✅ Ready: train.txt, val.txt, tokenizer.json
│   └── raw/                      # Raw data files
├── models/                       # Model outputs
│   └── l1-gpu-compatible/        # Current model checkpoints
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── demo.py                       # Quick demo script
├── train_gpu_compatible.py       # Main GPU training script
├── generate_simple.py            # Text generation
├── add_dataset.py                # Dataset management
└── prepare_large_dataset.py      # Data preprocessing
```

## 🚨 Troubleshooting

### CUDA Issues
```bash
# If CUDA not detected
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA version
nvidia-smi
```

### Memory Issues
```bash
# Reduce batch size in configs/train_config_gpu.yaml
batch_size: 4  # Instead of 8

# Or enable more aggressive memory optimization
gradient_checkpointing: true
```

### Training Not Starting
```bash
# Check data files exist
ls data/processed/train.txt data/processed/val.txt

# If missing, regenerate data
python prepare_large_dataset.py
```

## 📚 Documentation

- **[Architecture](docs/architecture.md)**: Model design details
- **[GPU Training](docs/GPU_TRAINING_GUIDE.md)**: RTX optimization guide
- **[Dataset Setup](docs/DATASET_SETUP_GUIDE.md)**: Data management
- **[Training Guide](docs/training.md)**: Advanced training techniques

## ✅ You're Ready!

Your L1 setup includes:
- ✅ **Production-ready model** (155.8M parameters)
- ✅ **High-quality dataset** (Wikipedia Simple, 90k samples)
- ✅ **GPU optimization** (RTX 5060 Ti compatible)
- ✅ **Smart training** (auto-checkpointing, resume, monitoring)

**Next step**: Run `python train_gpu_compatible.py` to begin training your LLM! 🚀
