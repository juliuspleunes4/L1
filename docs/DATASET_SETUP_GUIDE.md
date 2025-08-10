# 📚 L1 Dataset Setup Guide

## 🚀 Current Status
L1 is already set up with **Wikipedia Simple English** dataset:
- ✅ **90,000+ high-quality samples** ready for training
- ✅ **BPE tokenizer** with 32,003 vocabulary (32k base + special tokens)
- ✅ **Preprocessed data** in `data/processed/`

## 🎯 Quick Dataset Options

### Option 1: Use Pre-configured Datasets (Recommended)
L1 includes **15+ curated datasets** ready to download:

```bash
# Beginner: Fast training (50k samples)
python add_dataset.py --preset beginner

# Intermediate: Balanced training (150k samples)  
python add_dataset.py --preset intermediate

# Advanced: Comprehensive training (500k samples)
python add_dataset.py --preset advanced
```

### Option 2: Choose Specific Datasets
```bash
# High-quality options
python add_dataset.py wikipedia_full      # 500k Wikipedia articles
python add_dataset.py papers_arxiv        # Scientific papers  
python add_dataset.py books_gutenberg     # Classic literature
python add_dataset.py openwebtext         # GPT-style web text

# Specialized datasets
python add_dataset.py code_stackoverflow  # Programming Q&A
python add_dataset.py reddit_comments     # Conversational text
python add_dataset.py news_all            # Current events
```

### Option 3: Continue with Current Dataset
Your current Wikipedia Simple dataset is excellent for:
- High text quality and educational content
- Manageable size for development and testing
- Good vocabulary coverage for general language

```bash
# Continue training with current dataset
python train_gpu_compatible.py
```

## 📊 Available Dataset Presets

L1's `datasets.yaml` includes curated collections:

### Beginner Preset (Recommended for testing)
- **Wikipedia Simple** + **News articles**
- **50,000 samples** - Fast training and iteration  
- **Vocab size**: 15,000 tokens
- **Training time**: 2-4 hours on GPU

### Intermediate Preset (Balanced)
- **Wikipedia** + **Books** + **News** 
- **150,000 samples** - Good performance balance
- **Vocab size**: 25,000 tokens  
- **Training time**: 6-12 hours on GPU

### Advanced Preset (Production)
- **Wikipedia** + **Research Papers** + **Books**
- **500,000 samples** - High-quality results
- **Vocab size**: 50,000 tokens
- **Training time**: 24-48 hours on GPU

### Specialized Presets
```bash
python add_dataset.py --preset conversational  # Reddit + Twitter + Chat
python add_dataset.py --preset technical       # GitHub + Stack Overflow  
python add_dataset.py --preset knowledge       # Full Wikipedia + Papers
python add_dataset.py --preset creative        # Books + Literature + Stories
```
## 🔧 Adding Custom Kaggle Datasets

L1 makes it incredibly easy to add any Kaggle dataset:

### Method 1: KaggleHub (Easiest)
```python
import kagglehub

# Download any Kaggle dataset directly
dataset_path = kagglehub.dataset_download("username/dataset-name")

# Use with L1
python prepare_large_dataset.py --custom_path dataset_path
```

### Method 2: Add to datasets.yaml (Permanent)
Edit `datasets.yaml` and add your dataset:
```yaml
your_awesome_dataset:
  name: "Your Dataset Name"
  description: "Dataset description"
  download_method: "kagglehub" 
  kagglehub_path: "username/dataset-name"  # From Kaggle URL
  auto_detect_format: true
  recommended_samples: 100000
  recommended_vocab: 20000
  quality: "high"
  topics: ["your", "topics"]
```

Then use it:
```bash
python add_dataset.py your_awesome_dataset
```

### Method 3: Kaggle API (Traditional)
```bash
# Setup Kaggle API (one time)
pip install kaggle
# Add kaggle.json credentials to ~/.kaggle/

# Download and use
kaggle datasets download username/dataset-name -p data/raw/
python add_dataset.py --custom data/raw/dataset-name
```

## 📈 Dataset Management & Verification

### View Available Datasets
```bash
# List all pre-configured datasets
python dataset_manager.py --list

# Get detailed info about a dataset
python dataset_manager.py --info wikipedia_simple

# Preview dataset samples
python dataset_manager.py --preview wikipedia_simple --samples 5
```

### Verify Your Current Data
```bash
# Check current dataset status
ls data/processed/
# Should show: train.txt, val.txt, tokenizer.json

# View dataset statistics 
python dataset_manager.py --stats
```

### Expected Output
```
📊 Current Dataset (Wikipedia Simple):
   ├── Training samples: 90,247
   ├── Validation samples: 10,027  
   ├── Vocabulary size: 1,783 tokens
   ├── Average length: 512.3 characters
   └── Quality: High (educational content)

✅ Ready for training with:
   └── python train_gpu_compatible.py
```

## 🎯 Dataset Selection Guide

### For Current Development (Recommended)
✅ **Keep your current Wikipedia Simple dataset**
- High-quality educational content
- Perfect for model development and testing
- Already processed and ready
- Good vocabulary coverage (1,783 tokens)

### For Better Performance
Consider upgrading vocabulary size:
```bash
# Current vocab (1,783) is small for production
# Recommend 8,000-32,000 for better text quality
# Note: Requires retraining from scratch
```

### For Specialized Applications
| Use Case | Recommended Dataset | Samples | Training Time |
|----------|-------------------|---------|---------------|
| **General AI** | intermediate preset | 150k | 6-12 hours |
| **Educational** | Current (Wikipedia) | 90k | ✅ **Current** |
| **Conversational** | conversational preset | 200k | 12-24 hours |
| **Technical** | technical preset | 100k | 4-8 hours |
| **Creative Writing** | creative preset | 80k | 4-6 hours |

## 💾 Storage & Performance

### Current Setup Requirements
| Component | Size | Status |
|-----------|------|--------|
| Raw data | ~150MB | ✅ Ready |
| Processed data | ~50MB | ✅ Ready |
| Model checkpoints | ~2GB | ✅ Auto-managed |
| Tokenizer | <1MB | ✅ Ready |

### Scaling Up Requirements
| Dataset Size | Disk Space | Processing Time | VRAM | Training Time |
|--------------|------------|-----------------|------|---------------|
| **Current (90k)** | **200MB** | **✅ Done** | **8GB** | **6-12 hours** |
| Intermediate (150k) | 500MB | 10-20 min | 8GB | 12-24 hours |
| Advanced (500k) | 2GB | 1-2 hours | 12GB | 48-96 hours |

## 🔍 Troubleshooting

### Common Dataset Issues
```bash
# Check if dataset is properly loaded
python -c "
import os
print('Train file:', os.path.exists('data/processed/train.txt'))
print('Val file:', os.path.exists('data/processed/val.txt'))  
print('Tokenizer:', os.path.exists('data/processed/tokenizer.json'))
"

# If files are missing, re-run current setup:
python prepare_large_dataset.py
```

### Performance Issues
```bash
# If training is slow, check:
nvidia-smi                    # GPU utilization
ls models/l1-gpu-compatible/  # Checkpoints being saved

# If out of memory, reduce batch size:
# Edit configs/train_config_gpu.yaml → batch_size: 4
```

## ✅ Recommendations

### For Immediate Use
Keep your current setup - it's well-optimized:
- ✅ High-quality Wikipedia Simple dataset  
- ✅ Proper vocabulary and tokenization
- ✅ Ready for GPU training
- ✅ Good for development and experimentation

### For Future Improvements
Consider these upgrades after current training:
1. **Vocabulary optimization**: Current 32k vocab is already excellent for quality
2. **Add more data**: intermediate preset for more diverse training
3. **Specialized datasets**: Match your specific use case

Your current setup with 32k BPE vocabulary is production-ready!
