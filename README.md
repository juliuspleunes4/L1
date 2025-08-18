# L1 

<p align="left">
  <img src="assets/l1.jpeg" alt="L1" width="320">
</p>

<p>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="Python" width="32" title="Python">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PyTorch_logo_icon.svg/744px-PyTorch_logo_icon.svg.png?20200318225611" alt="PyTorch" width="32" title="PyTorch">
</p>

L1 is a transformer-based large language model implementation built from scratch using PyTorch. This project provides a complete pipeline for training, fine-tuning, and deploying your own language model with comprehensive documentation and best practices.

## 🚀 Features

- **Custom Transformer Architecture**: Complete implementation with multi-head attention, feed-forward networks, and positional embeddings
- **GPU Accelerated Training**: Full CUDA support with RTX 50.. Series optimization and mixed precision training
- **Advanced Checkpointing**: Automatic saves every 1000 steps with intelligent cleanup and seamless resume capability
- **Memory Optimization**: Gradient checkpointing, model compilation, and memory-efficient training for high-end GPUs
- **BPE Tokenization**: Byte Pair Encoding implementation from scratch for intelligent text understanding
- **Intelligence-Optimized**: 32k vocabulary with subword tokenization for coherent reasoning
- **Flexible Training Pipeline**: Support for pretraining, fine-tuning, and distributed training
- **Model Serving**: REST API for text generation and inference
- **Configuration Management**: YAML-based configuration system for easy experimentation
- **Comprehensive Logging**: Training metrics, tensorboard integration, and monitoring
- **Production Ready**: Optimized for both research and deployment

## 📁 Project Structure

```
L1/
├── src/                # Source code
│   ├── models/         # Model architectures (transformer, config, embeddings)
│   ├── training/       # Training pipeline (trainer, optimizer, loss)
│   ├── data/           # Data processing (tokenizer, dataset, preprocessing)
│   ├── inference/      # Model inference and serving
│   └── utils/          # Utilities (logging, device management, etc.)
├── configs/            # Configuration files (YAML)
├── scripts/            # Training and inference scripts  
├── data/               # Dataset storage (raw and processed)
├── tests/              # Unit tests
├── docs/               # Documentation
├── checkpoints/        # Model checkpoints (auto-created)
├── logs/               # Training logs (auto-created)
└── demo.py             # Quick demo script
```

## 🛠️ Installation

### Prerequisites
- **Python 3.8+**
- **CUDA 12.8+** (for GPU training)
- **16GB+ RAM** (32GB recommended for large models)
- **Modern GPU** (RTX 4060+, tested on RTX 5060 Ti 16GB )

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/juliuspleunes4/L1
   cd L1
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv l1_env
   l1_env\Scripts\activate       # Windows
   # source l1_env/bin/activate  # Linux/Mac
   ```

3. **Install PyTorch with CUDA support**:
   ```bash
   # For CUDA 12.1+ (RTX 5060+ Optimised, also works on the 40+ series)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Or for CPU-only training (slower)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install remaining dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation** (optional):
   ```bash
   # Test data preparation (should work immediately)
   python add_dataset.py --help
   
   # Test GPU setup (requires PyTorch)
   python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

6. **Run validation test**:
* Note: The test will most likely fail 3 out of the 5, because of CUDA problems. This can be safely ignored.
   ```bash
   python validate_setup.py
   ```

## 🎯 Quick Start

Ready to train your own intelligent language model? Here's the fastest way:

### **🚀 4-Step Quick Start**

```bash
# Step 1: Get high-quality training data (500k samples)
python add_dataset.py --preset advanced

# Step 2: Prepare the dataset with BPE tokenization (for intelligence)
python prepare_large_dataset.py data/raw/combined_dataset.txt --vocab-size 32000

# Step 3: Start GPU training (resume-capable)
python train_gpu_compatible.py

# Step 4: Generate text with your trained model
python generate_simple.py --model_path models/l1-gpu-compatible --prompt "The future of AI is"
```

**That's it!** The preset automatically downloads Wikipedia + ArXiv papers, and the BPE tokenization creates a 32k vocabulary for intelligent text understanding and generation.

### **🧠 Intelligence Features:**
- **BPE Tokenization**: 32,000 subword tokens with special tokens for intelligent text understanding
- **Stable Architecture**: 12 layers optimized for reliability (134M parameters)
- **Coherent Generation**: Produces meaningful sentences and coherent text
- **Lightning Fast**: 108x speed improvement (18min→10sec per 100 steps)
- **Auto-Resume**: Training automatically resumes from latest checkpoint 

---

### **📚 Next Steps**

1. **Demo the Project**: Run `python demo.py` to test all components
2. **Customize Training**: Edit `configs/train_config_gpu.yaml` for your hardware
3. **Add Custom Data**: See the Data Preparation section below for advanced options
4. **Monitor Training**: Use `tail -f models/l1-gpu-compatible/training.log`

## 📊 Model Architecture

L1 implements a decoder-only transformer architecture with:

- **Multi-Head Self-Attention**: Configurable number of attention heads with causal masking
- **Feed-Forward Networks**: Position-wise fully connected layers with GELU activation
- **Layer Normalization**: Pre-norm architecture for training stability
- **Positional Encoding**: Learnable positional embeddings
- **Residual Connections**: Skip connections for gradient flow
- **BPE Tokenization**: Byte Pair Encoding implementation from scratch

### Model Sizes

| Model | Layers | Heads | Embedding | Parameters | GPU Memory | Use Case | Config File |
|-------|---------|-------|-----------|------------|------------|----------|-------------|
| Small | 6      | 8     | 512       | ~25M       | 4GB        | Experiments | `train_config.yaml` |
| **L1 Stable** | **12** | **12** | **768** | **~134M** | **8GB** | **Stable Training** | **`train_config_gpu.yaml`** |
| L1 Large | 16     | 16    | 1024      | ~220M      | 12GB       | Advanced (experimental) | Custom config |

> **Note**: L1 Stable model uses BPE tokenization (32k vocab) with conservative settings to prevent system freezing. Still intelligent but prioritizes stability.

## 🔧 Configuration

The model and training parameters are configured via YAML files in the `configs/` directory:

### Stable GPU Training Configuration (`configs/train_config_gpu.yaml`)
```yaml
model:
  vocab_size: 32000              # BPE tokenization for intelligence (overwritten by actual tokenizer size during training)
  max_seq_length: 512            # Conservative for system stability
  n_layers: 12                   # Balanced depth for stable training
  n_heads: 12                    # Sufficient attention heads
  n_embd: 768                    # Good embedding size
  n_inner: 3072                  # 4x embedding size
  dropout: 0.1

training:
  num_epochs: 10                 # Reasonable training duration
  batch_size: 4                  # Conservative batch size
  learning_rate: 0.0001          # Stable learning rate
  mixed_precision: true          # Memory efficiency
  checkpoint_every_steps: 1000   # Every ~2 minutes
  max_checkpoints_to_keep: 5     # Auto-cleanup
  gradient_accumulation_steps: 4 # Effective batch size 16
```

### Minimal Configuration (`configs/base_config.yaml`)
```yaml
model:
  vocab_size: 50257
  max_seq_length: 512
  n_layers: 6
  n_heads: 8
  n_embd: 512
  dropout: 0.1

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 5e-4
```

## 📊 Data Preparation & Management

L1 includes a powerful dataset management system that makes adding datasets incredibly easy. You have **15+ pre-configured datasets** ready to use, plus simple ways to add your own.

### 🚀 Using Pre-configured Dataset Presets

Choose from curated datasets in `datasets.yaml`:

```bash
# Advanced: Comprehensive training (recommended)
python add_dataset.py --preset advanced
# → Wikipedia Simple + ArXiv Papers (500k samples)

# Beginner: Quick training with high-quality data
python add_dataset.py --preset beginner
# → Wikipedia Simple + News (50k samples)

# Intermediate: Balanced training 
python add_dataset.py --preset intermediate  
# → Wikipedia + Books + News (150k samples)

# Specialized presets
python add_dataset.py --preset conversational  # Reddit + Twitter + Wikipedia
python add_dataset.py --preset technical       # GitHub + Stack Overflow + Papers
python add_dataset.py --preset knowledge       # Full Wikipedia + Papers + Books
```

**What happens when you run a preset:**
1. 🔄 Downloads the specified datasets automatically
2. 📝 Combines them into a single training file
3. ✅ Processes and saves to `data/processed/` for training

### 📚 Available Datasets

| Dataset | Samples | Quality | Topics | Use Case |
|---------|---------|---------|--------|----------|
| **Wikipedia Simple** | 100k | High | Encyclopedia | **Current default** |
| Wikipedia Full | 500k | Very High | Comprehensive | Large-scale training |
| ArXiv Papers | 150k | Very High | Scientific | Technical knowledge |
| Project Gutenberg | 80k | Very High | Literature | Creative writing |
| Stack Overflow | 100k | High | Programming | Code understanding |
| Reddit Comments | 200k | Medium | Conversation | Chat/dialogue |
| News Articles | 50k | High | Current events | Factual knowledge |
| OpenWebText | 500k | High | General web | GPT-style training |

### 🔧 Adding Custom Datasets

#### Method 1: Add to Configuration (Recommended)

1. **Find your Kaggle dataset**: Go to kaggle.com, find your dataset
2. **Edit `datasets.yaml`**: Add your dataset configuration

```yaml
# Example: Adding a new dataset
your_awesome_dataset:
  name: "Your Dataset Name"
  description: "What this dataset contains"
  download_method: "kagglehub"
  kagglehub_path: "username/dataset-name"     # From Kaggle URL
  auto_detect_format: true
  recommended_samples: 100000
  recommended_vocab: 20000
  quality: "high"  # high, very_high, medium
  topics: ["your", "topic", "tags"]

# Add to a preset
presets:
  your_preset:
    name: "Your Custom Training"
    recommended_datasets: ["your_awesome_dataset", "wikipedia_simple"]
    max_samples: 150000
    vocab_size: 25000
    description: "Your custom training mix"
```

3. **Use your dataset**:
```bash
# Use a specific dataset
python add_dataset.py --dataset-id your_awesome_dataset \
    --name "Your Dataset" \
    --description "Description" \
    --method kagglehub \
    --path "username/dataset-name"

# Or use in a preset (edit datasets.yaml first)
python add_dataset.py --preset your_preset
```

#### Method 2: Direct Download (Quick Testing)

```python
import kagglehub

# Download any Kaggle dataset directly
dataset_path = kagglehub.dataset_download("huggingface/squad")
dataset_path = kagglehub.dataset_download("Cornell-University/arxiv")
dataset_path = kagglehub.dataset_download("your-username/your-dataset")

# Then process with L1
python prepare_large_dataset.py "path/to/downloaded/dataset.txt"
```

#### Method 3: Kaggle API (Advanced)

```bash
# Setup Kaggle API (one time)
pip install kaggle
# Add your kaggle.json credentials to ~/.kaggle/

# Download dataset
kaggle datasets download username/dataset-name -p data/raw/
unzip data/raw/dataset-name.zip -d data/raw/

# Process with L1
python prepare_large_dataset.py data/raw/your-extracted-file.txt
```

#### Method 4: Custom Text Files
```bash
# Prepare your own text files
python prepare_large_dataset.py data/raw/your_text.txt

# Or use the scripts directory
python scripts/prepare_data.py \
    --input data/raw/your_text.txt \
    --output data/processed/
```

### 🎯 Dataset Selection Tips

**For beginners:**
- Start with `wikipedia_simple` (current default) - high quality, manageable size
- Add `news_all` for current events knowledge

**For specific use cases:**
- **Conversational AI**: `reddit_comments` + `twitter_sentiment`  
- **Technical/Code**: `code_stackoverflow` + `papers_arxiv`
- **Creative Writing**: `books_gutenberg` + `books_openlib`
- **Scientific**: `papers_arxiv` + `papers_pubmed`

**For production models:**
- Combine multiple high-quality sources
- Use `openwebtext` for general knowledge
- Include domain-specific data for your use case

### 🔍 Verifying Your Dataset

After adding a dataset, verify it's working:

```bash
# Check dataset info
python dataset_manager.py --info your_dataset

# Preview samples  
python dataset_manager.py --preview your_dataset --samples 5

# Validate format
python dataset_manager.py --validate your_dataset
```

## � Training

### GPU Training (Recommended)
Train with GPU acceleration and advanced optimizations:
```bash
# RTX 50.. Series optimized training
python train_gpu_compatible.py

# Monitor progress
tail -f models/l1-gpu-compatible/training.log
```

**GPU Training Features:**
- ✅ **Mixed Precision**: Automatic FP16 for 2x speed improvement
- ✅ **Lightning Fast**: 10+ steps/second 
- ✅ **Gradient Checkpointing**: Memory-efficient training for large models
- ✅ **Smart Checkpointing**: Save every 1000 steps (~2 minutes) with auto-cleanup
- ✅ **Best Model Tracking**: Automatically keeps the best checkpoint based on average loss
- ✅ **Automatic Resume**: Seamless training continuation from interruptions
- ✅ **Optimized Architecture**: BPE tokenization + stable 12-layer configuration

### CPU Training
For systems without CUDA support:
```bash
python train_minimal.py
```

### Training Monitoring
L1 provides comprehensive training monitoring with detailed logging:

```bash
# Real-time training metrics
🎓 Training Configuration:
   ├── Epochs: 10
   ├── Total steps: 112,500
   ├── Checkpoint every: 1000 steps (~2 minutes)
   ├── Keep checkpoints: 5 latest
   ├── Mixed precision: True
   └── Optimizer: AdamW

# Progress tracking with best model detection
🏆 NEW BEST LOSS! Saving best checkpoint at step 5000 (loss: 1.2341, best: 1.2341)
💾 Saving progress checkpoint at step 6000 (loss: 1.2456, best: 1.2341)
```

**📋 Comprehensive Training Log (`training.log`):**
```
2025-08-12 18:06:24 | INFO | TRAINING SESSION STARTED
2025-08-12 18:06:24 | INFO | Model: 12 layers, 12 heads, Vocabulary: 32000
2025-08-12 18:06:25 | INFO | BEST | Epoch: 1 | Step: 1000 | Loss: 2.543200 | Avg_Loss: 2.521000 | Best: 2.543200 | LR: 1.00e-04
2025-08-12 18:06:26 | INFO | CHECKPOINT | Epoch: 1 | Step: 2000 | Loss: 2.456789 | Avg_Loss: 2.445000 | Best: 2.543200 | LR: 1.00e-04
2025-08-12 18:06:27 | INFO | EPOCH 1 COMPLETED | Loss: 2.456789 | Avg_Loss: 2.445000 | LR: 1.00e-04 | EPOCH_COMPLETE
```

**Training Log Features:**
- 📊 **Detailed Metrics**: Timestamp, epoch, step, loss, avg_loss, learning rate for every checkpoint
- 🏆 **Best Tracking**: Clear marking of new best checkpoints with "BEST" indicator
- 📈 **Dual Loss Tracking**: Both instantaneous loss and running average loss
- 🔄 **Resume Compatible**: Logs continue seamlessly across training sessions
- 📈 **Progress Analysis**: Easy to track loss trends and training progress over time
- 💾 **Persistent**: All training history preserved in `models/[model-name]/training.log`

**Best Model Tracking:**
- 🏆 **Smart Best Detection**: Tracks the best model at every checkpoint (1000 steps)
- 📊 **Automatic Updates**: `best_checkpoint.pt` is updated whenever loss improves
- 🎯 **Generation Ready**: Best checkpoint is automatically saved in `pytorch_model.bin` format
- 🔄 **Resume Compatible**: Best loss tracking continues across training sessions

### Resume Training
Training automatically resumes from the last checkpoint:
```bash
# Same command detects and resumes automatically
python train_gpu_compatible.py

# Output:
📥 Loading checkpoint from models/l1-gpu-compatible/latest_checkpoint.pt
✅ Resumed from epoch 2, step 1847, loss: 2.1432
```

**📊 Monitor Training Progress:**
```bash
# Watch live training log
tail -f models/l1-gpu-compatible/training.log

# Check recent progress (last 20 lines)
tail -20 models/l1-gpu-compatible/training.log

# Search for best checkpoints
grep "BEST" models/l1-gpu-compatible/training.log

# Track loss progression
grep "Step:" models/l1-gpu-compatible/training.log | tail -10
```

## 🎛️ Text Generation

L1 supports various generation strategies with the simple generation script:

### Quick Generation
```bash
python generate_simple.py --prompt "The future of AI"
```

### Advanced Generation Options
```bash
python generate_simple.py \
    --prompt "The future of artificial intelligence" \
    --max_tokens 100 \
    --temperature 0.8 \
    --model_path models/l1-gpu-compatible/best_checkpoint.pt
```

**Using the Best Checkpoint:**
```bash
# Use the automatically tracked best model
python generate_simple.py --model_path models/l1-gpu-compatible --prompt "Your prompt here"

# The script automatically uses pytorch_model.bin (best checkpoint format)
# No need to specify best_checkpoint.pt directly
```

### Generation Parameters
- **Temperature**: Control randomness (0.1 = conservative, 1.0 = creative)
- **Max Tokens**: Maximum number of tokens to generate
- **Model Path**: Path to trained model checkpoint

### Example Output
```
Input: "The future of artificial intelligence"
Generated: "The future of artificial intelligence will be shaped by advances in 
machine learning, neural networks, and computational power. These technologies 
will enable more sophisticated reasoning..."
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run the demo script:
```bash
python demo.py
```

## 📚 Documentation

- **[Architecture](docs/architecture.md)**: Detailed L1 transformer architecture  
- **[GPU Training Guide](docs/GPU_TRAINING_GUIDE.md)**: RTX 5060 Ti setup and optimization
- **[Dataset Setup](docs/DATASET_SETUP_GUIDE.md)**: Comprehensive data preparation
- **[Wikipedia Setup](docs/WIKIPEDIA_SETUP.md)**: Wikipedia Simple English dataset guide
- **[Easy Datasets](docs/EASY_DATASETS.md)**: Quick dataset options
- **[Training Guide](docs/training.md)**: Advanced training techniques

## 🚀 Performance Optimization

### GPU Optimization (RTX 5060 Ti and similar)
- **Mixed Precision**: Enabled by default (`mixed_precision: true`)
- **Lightning Fast Training**: 108x speed improvement since its original design
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Conservative Settings**: Stable batch size and sequence length for reliability
- **BPE Tokenization**: Smart 32k vocabulary for efficient learning

### Memory Management
- **Automatic Checkpointing**: Saves every 1000 steps with cleanup
- **GPU Cache Clearing**: Automatic CUDA cache management
- **Gradient Accumulation**: Simulate larger batch sizes (4 steps)
- **Pin Memory**: Enabled for faster GPU data transfer

### Training Safety
- **Resume Capability**: Automatic recovery from interruptions
- **Checkpoint Cleanup**: Keeps only 5 most recent checkpoints
- **Error Handling**: Graceful fallback to CPU on CUDA errors
- **Progress Tracking**: Detailed logging and monitoring

### Speed Optimization
```yaml
# Optimal settings for RTX 5060 Ti 16GB
training:
  batch_size: 4                  # Conservative for stability
  mixed_precision: true          # Memory efficiency
  gradient_accumulation_steps: 4 # Effective batch size 16
  checkpoint_every_steps: 1000   # Reasonable saves (~2 minutes)
  max_checkpoints_to_keep: 5     # Auto-cleanup
```

## 🔧 Troubleshooting

Run the comprehensive test suite:
```bash
python -m pytest tests/ -v
```

Test model functionality:
```bash
python test_model.py
```

Run the demo script:
```bash
python demo.py
```

### Tokenizer Issues

If you're seeing excessive `<unk>` tokens or garbled output during text generation:

```bash
# Fix existing tokenizers (adds missing essential tokens)
python fix_existing_tokenizer.py

# Then test generation
python generate_simple.py --prompt "The future of AI is"
```

This commonly happens with older trained models where the tokenizer was missing basic punctuation and space tokens.

## 🔧 Troubleshooting

### Validation Script
Run this to check if everything is set up correctly:
```bash
python validate_setup.py
```

### Common Issues

**1. `ModuleNotFoundError: No module named 'kagglehub'` or `'tokenizers'`**
```bash
pip install kagglehub pandas tokenizers
```

**2. `--preset` argument not recognized**
Make sure you're using the latest version with the fixed `add_dataset.py`

**3. Dataset download fails**
Some Kaggle datasets require authentication or have access restrictions. The preset will continue with available datasets.

**4. PyTorch DLL load failed (Windows)**
Reinstall PyTorch with the correct CUDA version:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**5. Unicode encoding errors (Windows)**
The scripts are designed to handle Windows encoding. If you see Unicode errors, try running in Windows Terminal or PowerShell.

**6. Text generation produces excessive `<unk>` tokens or garbled output**
```bash
# Fix tokenizer (adds missing essential tokens like spaces and punctuation)
python fix_existing_tokenizer.py

# Then test
python generate_simple.py --prompt "Hello world"
```

**7. GPU out of memory**
Reduce batch size in `configs/train_config_gpu.yaml`:
```yaml
training:
  batch_size: 4  # Reduce from 8 to 4
```

**8. Training seems stuck or very slow**
Reduce batch size in `configs/train_config_gpu.yaml`:
```yaml
training:
  batch_size: 4  # Reduce from 8 to 4
```

**9. Vocabulary size mismatch errors (RuntimeError about embeddings)**
This happens when config files don't match the actual tokenizer vocabulary:
```bash
# Check actual tokenizer size
python -c "import json; data=json.load(open('data/processed/tokenizer.json')); print(f'Actual vocab: {len(data[\"vocab\"])}')"

# Quick fix: Add missing vocab_size field to tokenizer
python -c "
import json
with open('data/processed/tokenizer.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
data['vocab_size'] = len(data['vocab'])
with open('data/processed/tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print('✅ Fixed tokenizer file')
"

# The training script automatically handles this, but you can manually update configs:
# Edit configs/train_config_gpu.yaml and models/*/config.json
# Set vocab_size to match the actual tokenizer size (typically 32,000 for BPE tokenization)
```

## 📊 Current Status

**L1 is actively being trained and improved:**

- ✅ **GPU Compatibility**: Full RTX 5060 Ti support with CUDA 12.8
- ✅ **Model Architecture**: 140M parameter transformer (12 layers, 12 heads)
- ✅ **BPE Tokenization**: 32k vocabulary for intelligent text understanding
- ✅ **Training Pipeline**: Lightning-fast checkpointing every 1000 steps (~2 minutes)
- ✅ **Dataset**: Wikipedia + ArXiv (500,000 samples)
- ✅ **Optimization**: Mixed precision, gradient checkpointing, stable configuration
- 🔄 **Current Training**: Ultra-fast training with automatic resume capability
- 📈 **Performance**: Excellent loss reduction and 108x speed improvement

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **"Attention Is All You Need"** - Vaswani et al. (The Transformer paper)
- **GPT and BERT architectures** - OpenAI and Google Research
- **PyTorch community** - For the excellent deep learning framework
- **Hugging Face** - For transformers library inspiration

## 📞 Support

For questions and support:
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check the `docs/` directory

---

**Built with ❤️ by [Julius Pleunes](https://linkedin.com/in/juliuspleunes)**
