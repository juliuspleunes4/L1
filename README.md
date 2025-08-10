# L1 - Large Language Model

L1 is a transformer-based large language model implementation built from scratch using PyTorch. This project provides a complete pipeline for training, fine-tuning, and deploying your own language model with comprehensive documentation and best practices.

## 🚀 Features

- **Custom Transformer Architecture**: Complete implementation with multi-head attention, feed-forward networks, and positional embeddings
- **GPU Accelerated Training**: Full CUDA support with RTX 50.. Series optimization and mixed precision training
- **Advanced Checkpointing**: Automatic saves every 100 steps with intelligent cleanup and seamless resume capability
- **Memory Optimization**: Gradient checkpointing, model compilation, and memory-efficient training for high-end GPUs
- **BPE Tokenization**: Byte Pair Encoding tokenizer implementation from scratch
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

Ready to train your own language model? Here's the fastest way:

### **🚀 4-Step Quick Start**

```bash
# Step 1: Get high-quality training data (500k samples)
python add_dataset.py --preset advanced

# Step 2: Prepare the large dataset
python prepare_large_dataset.py data/raw/combined_dataset.txt

# Step 3: Start GPU training
python train_gpu_compatible.py

# Step 4: Generate text with your trained model
python generate_simple.py --prompt "The future of AI is"
```

**That's it!** The preset automatically downloads Wikipedia + ArXiv papers, processes the data, and you're ready to train.

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
| Medium | 12     | 12    | 768       | ~117M      | 6GB        | Balanced | `base_config.yaml` |
| **L1 Current** | **12** | **16** | **1024** | **~155M** | **8GB** | **Production** | **`train_config_gpu.yaml`** |

> **Note**: L1 Current model is optimized for RTX 5060 Ti (16GB VRAM) with batch size 8 and mixed precision training. You can customize these configurations by editing the YAML files directly.

## 🔧 Configuration

The model and training parameters are configured via YAML files in the `configs/` directory:

### GPU Training Configuration (`configs/train_config_gpu.yaml`)
```yaml
model:
  max_seq_length: 1024              # Optimized for RTX 5060 Ti 16GB memory
  n_layers: 12
  n_heads: 16  
  n_embd: 1024
  n_inner: 4096
  dropout: 0.1

training:
  num_epochs: 10
  batch_size: 8                     # Memory-optimized for RTX 5060 Ti
  learning_rate: 0.0001
  mixed_precision: true             # AMP for speed and memory efficiency
  checkpoint_every_steps: 100       # Save every ~18 minutes
  max_checkpoints_to_keep: 5        # Auto-cleanup old checkpoints
  gradient_accumulation_steps: 4
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
- ✅ **Model Compilation**: PyTorch 2.0+ compilation for optimization
- ✅ **Gradient Checkpointing**: Memory-efficient training for large models
- ✅ **Smart Checkpointing**: Save every 100 steps (~18 min) with auto-cleanup
- ✅ **Automatic Resume**: Seamless training continuation from interruptions
- ✅ **Memory Optimization**: RTX 50.. Series specific optimizations

### CPU Training
For systems without CUDA support:
```bash
python train_minimal.py
```

### Training Monitoring
L1 provides comprehensive training monitoring:

```bash
# Real-time training metrics
🎓 Training Configuration:
   ├── Epochs: 10
   ├── Total steps: 11,250
   ├── Checkpoint every: 100 steps (~18 min)
   ├── Keep checkpoints: 5 latest
   ├── Mixed precision: True
   └── Optimizer: AdamW

# Progress tracking
💾 Saving progress checkpoint at step 1100...
🗑️ Cleaned up old checkpoint: checkpoint_epoch_1_step_600.pt
```

### Resume Training
Training automatically resumes from the last checkpoint:
```bash
# Same command detects and resumes automatically
python train_gpu_compatible.py

# Output:
📥 Loading checkpoint from models/l1-gpu-compatible/latest_checkpoint.pt
✅ Resumed from epoch 2, step 1847, loss: 2.1432
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
- **Model Compilation**: Automatic PyTorch 2.0+ compilation
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Batch Size**: Optimized to 8 for 16GB VRAM
- **Sequence Length**: Reduced to 1024 for memory efficiency

### Memory Management
- **Automatic Checkpointing**: Saves every 100 steps with cleanup
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
  batch_size: 8
  mixed_precision: true
  gradient_accumulation_steps: 4
  checkpoint_every_steps: 100
  max_checkpoints_to_keep: 5
```

## 🧪 Testing & Validation

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

## 🔧 Troubleshooting

### Validation Script
Run this to check if everything is set up correctly:
```bash
python validate_setup.py
```

### Common Issues

**1. `ModuleNotFoundError: No module named 'kagglehub'`**
```bash
pip install kagglehub pandas
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

**6. GPU out of memory**
Reduce batch size in `configs/train_config_gpu.yaml`:
```yaml
training:
  batch_size: 4  # Reduce from 8 to 4
```

## 📊 Current Status

**L1 is actively being trained and improved:**

- ✅ **GPU Compatibility**: Full RTX 5060 Ti support with CUDA 12.8
- ✅ **Model Architecture**: 155.8M parameter transformer (12 layers, 16 heads)
- ✅ **Training Pipeline**: Advanced checkpointing every 100 steps
- ✅ **Dataset**: Wikipedia Simple English (90,000+ samples)
- ✅ **Optimization**: Mixed precision, gradient checkpointing, model compilation
- 🔄 **Current Training**: Active training with automatic resume capability
- 📈 **Performance**: Excellent loss reduction and convergence

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
