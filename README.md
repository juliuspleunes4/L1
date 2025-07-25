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
   # For CUDA 12.8+ (RTX 50.. Series compatible)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Or for CPU-only training
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install remaining dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify GPU setup** (if using GPU):
   ```bash
   python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

6. **Verify installation**:
   ```bash
   python demo.py
   ```

## 🎯 Quick Start

### 1. Demo the Project
```bash
python demo.py
```
This runs a comprehensive test of all components and shows you that everything is working.

### 2. Prepare Your Data

#### Using Pre-configured Datasets (Recommended)
```bash
# Use Wikipedia Simple English (current default - recommended)
python prepare_large_dataset.py

# Or choose from 15+ pre-configured datasets
python add_dataset.py --preset beginner    # 50k samples, quick training
python add_dataset.py --preset intermediate # 150k samples, balanced
python add_dataset.py --preset advanced     # 500k samples, comprehensive
```

#### Adding Your Own Kaggle Dataset
L1 makes it incredibly easy to add any Kaggle dataset:

**Method 1: Add to datasets.yaml (Permanent)**
```yaml
# Edit datasets.yaml and add your dataset:
your_dataset:
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

**Method 2: Direct Download (Quick)**
```python
import kagglehub
dataset_path = kagglehub.dataset_download("username/dataset-name")
python prepare_large_dataset.py --custom_path dataset_path
```

**Method 3: Using Kaggle API**
```bash
# Install and setup Kaggle API
pip install kaggle
kaggle datasets download username/dataset-name -p data/raw/
python add_dataset.py --custom data/raw/dataset-name
```

#### Custom Data
```bash
# Prepare your own text files
python scripts/prepare_data.py \
    --input data/raw/your_text.txt \
    --output data/processed/
```

### 3. Train a Model

#### GPU Training (Recommended)
```bash
# RTX 50.. Series optimized
python train_gpu_compatible.py

# Use custom config
python train_gpu_compatible.py --config configs/train_config_gpu.yaml
```

#### CPU Training
```bash
python train_minimal.py
```

### 4. Generate Text
```bash
python generate_simple.py \
    --prompt "The future of artificial intelligence is" \
    --max-tokens 50
```

### 5. Resume Training
Training automatically resumes from the last checkpoint:
```bash
# Same command automatically resumes
python train_gpu_compatible.py
```

## 📊 Model Architecture

L1 implements a decoder-only transformer architecture with:

- **Multi-Head Self-Attention**: Configurable number of attention heads with causal masking
- **Feed-Forward Networks**: Position-wise fully connected layers with GELU activation
- **Layer Normalization**: Pre-norm architecture for training stability
- **Positional Encoding**: Learnable positional embeddings
- **Residual Connections**: Skip connections for gradient flow
- **BPE Tokenization**: Byte Pair Encoding implementation from scratch

### Model Sizes

| Model | Layers | Heads | Embedding | Parameters | GPU Memory | Use Case |
|-------|---------|-------|-----------|------------|------------|----------|
| Tiny  | 2      | 4     | 128       | ~540K      | 2GB        | Testing/Demo |
| Small | 6      | 6     | 384       | ~25M       | 4GB        | Experiments |
| **L1 Current** | **12** | **16** | **1024** | **~155M** | **8GB** | **Production** |
| Large | 24     | 16    | 1024      | ~340M      | 16GB       | Research |

> **Note**: L1 Current model is optimized for RTX 5060 Ti (16GB VRAM) with batch size 8 and mixed precision training.

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

## � Adding Kaggle Datasets

L1 includes a powerful dataset management system that makes adding Kaggle datasets incredibly easy. You have **15+ pre-configured datasets** ready to use, plus simple ways to add your own.

### 🚀 Quick Start with Pre-configured Datasets

Choose from curated datasets in `datasets.yaml`:

```bash
# Beginner: Quick training with high-quality data
python add_dataset.py --preset beginner
# → Wikipedia Simple + News (50k samples)

# Intermediate: Balanced training 
python add_dataset.py --preset intermediate  
# → Wikipedia + Books + News (150k samples)

# Advanced: Comprehensive training
python add_dataset.py --preset advanced
# → Wikipedia + Research Papers + Books (500k samples)

# Specialized presets
python add_dataset.py --preset conversational  # Reddit + Twitter + Wikipedia
python add_dataset.py --preset technical       # GitHub + Stack Overflow + Papers
python add_dataset.py --preset knowledge       # Full Wikipedia + Papers + Books
```

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

### 🔧 Adding Your Own Kaggle Dataset

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
python add_dataset.py your_awesome_dataset
# or
python add_dataset.py --preset your_preset
```

#### Method 2: Direct Download (Quick Testing)

```python
import kagglehub

# Download any Kaggle dataset directly
dataset_path = kagglehub.dataset_download("huggingface/squad")
dataset_path = kagglehub.dataset_download("Cornell-University/arxiv")
dataset_path = kagglehub.dataset_download("your-username/your-dataset")

# Use with L1
python prepare_large_dataset.py --custom_path dataset_path
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
python add_dataset.py --custom data/raw/dataset-name
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

## �📈 Training

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

- **[Model Architecture](docs/architecture.md)**: Detailed architecture overview
- **[Training Guide](docs/training.md)**: Comprehensive training instructions
- **[API Reference](docs/api.md)**: API documentation (coming soon)
- **[Configuration Guide](docs/configuration.md)**: Configuration options (coming soon)

## 🚀 Performance Tips

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

## 📚 Documentation

- **[Architecture](docs/architecture.md)**: Detailed L1 transformer architecture
- **[GPU Training Guide](docs/GPU_TRAINING_GUIDE.md)**: RTX 5060 Ti setup and optimization
- **[Dataset Setup](docs/DATASET_SETUP_GUIDE.md)**: Comprehensive data preparation
- **[Wikipedia Setup](docs/WIKIPEDIA_SETUP.md)**: Wikipedia Simple English dataset guide
- **[Easy Datasets](docs/EASY_DATASETS.md)**: Quick dataset options
- **[Training Guide](docs/training.md)**: Advanced training techniques

## 🧪 Testing

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
