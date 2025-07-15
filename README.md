# L1 - Large Language Model

L1 is a transformer-based large language model implementation built from scratch using PyTorch. This project provides a complete pipeline for training, fine-tuning, and deploying your own language model with comprehensive documentation and best practices.

## 🚀 Features

- **Custom Transformer Architecture**: Complete implementation with multi-head attention, feed-forward networks, and positional embeddings
- **Flexible Training Pipeline**: Support for pretraining, fine-tuning, and distributed training
- **BPE Tokenization**: Byte Pair Encoding tokenizer implementation from scratch
- **Model Serving**: REST API for text generation and inference
- **Configuration Management**: YAML-based configuration system for easy experimentation
- **Comprehensive Logging**: Training metrics, tensorboard integration, and monitoring
- **Checkpoint Management**: Automatic saving, loading, and best model selection
- **Production Ready**: Optimized for both research and deployment

## 📁 Project Structure

```
L1/
├── src/                 # Source code
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
└── demo.py            # Quick demo script
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd L1
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv l1_env
   l1_env\Scripts\activate  # Windows
   # source l1_env/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install PyYAML tqdm regex transformers datasets tensorboard
   ```

4. **Verify installation**:
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
```bash
python scripts/prepare_data.py \
    --input data/raw/sample_text.txt \
    --output data/processed/
```

### 3. Train a Model
```bash
python scripts/train.py --config configs/base_config.yaml
```

### 4. Generate Text
```bash
python scripts/generate.py \
    --model checkpoints/best_model.pt \
    --prompt "The future of artificial intelligence is" \
    --max-tokens 50
```

### 5. Start API Server (Coming Soon)
```bash
python scripts/serve.py \
    --model checkpoints/best_model.pt \
    --port 8000
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

| Model | Layers | Heads | Embedding | Parameters | Use Case |
|-------|---------|-------|-----------|------------|----------|
| Tiny  | 2      | 4     | 128       | ~540K      | Testing/Demo |
| Small | 6      | 6     | 384       | ~25M       | Experiments |
| Base  | 12     | 12    | 768       | ~110M      | Production |
| Large | 24     | 16    | 1024      | ~340M      | Research |

## 🔧 Configuration

The model and training parameters are configured via YAML files in the `configs/` directory:

### Model Configuration
```yaml
model:
  vocab_size: 50257
  max_seq_length: 1024
  n_layers: 12
  n_heads: 12
  n_embd: 768
  dropout: 0.1
```

### Training Configuration
```yaml
training:
  num_epochs: 3
  batch_size: 8
  learning_rate: 5e-4
  optimizer_type: "adamw"
  scheduler_type: "cosine"
  warmup_steps: 500
```

## 📈 Training

### Pretraining
Train from scratch on a large text corpus:
```bash
python scripts/train.py \
    --config configs/pretrain_config.yaml \
    --mode pretrain
```

### Fine-tuning
Fine-tune on specific tasks:
```bash
python scripts/train.py \
    --config configs/finetune_config.yaml \
    --mode finetune \
    --checkpoint checkpoints/pretrained.pt
```

### Monitoring
- **Tensorboard**: `tensorboard --logdir checkpoints/tensorboard`
- **Weights & Biases**: Configure in training config
- **Console**: Real-time metrics during training

## 🎛️ Text Generation

L1 supports various generation strategies:

### Sampling Methods
- **Greedy**: Always select the most likely token
- **Top-k**: Sample from k most likely tokens  
- **Top-p (Nucleus)**: Sample from tokens with cumulative probability p
- **Temperature**: Control randomness of sampling

### Example
```python
# Generate with nucleus sampling
python scripts/generate.py \
    --model checkpoints/best_model.pt \
    --prompt "The future of AI" \
    --max-tokens 100 \
    --temperature 0.8 \
    --top-p 0.9
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

### Memory Optimization
- Use `gradient_checkpointing: true` for large models
- Enable `fp16: true` for mixed precision training
- Adjust `batch_size` based on available GPU memory

### Speed Optimization  
- Use multiple `dataloader_num_workers`
- Enable `pin_memory: true` for GPU training
- Consider model compilation with `torch.compile()`

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

**Built with ❤️ for the open source AI community**
