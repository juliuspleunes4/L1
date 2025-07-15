# L1 - Large Language Model

L1 is a transformer-based large language model implementation built from scratch using PyTorch. This project provides a complete pipeline for training, fine-tuning, and deploying your own language model.

## 🚀 Features

- **Custom Transformer Architecture**: Implement your own transformer model with attention mechanisms
- **Flexible Training Pipeline**: Support for pretraining and fine-tuning workflows
- **Efficient Data Processing**: Optimized tokenization and data loading
- **Model Serving**: REST API for text generation and inference
- **Configuration Management**: YAML-based configuration system
- **Comprehensive Logging**: Training metrics and model monitoring
- **Checkpoint Management**: Save and resume training from checkpoints

## 📁 Project Structure

```
L1/
├── src/
│   ├── models/           # Model architectures
│   ├── training/         # Training and optimization
│   ├── data/            # Data processing and loading
│   ├── inference/       # Model inference and serving
│   ├── utils/           # Utility functions
│   └── config/          # Configuration management
├── configs/             # Configuration files
├── data/               # Dataset storage
├── checkpoints/        # Model checkpoints
├── logs/               # Training logs
├── tests/              # Unit tests
├── scripts/            # Training and inference scripts
├── docs/               # Documentation
└── requirements.txt    # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd L1
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Quick Start

### 1. Prepare Data
```bash
python scripts/prepare_data.py --input data/raw/text.txt --output data/processed/
```

### 2. Train Model
```bash
python scripts/train.py --config configs/base_config.yaml
```

### 3. Generate Text
```bash
python scripts/generate.py --model checkpoints/l1_model.pt --prompt "Hello, world!"
```

### 4. Start API Server
```bash
python scripts/serve.py --model checkpoints/l1_model.pt --port 8000
```

## 📊 Model Architecture

L1 implements a decoder-only transformer architecture with:

- **Multi-Head Self-Attention**: Configurable number of attention heads
- **Feed-Forward Networks**: Position-wise fully connected layers
- **Layer Normalization**: Pre-norm architecture for stability
- **Positional Encoding**: Learnable positional embeddings
- **Residual Connections**: Skip connections for gradient flow

## 🔧 Configuration

The model and training parameters are configured via YAML files in the `configs/` directory:

- `base_config.yaml`: Basic model configuration
- `training_config.yaml`: Training hyperparameters
- `data_config.yaml`: Data processing settings

## 📈 Training

### Pretraining
Train from scratch on a large text corpus:
```bash
python scripts/train.py --config configs/pretrain_config.yaml --mode pretrain
```

### Fine-tuning
Fine-tune on specific tasks:
```bash
python scripts/train.py --config configs/finetune_config.yaml --mode finetune --checkpoint checkpoints/pretrained.pt
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## 📚 Documentation

- [Model Architecture](docs/architecture.md)
- [Training Guide](docs/training.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Attention Is All You Need (Transformer paper)
- GPT and BERT architectures
- PyTorch community

## 📞 Support

For questions and support, please open an issue on GitHub.
