# L1 Development Environment Setup

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended)
- Git

## Setup Instructions

1. **Create virtual environment**:
   ```bash
   python -m venv l1_env
   ```

2. **Activate virtual environment**:
   - Windows: `l1_env\Scripts\activate`
   - Linux/Mac: `source l1_env/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

## Quick Start

1. **Prepare sample data**:
   ```bash
   python scripts/prepare_data.py --input data/raw/sample_text.txt --output data/processed/
   ```

2. **Train model**:
   ```bash
   python scripts/train.py --config configs/base_config.yaml
   ```

3. **Generate text**:
   ```bash
   python scripts/generate.py --model checkpoints/best_model.pt --prompt "Hello, world!"
   ```

## Development Guidelines

- Run tests: `python -m pytest tests/ -v`
- Format code: `black src/ scripts/ tests/`
- Type checking: `mypy src/`
- Lint code: `flake8 src/ scripts/ tests/`

## Project Structure

```
L1/
├── src/              # Source code
├── configs/          # Configuration files  
├── scripts/          # Training and inference scripts
├── data/            # Data directory
├── tests/           # Unit tests
├── checkpoints/     # Model checkpoints
└── logs/            # Training logs
```
