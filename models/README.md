# L1 Trained Models

This directory contains your trained L1 language models. Each model version is stored in its own subdirectory.

## Directory Structure

```
models/
├── l1-v1/                     # Your first trained model
│   ├── config.json            # Model architecture configuration
│   ├── pytorch_model.bin      # Trained model weights
│   ├── tokenizer.json         # Tokenizer vocabulary and rules
│   ├── training_args.json     # Training configuration used
│   ├── checkpoint-*/          # Training checkpoints
│   └── README.md              # Model-specific documentation
└── README.md                  # This file
```

## Using Your Trained Models

### Generate Text
```bash
# Generate text with default prompt
python scripts/generate_new.py --model_path models/l1-v1/

# Generate with custom prompt
python scripts/generate_new.py --model_path models/l1-v1/ --prompt "The future of AI is"

# Adjust generation parameters
python scripts/generate_new.py --model_path models/l1-v1/ \
    --prompt "Once upon a time" \
    --max_new_tokens 100 \
    --temperature 0.8 \
    --top_k 40 \
    --top_p 0.9
```

### Model Information

Each model directory contains:

- **config.json**: Model architecture and configuration parameters
- **pytorch_model.bin**: The actual trained model weights
- **tokenizer.json**: Vocabulary and tokenization rules 
- **training_args.json**: Training hyperparameters and settings used

## Generation Parameters

- **temperature**: Controls randomness (0.1 = conservative, 1.0 = creative)
- **top_k**: Consider only top K most likely tokens (40-50 recommended)
- **top_p**: Nuclear sampling threshold (0.9 recommended)
- **max_new_tokens**: Maximum number of tokens to generate

## Model Versions

### l1-v1
- **Architecture**: 6 layers, 8 heads, 512 embedding dimension
- **Vocabulary Size**: ~1000 tokens
- **Training Data**: Sample text data
- **Use Case**: Proof of concept and learning

Add more model versions as you train them!
