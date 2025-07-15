# Training Guide for L1 Model

## Overview

This guide covers the complete training pipeline for the L1 language model, from data preparation to model deployment.

## Data Preparation

### 1. Data Format

L1 expects text data in simple format:
- One sample per line in `.txt` files
- UTF-8 encoding
- Clean, tokenizable text

Example:
```
The quick brown fox jumps over the lazy dog.
Artificial intelligence is transforming the world.
Large language models demonstrate emergent capabilities.
```

### 2. Data Preprocessing

Use the data preparation script:

```bash
python scripts/prepare_data.py \
    --input data/raw/your_text.txt \
    --output data/processed/ \
    --vocab-size 50257 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --seed 42
```

This will:
- Split data into train/validation/test sets
- Train a BPE tokenizer on the training data
- Save processed data and tokenizer

### 3. Data Requirements

**Minimum Dataset Size:**
- Tiny model: 1M+ tokens
- Small model: 10M+ tokens  
- Base model: 100M+ tokens
- Large model: 1B+ tokens

**Quality Guidelines:**
- Remove duplicate content
- Filter out low-quality text
- Ensure diverse domain coverage
- Clean formatting artifacts

## Training Configuration

### 1. Model Configuration

Edit `configs/base_config.yaml`:

```yaml
model:
  vocab_size: 50257
  max_seq_length: 1024
  n_layers: 12
  n_heads: 12
  n_embd: 768
  dropout: 0.1
```

### 2. Training Configuration

```yaml
training:
  num_epochs: 3
  batch_size: 8
  learning_rate: 5e-4
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Hardware settings
  device: "auto"
  fp16: false
  gradient_checkpointing: false
```

### 3. Configuration Examples

#### Quick Testing (Tiny Model)
```yaml
model:
  n_layers: 2
  n_heads: 4
  n_embd: 128
training:
  batch_size: 4
  num_epochs: 1
  max_steps: 100
```

#### Production Training (Base Model)
```yaml
model:
  n_layers: 12
  n_heads: 12
  n_embd: 768
training:
  batch_size: 16
  num_epochs: 3
  gradient_checkpointing: true
  fp16: true
```

## Training Process

### 1. Start Training

```bash
python scripts/train.py --config configs/base_config.yaml
```

### 2. Resume Training

```bash
python scripts/train.py \
    --config configs/base_config.yaml \
    --resume checkpoints/checkpoint-1000.pt
```

### 3. Monitor Training

**Tensorboard:**
```bash
tensorboard --logdir checkpoints/tensorboard
```

**Weights & Biases:**
Set in config:
```yaml
training:
  wandb_project: "my-l1-experiment"
  wandb_run_name: "base-model-v1"
```

### 4. Training Metrics

Monitor these key metrics:
- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should track training loss
- **Perplexity**: Lower is better
- **Learning Rate**: Follow schedule
- **Gradient Norm**: Should be stable

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+ 
- **Storage**: 10GB+ free space
- **GPU**: Optional but recommended

### Recommended Setup
- **GPU**: NVIDIA RTX 3080+ or equivalent
- **VRAM**: 12GB+ for base model
- **RAM**: 32GB+ for large datasets
- **Storage**: SSD with 50GB+ free space

### Multi-GPU Training

For multiple GPUs, use PyTorch DDP:
```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/train.py \
    --config configs/base_config.yaml
```

## Training Best Practices

### 1. Learning Rate

**Finding optimal LR:**
- Start with 1e-4 to 1e-3
- Use learning rate finder
- Apply warmup (500-2000 steps)
- Use cosine decay

### 2. Batch Size

**Guidelines:**
- Larger batch sizes → more stable training
- Limited by GPU memory
- Use gradient accumulation if needed
- Typical range: 8-64 per GPU

### 3. Sequence Length

**Trade-offs:**
- Longer sequences → better context
- Memory usage scales quadratically
- Start with 512, increase gradually
- Maximum: 2048 for most hardware

### 4. Regularization

**Prevent overfitting:**
- Dropout: 0.1-0.2
- Weight decay: 0.01-0.1
- Early stopping on validation loss
- Data augmentation

## Troubleshooting

### Common Issues

**1. Training Loss Not Decreasing**
- Check learning rate (too high/low)
- Verify data quality
- Check for gradient clipping
- Ensure model convergence

**2. Out of Memory Errors**
- Reduce batch size
- Enable gradient checkpointing
- Use FP16 training
- Reduce sequence length

**3. Training Instability**
- Lower learning rate
- Increase warmup steps
- Check gradient norms
- Use gradient clipping

**4. Slow Training Speed**
- Use FP16 if available
- Optimize data loading
- Enable gradient checkpointing
- Use appropriate batch size

### Performance Tips

**Speed Optimization:**
- Use compiled models (torch.compile)
- Optimize data pipeline
- Pin memory for data loaders
- Use multiple workers

**Memory Optimization:**
- Gradient checkpointing
- Mixed precision training
- Optimize attention computation
- Use efficient optimizers

## Evaluation

### During Training

**Automatic Evaluation:**
- Validation loss every N steps
- Perplexity calculation
- Gradient norm monitoring
- Learning rate tracking

### Post-Training

**Model Quality:**
```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --data data/processed/test.txt
```

**Text Generation:**
```bash
python scripts/generate.py \
    --model checkpoints/best_model.pt \
    --prompt "The future of AI is" \
    --max-tokens 100
```

## Next Steps

After training:
1. **Evaluate** model performance
2. **Fine-tune** on specific tasks
3. **Deploy** for inference
4. **Monitor** in production
5. **Iterate** on architecture/data

## Advanced Topics

### 1. Curriculum Learning
- Start with shorter sequences
- Gradually increase difficulty
- Domain-specific ordering

### 2. Transfer Learning
- Start from pretrained checkpoint
- Adjust learning rates per layer
- Freeze certain parameters

### 3. Model Pruning
- Remove unnecessary parameters
- Quantization for efficiency
- Knowledge distillation

### 4. Distributed Training
- Data parallelism
- Model parallelism  
- Pipeline parallelism
