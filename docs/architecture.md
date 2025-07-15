# L1 Model Architecture Documentation

## Overview

L1 is a transformer-based large language model implementation built from scratch using PyTorch. The architecture follows the decoder-only transformer design popularized by GPT models.

## Model Components

### 1. L1Config

Configuration class that defines all model hyperparameters:

```python
config = L1Config(
    vocab_size=50257,      # Vocabulary size
    max_seq_length=1024,   # Maximum sequence length
    n_layers=12,           # Number of transformer layers
    n_heads=12,            # Number of attention heads
    n_embd=768,           # Embedding dimension
    n_inner=3072,         # Feed-forward inner dimension
    dropout=0.1,          # Dropout probability
)
```

### 2. Model Architecture

#### Embeddings
- **Token Embedding**: Maps token IDs to dense vectors
- **Positional Embedding**: Adds positional information to token embeddings
- **Support for both learnable and sinusoidal positional encodings**

#### Transformer Blocks
Each transformer block contains:
1. **Multi-Head Self-Attention** with causal masking
2. **Layer Normalization** (pre-norm architecture)
3. **Feed-Forward Network** with GELU activation
4. **Residual connections** for gradient flow

#### Language Modeling Head
- Linear projection from hidden states to vocabulary logits
- Optional weight tying with token embeddings

### 3. Attention Mechanism

#### Multi-Head Attention
- Splits embedding dimension across multiple attention heads
- Computes attention scores using scaled dot-product attention
- Applies causal masking for autoregressive generation
- Uses dropout for regularization

#### Key Features:
- **Causal masking**: Prevents attention to future tokens
- **Key-value caching**: Efficient generation with past context
- **Attention dropout**: Regularization during training

### 4. Feed-Forward Network

Position-wise feed-forward network:
```
Input → Linear(n_embd → n_inner) → GELU → Linear(n_inner → n_embd) → Dropout
```

## Training Architecture

### Loss Function
- **Cross-entropy loss** with label smoothing support
- **Causal language modeling**: Predicts next token given previous context
- **Ignore padding tokens** in loss calculation

### Optimization
- **AdamW optimizer** with weight decay
- **Cosine learning rate scheduling** with warmup
- **Gradient clipping** for training stability
- **Mixed precision training** support (FP16)

### Regularization
- **Dropout** in attention and feed-forward layers
- **Layer normalization** for stable training
- **Weight decay** for parameter regularization
- **Gradient checkpointing** for memory efficiency

## Model Sizes

### Tiny (Demo)
- Layers: 2, Heads: 4, Embedding: 128
- Parameters: ~540K
- Use case: Testing and development

### Small
- Layers: 6, Heads: 6, Embedding: 384  
- Parameters: ~25M
- Use case: Proof of concept, small datasets

### Base
- Layers: 12, Heads: 12, Embedding: 768
- Parameters: ~110M
- Use case: Medium-scale experiments

### Large
- Layers: 24, Heads: 16, Embedding: 1024
- Parameters: ~340M
- Use case: Production-ready model

## Generation

### Text Generation Methods
1. **Greedy decoding**: Always select most likely token
2. **Top-k sampling**: Sample from k most likely tokens
3. **Top-p (nucleus) sampling**: Sample from tokens with cumulative probability p
4. **Temperature scaling**: Control randomness of sampling

### Generation Pipeline
```python
model.generate(
    input_ids=prompt_tokens,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    do_sample=True
)
```

## Memory and Computation

### Memory Requirements
- **Model parameters**: Depends on model size
- **Activations**: Scales with batch_size × seq_length × n_embd
- **Attention**: O(seq_length²) memory for attention matrices

### Optimization Strategies
- **Gradient checkpointing**: Trade computation for memory
- **Attention cache**: Reuse key-value pairs during generation
- **Mixed precision**: Use FP16 for memory efficiency
- **Model parallelism**: Distribute across multiple GPUs

## Implementation Details

### Key Design Decisions
1. **Pre-norm architecture**: Layer norm before attention/FFN
2. **GELU activation**: Better than ReLU for transformers
3. **Learned positional embeddings**: More flexible than fixed
4. **Weight tying**: Share embedding and output layer weights
5. **Causal attention**: Enable autoregressive generation

### Code Structure
```
src/models/
├── __init__.py          # Public API
├── config.py           # Model configuration
├── transformer.py      # Core model implementation
└── embeddings.py       # Embedding layers
```

## Comparison with Other Models

| Feature | L1 | GPT-2 | BERT |
|---------|----|----|------|
| Architecture | Decoder-only | Decoder-only | Encoder-only |
| Attention | Causal | Causal | Bidirectional |
| Training | Autoregressive | Autoregressive | Masked LM |
| Use case | Generation | Generation | Understanding |

## References

1. "Attention Is All You Need" - Vaswani et al. (2017)
2. "Language Models are Unsupervised Multitask Learners" - Radford et al. (2019)
3. "GPT-3: Language Models are Few-Shot Learners" - Brown et al. (2020)
