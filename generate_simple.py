#!/usr/bin/env python3
"""
Simple text generation script for L1 model.
"""

import os
import torch
import json
import argparse

# Copy the model classes from training script
class L1Config:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 1000)
        self.max_seq_length = kwargs.get('max_seq_length', 512)
        self.n_layers = kwargs.get('n_layers', 6)
        self.n_heads = kwargs.get('n_heads', 8)
        self.n_embd = kwargs.get('n_embd', 512)
        self.n_inner = kwargs.get('n_inner', 2048)
        self.dropout = kwargs.get('dropout', 0.1)
        self.layer_norm_epsilon = kwargs.get('layer_norm_epsilon', 1e-5)
        self.initializer_range = kwargs.get('initializer_range', 0.02)
        self.use_cache = kwargs.get('use_cache', True)
        self.pad_token_id = kwargs.get('pad_token_id', 0)
        self.eos_token_id = kwargs.get('eos_token_id', 3)
        self.bos_token_id = kwargs.get('bos_token_id', 2)

# [Copy the same model classes from train_minimal.py - L1Model, TransformerBlock, etc.]
import torch.nn as nn
import math

class L1Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Output projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states = token_embeds + position_embeds
        
        # Apply transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        return type('ModelOutput', (), {'logits': logits})()

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.ln1(x))
        # MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_heads
        
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.output = nn.Linear(config.n_embd, config.n_embd)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        batch_size, seq_length, n_embd = x.shape
        
        # Get Q, K, V
        q = self.query(x).view(batch_size, seq_length, self.n_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.n_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.n_heads, self.head_size).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        scores.masked_fill_(mask.to(scores.device), float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, n_embd
        )
        
        return self.output(attention_output)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_inner)
        self.fc2 = nn.Linear(config.n_inner, config.n_embd)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleTokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = data['special_tokens']['<pad>']
        self.eos_token_id = data['special_tokens']['<eos>']
        self.bos_token_id = data['special_tokens']['<bos>']
        self.unk_token_id = data['special_tokens']['<unk>']
    
    def encode(self, text):
        # Simple character-level tokenization for demo
        tokens = [self.bos_token_id]
        for char in text.lower():
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.unk_token_id)
        return tokens
    
    def decode(self, token_ids):
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                    tokens.append(token)
        return ''.join(tokens)


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)
    
    generated_ids = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model output
            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]  # Last token
            
            # Apply temperature
            logits = logits / temperature
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Add to sequence
            generated_ids.append(next_token)
            
            # Stop if EOS
            if next_token == tokenizer.eos_token_id:
                break
            
            # Update input tensor
            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=device)
            ], dim=1)
            
            # Trim if too long
            if input_tensor.size(1) > 100:  # Keep last 100 tokens
                input_tensor = input_tensor[:, -100:]
    
    # Decode and return
    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser(description='Generate text with L1 model')
    parser.add_argument('--model_path', type=str, default='./models/l1-v1',
                       help='Path to trained model directory')
    parser.add_argument('--prompt', type=str, default='the',
                       help='Text prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    
    args = parser.parse_args()
    
    print(f"🚀 Loading L1 model from: {args.model_path}")
    
    # Load config
    with open(os.path.join(args.model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    config = L1Config(**config_dict)
    
    # Load tokenizer
    tokenizer = SimpleTokenizer(os.path.join(args.model_path, 'tokenizer.json'))
    
    # Load model
    model = L1Model(config)
    state_dict = torch.load(os.path.join(args.model_path, 'pytorch_model.bin'), 
                           map_location='cpu')
    model.load_state_dict(state_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 Device: {device}")
    print(f"📝 Vocabulary size: {len(tokenizer.vocab)}")
    
    print(f"\n🎭 Generating text with prompt: '{args.prompt}'")
    print("=" * 60)
    
    # Generate text
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    print(generated_text)
    print("=" * 60)
    print(f"🎉 Generated {len(generated_text.split())} words!")


if __name__ == '__main__':
    main()
