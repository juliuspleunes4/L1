#!/usr/bin/env python3
"""
@file       : train_minimal.py
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Minimal working training script for L1 model.
@details    : This script provides a minimal setup to train a simple L1 model
              using a basic dataset and configuration. It includes model definition,
              training loop, and evaluation.
@version    : 1.0

@license    : MIT License
Copyright (c) 2025 Julius Pleunes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import math
from typing import List, Dict, Any
import logging
from tqdm import tqdm

# Setup basic warning management
try:
    from warning_manager import setup_training_warnings
    setup_training_warnings("low")  # Minimal warning suppression for debugging
except ImportError:
    pass  # Warning manager not available

# Minimal model configuration
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

# Minimal transformer model
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

# Simple tokenizer
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
        tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, token_ids):
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                    tokens.append(token)
        return ''.join(tokens)

# Simple dataset
class SimpleTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Ensure we have at least 2 tokens
        if len(tokens) < 2:
            tokens = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
        
        # Create input and target (shifted by 1)
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        # Pad to max_length
        pad_length = self.max_length - 1 - len(input_ids)
        if pad_length > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
            target_ids.extend([self.tokenizer.pad_token_id] * pad_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(target_ids, dtype=torch.long)
        }

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def main():
    print("🚀 Starting L1 LLM Training...")
    
    # Load configuration
    config = load_config('configs/train_config.yaml')
    
    # Setup
    output_dir = config['training']['output_dir']
    setup_logging(output_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_path = './data/processed/tokenizer.json'
    if not os.path.exists(tokenizer_path):
        print("Error: Tokenizer not found. Please run data preparation first.")
        return
    
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = SimpleTokenizer(tokenizer_path)
    
    # Copy tokenizer to output directory
    os.makedirs(output_dir, exist_ok=True)
    import shutil
    shutil.copy(tokenizer_path, os.path.join(output_dir, 'tokenizer.json'))
    
    # Create model config
    model_config = L1Config(
        vocab_size=len(tokenizer.vocab),
        max_seq_length=config['model']['max_seq_length'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        n_embd=config['model']['n_embd'],
        n_inner=config['model']['n_inner'],
        dropout=config['model']['dropout'],
        layer_norm_epsilon=config['model']['layer_norm_epsilon'],
        initializer_range=config['model']['initializer_range'],
        use_cache=config['model']['use_cache'],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )
    
    # Create model
    model = L1Model(model_config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load training data
    train_texts = []
    with open(config['data']['train_data_path'], 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                train_texts.append(line)
    
    print(f"Loaded {len(train_texts)} training examples")
    
    # Create dataset and dataloader
    train_dataset = SimpleTextDataset(train_texts, tokenizer, config['data']['max_length'])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=0
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Setup loss function
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Training loop
    model.train()
    num_epochs = config['training']['num_epochs']
    max_steps = config['training'].get('max_steps', None)
    save_steps = config['training']['save_steps']
    logging_steps = config['training']['logging_steps']
    
    global_step = 0
    total_loss = 0
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            
            optimizer.step()
            
            # Update metrics
            global_step += 1
            total_loss += loss.item()
            epoch_loss += loss.item()
            
            # Logging
            if global_step % logging_steps == 0:
                avg_loss = total_loss / logging_steps
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                logging.info(f"Step {global_step}, Loss: {avg_loss:.4f}")
                total_loss = 0
            
            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint-{global_step}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'loss': loss.item(),
                    'config': config
                }, checkpoint_path)
                logging.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Stop if max steps reached
            if max_steps and global_step >= max_steps:
                break
        
        if max_steps and global_step >= max_steps:
            break
        
        logging.info(f"Epoch {epoch+1} completed, Average loss: {epoch_loss/len(train_dataloader):.4f}")
    
    # Save final model
    print("Saving final model...")
    
    # Save model config
    model_config_dict = {
        'vocab_size': model_config.vocab_size,
        'max_seq_length': model_config.max_seq_length,
        'n_layers': model_config.n_layers,
        'n_heads': model_config.n_heads,
        'n_embd': model_config.n_embd,
        'n_inner': model_config.n_inner,
        'dropout': model_config.dropout,
        'layer_norm_epsilon': model_config.layer_norm_epsilon,
        'initializer_range': model_config.initializer_range,
        'use_cache': model_config.use_cache,
        'pad_token_id': model_config.pad_token_id,
        'eos_token_id': model_config.eos_token_id,
        'bos_token_id': model_config.bos_token_id,
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(model_config_dict, f, indent=2)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    
    # Save training config
    with open(os.path.join(output_dir, 'training_args.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Training completed!")
    print(f"📁 Model saved to: {output_dir}")
    print("📄 Files created:")
    print(f"   ├── pytorch_model.bin (model weights)")
    print(f"   ├── config.json (model config)")
    print(f"   ├── tokenizer.json (tokenizer)")
    print(f"   └── training_args.json (training config)")
    print(f"\n🎯 To generate text:")
    print(f"   python generate_simple.py --model_path {output_dir}")

if __name__ == '__main__':
    main()
