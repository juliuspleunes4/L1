#!/usr/bin/env python3
"""
Simple training script for L1 model that works around import issues.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
from typing import List, Dict, Any
import logging
from tqdm import tqdm

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
sys.path.insert(0, src_dir)

# Import individual modules
exec(open(os.path.join(src_dir, 'models', 'config.py')).read())
exec(open(os.path.join(src_dir, 'models', 'embeddings.py')).read())
exec(open(os.path.join(src_dir, 'models', 'transformer.py')).read())
exec(open(os.path.join(src_dir, 'data', 'tokenizer.py')).read())

# Simple dataset class
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
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input and target (shifted by 1)
        if len(tokens) < 2:
            # Skip too short sequences
            tokens = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
        
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
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(output_dir: str):
    """Setup logging."""
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
    # Load configuration
    config = load_config('configs/train_config.yaml')
    
    # Setup
    output_dir = config['training']['output_dir']
    setup_logging(output_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or create tokenizer
    tokenizer_path = './data/processed/tokenizer.json'
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print("Error: Tokenizer not found. Please run data preparation first.")
        return
    
    # Copy tokenizer to output directory
    os.makedirs(output_dir, exist_ok=True)
    output_tokenizer_path = os.path.join(output_dir, 'tokenizer.json')
    tokenizer.save(output_tokenizer_path)
    
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
        num_workers=0  # Windows compatibility
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
    
    print(f"Training completed!")
    print(f"Model saved to: {output_dir}")
    print("Files created:")
    print(f"  - {output_dir}/pytorch_model.bin (model weights)")
    print(f"  - {output_dir}/config.json (model config)")
    print(f"  - {output_dir}/tokenizer.json (tokenizer)")
    print(f"  - {output_dir}/training_args.json (training config)")
    print(f"\nTo generate text:")
    print(f"  python scripts/generate_new.py --model_path {output_dir}")


if __name__ == '__main__':
    main()
