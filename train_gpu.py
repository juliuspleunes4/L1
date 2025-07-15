#!/usr/bin/env python3
"""
GPU-optimized training script for L1 model.
Enhanced for powerful GPU setups with larger datasets.
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
import time

# Setup warning management
try:
    from warning_manager import setup_training_warnings
    setup_training_warnings("medium")  # Suppress non-critical warnings
except ImportError:
    print("ℹ️  Warning manager not found - some non-critical warnings may appear")

# Import the core model components from train_minimal
from train_minimal import L1Config, L1Model, SimpleTokenizer, SimpleTextDataset

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

def get_gpu_info():
    """Get information about available GPUs"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / (1024**3)
        
        print(f"🎮 GPU Information:")
        print(f"   ├── Available GPUs: {gpu_count}")
        print(f"   ├── Current GPU: {gpu_name}")
        print(f"   ├── GPU Memory: {gpu_memory:.1f} GB")
        print(f"   └── CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("❌ No GPU available, using CPU")
        return False

def optimize_for_gpu(model, config, device):
    """Apply GPU-specific optimizations"""
    optimizations_applied = []
    is_cuda = device.type == 'cuda'
    
    # Enable gradient checkpointing for memory efficiency
    if config.get('performance', {}).get('gradient_checkpointing', False):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            optimizations_applied.append("Gradient Checkpointing")
    
    # Compile model for better performance (PyTorch 2.0+) - only if no warnings
    compile_model = config.get('performance', {}).get('compile_model', False)
    if compile_model and is_cuda:  # Only compile on GPU to avoid C++ compiler issues
        try:
            model = torch.compile(model)
            optimizations_applied.append("Model Compilation")
        except Exception as e:
            print(f"Warning: Could not compile model (this is okay): {e}")
    elif compile_model and not is_cuda:
        print("ℹ️  Model compilation disabled on CPU (avoiding C++ compiler requirement)")
    
    # Enable mixed precision training - use updated API
    use_amp = config.get('training', {}).get('mixed_precision', False) and is_cuda
    scaler = None
    
    if use_amp:
        try:
            # Use the new API format
            scaler = torch.amp.GradScaler('cuda')
            optimizations_applied.append("Mixed Precision (AMP)")
        except Exception:
            # Fallback for older PyTorch versions
            try:
                scaler = torch.cuda.amp.GradScaler()
                optimizations_applied.append("Mixed Precision (AMP)")
            except Exception as e:
                print(f"Warning: Could not enable mixed precision: {e}")
                use_amp = False
    elif config.get('training', {}).get('mixed_precision', False) and not is_cuda:
        print("ℹ️  Mixed precision disabled on CPU (CUDA required)")
    
    if optimizations_applied:
        print(f"🔧 Optimizations Applied: {', '.join(optimizations_applied)}")
    elif is_cuda:
        print("ℹ️  No optimizations applied (check config settings)")
    else:
        print("ℹ️  GPU optimizations disabled (CPU training)")
    
    return model, scaler, use_amp

def calculate_model_size(model):
    """Calculate model size and parameter breakdown"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Parameter breakdown
    param_breakdown = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') or hasattr(module, 'bias'):
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                param_breakdown[name] = module_params
    
    # Model size in MB
    param_size = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
    
    print(f"📊 Model Statistics:")
    print(f"   ├── Total Parameters: {total_params:,}")
    print(f"   ├── Trainable Parameters: {trainable_params:,}")
    print(f"   ├── Model Size: {param_size:.1f} MB")
    print(f"   └── Memory Usage (training): ~{param_size * 4:.1f} MB")
    
    return total_params, param_breakdown

def load_large_dataset(data_path: str, max_samples: int = None):
    """Load dataset with support for very large files"""
    print(f"📚 Loading dataset from {data_path}")
    
    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if line:
                texts.append(line)
            
            # Progress update for large files
            if i % 100000 == 0 and i > 0:
                print(f"   Loaded {i:,} samples...")
    
    print(f"✅ Dataset loaded: {len(texts):,} samples")
    return texts

def main():
    print("🚀 Starting L1 GPU-Optimized Training...")
    print("="*60)
    
    # Load configuration
    config_path = 'configs/train_config_gpu.yaml'
    if not os.path.exists(config_path):
        print(f"GPU config not found, using default config...")
        config_path = 'configs/train_config.yaml'
    
    config = load_config(config_path)
    
    # Setup
    output_dir = config['training']['output_dir']
    setup_logging(output_dir)
    
    # GPU setup and information
    has_gpu = get_gpu_info()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Training device: {device}")
    print("="*60)
    
    # Load tokenizer
    tokenizer_path = './data/processed/tokenizer.json'
    if not os.path.exists(tokenizer_path):
        print("Error: Tokenizer not found. Please run data preparation first.")
        return
    
    print(f"📝 Loading tokenizer from {tokenizer_path}")
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
    print("🏗️  Creating model...")
    model = L1Model(model_config).to(device)
    
    # Calculate model statistics
    total_params, param_breakdown = calculate_model_size(model)
    
    # Apply GPU optimizations
    model, scaler, use_amp = optimize_for_gpu(model, config, device)
    print("="*60)
    
    # Load training data
    print("📊 Loading training data...")
    max_samples = config.get('data', {}).get('max_samples', None)
    train_texts = load_large_dataset(config['data']['train_data_path'], max_samples)
    
    # Create dataset and dataloader
    train_dataset = SimpleTextDataset(train_texts, tokenizer, config['data']['max_length'])
    
    # Enhanced dataloader for GPU training
    num_workers = config.get('training', {}).get('dataloader_num_workers', 0)
    # Reduce workers on CPU to avoid overhead
    if device.type == 'cpu' and num_workers > 2:
        num_workers = 0
        print("ℹ️  Reduced dataloader workers for CPU training")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),  # Only pin memory for GPU
        persistent_workers=(num_workers > 0)
    )
    
    print(f"📦 DataLoader configured:")
    print(f"   ├── Batch size: {config['training']['batch_size']}")
    print(f"   ├── Workers: {num_workers}")
    print(f"   └── Pin memory: {device.type == 'cuda'}")
    print("="*60)
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Setup loss function
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Training configuration
    model.train()
    num_epochs = config['training']['num_epochs']
    max_steps = config['training'].get('max_steps', None)
    save_steps = config['training']['save_steps']
    logging_steps = config['training']['logging_steps']
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    use_amp = config['training'].get('mixed_precision', False)
    
    global_step = 0
    total_loss = 0
    best_loss = float('inf')
    
    print(f"🎓 Training Configuration:")
    print(f"   ├── Epochs: {num_epochs}")
    print(f"   ├── Max steps: {max_steps or 'No limit'}")
    print(f"   ├── Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   ├── Mixed precision: {use_amp}")
    print(f"   └── Save every: {save_steps} steps")
    print("="*60)
    
    print("🏁 Starting training...")
    training_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_start_time = time.time()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            if use_amp and scaler and device.type == 'cuda':
                try:
                    # Use the new API format
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids)
                        logits = outputs.logits
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                        loss = loss / gradient_accumulation_steps
                except Exception:
                    # Fallback for older PyTorch versions
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids)
                        logits = outputs.logits
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                        loss = loss / gradient_accumulation_steps
            else:
                outputs = model(input_ids)
                logits = outputs.logits
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if use_amp and scaler and device.type == 'cuda':
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp and scaler and device.type == 'cuda':
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                    optimizer.step()
                
                optimizer.zero_grad()
                global_step += 1
            
            # Update metrics
            total_loss += loss.item() * gradient_accumulation_steps
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Logging
            if global_step % logging_steps == 0:
                avg_loss = total_loss / logging_steps
                
                # GPU memory info (only if on GPU)
                if device.type == 'cuda':
                    memory_used = torch.cuda.memory_allocated() / (1024**3)
                    memory_cached = torch.cuda.memory_reserved() / (1024**3)
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'GPU': f'{memory_used:.1f}GB'
                    })
                else:
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                logging.info(f"Step {global_step}, Loss: {avg_loss:.4f}")
                total_loss = 0
            
            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint-{global_step}.pt')
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'loss': loss.item(),
                    'config': config
                }
                if scaler:
                    checkpoint_data['scaler_state_dict'] = scaler.state_dict()
                
                torch.save(checkpoint_data, checkpoint_path)
                logging.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Stop if max steps reached
            if max_steps and global_step >= max_steps:
                break
        
        if max_steps and global_step >= max_steps:
            break
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        
        print(f"✅ Epoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f"   └── Average loss: {avg_epoch_loss:.4f}")
        
        logging.info(f"Epoch {epoch+1} completed, Average loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.1f}s")
    
    training_time = time.time() - training_start_time
    print("="*60)
    print(f"🎉 Training completed in {training_time/60:.1f} minutes!")
    
    # Save final model
    print("💾 Saving final model...")
    
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
    
    # Save training config with performance metrics
    training_summary = config.copy()
    training_summary['training_summary'] = {
        'total_parameters': total_params,
        'training_time_minutes': training_time / 60,
        'final_step': global_step,
        'device_used': str(device),
        'gpu_info': torch.cuda.get_device_name(0) if has_gpu else 'CPU'
    }
    
    with open(os.path.join(output_dir, 'training_args.json'), 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"✅ Training completed!")
    print(f"📁 Model saved to: {output_dir}")
    print("📄 Files created:")
    print(f"   ├── pytorch_model.bin ({total_params:,} parameters)")
    print(f"   ├── config.json (model config)")
    print(f"   ├── tokenizer.json (tokenizer)")
    print(f"   └── training_args.json (training config + metrics)")
    print(f"\n🎯 To generate text:")
    print(f"   python generate_simple.py --model_path {output_dir}")
    
    if device.type == 'cuda':
        print(f"\n📈 GPU Training Stats:")
        print(f"   ├── Peak GPU Memory: {torch.cuda.max_memory_allocated() / (1024**3):.1f} GB")
        print(f"   └── Training Speed: {global_step / (training_time / 60):.1f} steps/minute")
    else:
        print(f"\n📈 CPU Training Stats:")
        print(f"   └── Training Speed: {global_step / (training_time / 60):.1f} steps/minute")

if __name__ == '__main__':
    main()
