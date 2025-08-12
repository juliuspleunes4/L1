#!/usr/bin/env python3
"""
@file       : train_gpu_compatible.py
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : GPU-compatible training script for L1 model.
@details    : This script is designed to train the L1 model on GPUs, specifically optimized
              for RTX 5060 Ti (sm_120) compatibility issues. Works on the 40 and 50 series.
@version    : 2.3

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
import glob
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import time
from datetime import datetime

# Force PyTorch to suppress dynamo errors and fallback to eager execution
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Setup warning management
try:
    from warning_manager import setup_training_warnings
    setup_training_warnings("low")  # Show more info for debugging
except ImportError:
    print("ℹ️  Warning manager not found - some warnings may appear")

# Import the core model components from train_minimal
from train_minimal import L1Config, L1Model, SimpleTextDataset
from src.data.tokenizer import BPETokenizer

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(output_dir: str):
    """Setup logging configuration with detailed formatting.

    Args:
        output_dir: Directory where training.log will be created
        
    Returns:
        Logger instance configured for l1_training
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a specific logger for this module
    logger = logging.getLogger('l1_training')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create a custom formatter for the log file
    log_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler with custom formatter
    log_file_path = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    
    # Setup console handler with simpler format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    # Add handlers to our specific logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_gpu_info():
    """Get information about available GPUs"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / (1024**3)
        return {
            'count': gpu_count,
            'current': current_gpu,
            'name': gpu_name,
            'memory_gb': gpu_memory
        }
    return None

def log_training_metrics(step: int, epoch: int, loss: float, learning_rate: float, best_loss: float, is_best: bool = False, additional_info: str = "", avg_loss: float = None):
    """Log detailed training metrics to the training log.

    Args:
        step: Current training step
        epoch: Current epoch number
        loss: Current loss value
        learning_rate: Current learning rate
        best_loss: Best loss achieved so far
        is_best: Whether this is a new best checkpoint
        additional_info: Extra information to append to log entry
        avg_loss: Average loss over processed batches (if available)
    """
    # Get the specific logger for training
    logger = logging.getLogger('l1_training')
    
    # Create a comprehensive log entry (timestamp will be added by the logging formatter)
    log_type = "BEST" if is_best else "CHECKPOINT"
    log_entry = f"{log_type} | Epoch: {epoch} | Step: {step} | Loss: {loss:.6f}"
    
    # Add avg_loss if provided
    if avg_loss is not None:
        log_entry += f" | Avg_Loss: {avg_loss:.6f}"
    
    log_entry += f" | Best: {best_loss:.6f} | LR: {learning_rate:.2e}"
    
    if additional_info:
        log_entry += f" | {additional_info}"
    
    logger.info(log_entry)

def apply_model_optimizations(model: nn.Module, config: dict, device: torch.device) -> List[str]:
    """Apply various optimizations to the model - compatible version"""
    optimizations_applied = []
    is_cuda = device.type == 'cuda'
    
    print("🔧 Applying GPU compatibility optimizations...")
    
    # Enable model compilation now that we have proper CUDA support
    compile_model = config.get('performance', {}).get('compile_model', True) and is_cuda
    if compile_model:
        try:
            model = torch.compile(model)
            optimizations_applied.append("Model Compilation")
            print("✅ Model compilation enabled")
        except Exception as e:
            print(f"⚠️  Model compilation failed: {e}")
    else:
        print("ℹ️  Model compilation disabled")
    
    # Enable mixed precision training with conservative approach
    use_amp = config.get('training', {}).get('mixed_precision', False) and is_cuda
    scaler = None
    
    if use_amp:
        try:
            # Use the modern API with proper CUDA 12.8 support
            scaler = torch.amp.GradScaler('cuda')
            optimizations_applied.append("Mixed Precision (AMP)")
            print("✅ Mixed precision enabled with modern API")
        except Exception as e:
            print(f"⚠️  Mixed precision disabled due to error: {e}")
            use_amp = False
    
    # Gradient checkpointing for memory efficiency (force enable for RTX 5060 Ti)
    gradient_checkpointing = config.get('training', {}).get('gradient_checkpointing', True)  # Default to True
    if hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
            optimizations_applied.append("Gradient Checkpointing")
            print("✅ Gradient checkpointing enabled (memory optimization)")
        except Exception as e:
            print(f"⚠️  Gradient checkpointing failed: {e}")
    else:
        print("ℹ️  Gradient checkpointing not available for this model")
    
    print(f"✅ Optimizations applied: {', '.join(optimizations_applied) if optimizations_applied else 'None (running in safe mode)'}")
    
    return optimizations_applied, scaler, use_amp

def calculate_model_size(model: nn.Module) -> Dict[str, Any]:
    """Calculate model statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    # Estimate memory usage during training (rough approximation)
    training_memory_mb = model_size_mb * 4  # Model + gradients + optimizer states + activations
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'estimated_training_memory_mb': training_memory_mb
    }

def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Create optimizer with the specified configuration"""
    optimizer_config = config.get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'adamw')
    learning_rate = optimizer_config.get('learning_rate', 5e-4)
    weight_decay = optimizer_config.get('weight_decay', 0.1)
    
    if optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),  # Common values for language models
            eps=1e-8
        )
    elif optimizer_type.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def create_scheduler(optimizer: optim.Optimizer, config: dict, total_steps: int):
    """Create learning rate scheduler"""
    scheduler_config = config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine')
    
    if scheduler_type == 'cosine':
        warmup_steps = int(total_steps * scheduler_config.get('warmup_ratio', 0.1))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif scheduler_type == 'linear':
        warmup_steps = int(total_steps * scheduler_config.get('warmup_ratio', 0.1))
        return optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    else:
        return None

def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: dict,
    scaler=None,
    use_amp: bool = False,
    scheduler=None,
    save_checkpoint_fn=None,
    output_dir: str = None,
    model_config=None,  # Add model_config parameter
    resumed_global_step: int = 0,  # Add resumed step parameter
    best_checkpoint_loss: float = float('inf')  # Add best checkpoint loss parameter
) -> Dict[str, float]:
    """Train for one epoch - compatible version with frequent checkpointing.
    
    Args:
        model: The neural network model to train
        dataloader: DataLoader containing training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to run training on (CPU/GPU)
        epoch: Current epoch number
        config: Training configuration dictionary
        scaler: GradScaler for mixed precision training
        use_amp: Whether to use automatic mixed precision
        scheduler: Learning rate scheduler
        save_checkpoint_fn: Function to save checkpoints
        output_dir: Directory to save checkpoints
        model_config: Model configuration object
        resumed_global_step: Step number when resuming training
        best_checkpoint_loss: Best checkpoint loss achieved so far (session-specific, 
                            may not reflect global best across all training sessions)
        
    Returns:
        Dictionary containing loss, learning_rate, and best_checkpoint_loss
        
    Note:
        The best_checkpoint_loss tracking is session-specific and may not reflect
        the true global best if training is resumed from a checkpoint that had
        a different best loss than the current session's starting point.
    """
    model.train()
    num_batches = len(dataloader)
    
    # Validate dataloader is not empty
    if num_batches == 0:
        raise ValueError("Dataloader is empty! Cannot train with no data batches.")
    
    # If resuming, initialize total_loss to reflect previous progress
    if resumed_global_step > 0 and hasattr(train_epoch, "previous_epoch_loss"):
        total_loss = train_epoch.previous_epoch_loss * resumed_global_step
        processed_batches = resumed_global_step
    else:
        total_loss = 0.0
        processed_batches = 0
    
    # Training configuration
    gradient_accumulation_steps = config.get('training', {}).get('gradient_accumulation_steps', 1)
    max_grad_norm = config.get('training', {}).get('max_grad_norm', 1.0)
    
    # Checkpoint every N steps to save progress during long epochs
    # For local training: 100 steps (~18 minutes) provides excellent safety with minimal overhead
    checkpoint_every_steps = config.get('training', {}).get('checkpoint_every_steps', 100)  # Default: every 100 steps (~18 minutes)
    
    # If resuming, skip batches already processed
    from itertools import islice
    start_batch = resumed_global_step if resumed_global_step > 0 else 0
    total_batches = len(dataloader)
    progress_bar = tqdm(
        islice(dataloader, start_batch, total_batches),
        desc=f"Epoch {epoch}",
        leave=False,
        initial=start_batch,
        total=total_batches
    )
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar, start=start_batch):
        try:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass with modern mixed precision API
            if use_amp and scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids)
                    logits = outputs.logits
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss = loss / gradient_accumulation_steps
                
                # Scale loss for mixed precision
                scaler.scale(loss).backward()
            else:
                # Standard forward pass without mixed precision
                outputs = model(input_ids)
                logits = outputs.logits
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp and scaler:
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
            
            total_loss += loss.item() * gradient_accumulation_steps
            processed_batches += 1
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / (processed_batches if processed_batches > 0 else 1)
            global_step = batch_idx + 1
            progress_bar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'global_step': f'{global_step}/{num_batches}'
            })
            
            # Save checkpoint periodically during training
            if (global_step) % checkpoint_every_steps == 0 and save_checkpoint_fn and output_dir:
                # Calculate step correctly considering resumption
                if resumed_global_step > 0:
                    # If we resumed, calculate from the resumed step
                    current_step = global_step
                else:
                    # Normal calculation for fresh training
                    current_step = (epoch - 1) * num_batches + global_step
                current_loss = total_loss / max(global_step, 1)  # Division by zero is prevented this way 
                
                # Check if this is the best loss so far (using explicit parameter for training state)
                is_best_checkpoint = current_loss < best_checkpoint_loss
                if is_best_checkpoint:
                    best_checkpoint_loss = current_loss
                    print(f"\n🏆 NEW BEST LOSS! Saving best checkpoint at step {current_step} (loss: {current_loss:.4f})...")
                    # Log new best checkpoint with detailed metrics including avg_loss
                    current_lr = optimizer.param_groups[0]['lr']
                    log_training_metrics(current_step, epoch, current_loss, current_lr, best_checkpoint_loss, is_best=True, avg_loss=avg_loss)
                else:
                    print(f"\n💾 Saving progress checkpoint at step {current_step} (loss: {current_loss:.4f}, best: {best_checkpoint_loss:.4f})...")
                    # Log regular checkpoint with detailed metrics including avg_loss
                    current_lr = optimizer.param_groups[0]['lr']
                    log_training_metrics(current_step, epoch, current_loss, current_lr, best_checkpoint_loss, is_best=False, avg_loss=avg_loss)
                
                save_checkpoint_fn(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    step=current_step,
                    loss=current_loss,
                    save_dir=output_dir,
                    is_best=is_best_checkpoint,
                    model_config=model_config
                )
            
        except RuntimeError as e:
            if "CUDA error" in str(e):
                print(f"\n⚠️  CUDA error encountered: {e}")
                print("💡 Falling back to CPU execution for this batch...")
                # Move batch to CPU as fallback
                input_ids = batch['input_ids'].to('cpu')
                labels = batch['labels'].to('cpu')
                model_cpu = model.to('cpu')
                
                outputs = model_cpu(input_ids)
                logits = outputs.logits
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # Move model back to GPU
                model.to(device)
                
                total_loss += loss.item() * gradient_accumulation_steps
            else:
                raise e
    
    # Store the running loss for future resumption
    final_avg_loss = total_loss / (processed_batches if processed_batches > 0 else 1)
    train_epoch.previous_epoch_loss = final_avg_loss
    return {
        'loss': total_loss / num_batches,  # Safe division since we validated num_batches > 0
        'avg_loss': final_avg_loss,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'best_checkpoint_loss': best_checkpoint_loss
    }

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: dict,
    epoch: int,
    step: int,
    loss: float,
    save_dir: str,
    is_best: bool = False,
    model_config=None  
):
    """Save training checkpoint with automatic cleanup of old checkpoints"""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'best_loss': loss if is_best else None
    }
    
    # Save the new checkpoint
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Always save as latest checkpoint for resuming
    latest_path = os.path.join(save_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)
    
    # Save as best checkpoint if this is the best loss
    if is_best:
        best_path = os.path.join(save_dir, 'best_checkpoint.pt')
        torch.save(checkpoint, best_path)
        print(f"🏆 NEW BEST CHECKPOINT! Loss: {loss:.4f} (saved as best_checkpoint.pt)")
        
        # Log best checkpoint save
        logger = logging.getLogger('l1_training')
        logger.info(f"BEST CHECKPOINT SAVED - File: {checkpoint_path}, Loss: {loss:.4f}, Epoch: {epoch}, Step: {step}")
        
        # Also save in format compatible with generate_simple.py
        try:
            # Save model state dict as pytorch_model.bin
            model_bin_path = os.path.join(save_dir, 'pytorch_model.bin')
            torch.save(model.state_dict(), model_bin_path)
            
            # Save config as config.json
            config_json_path = os.path.join(save_dir, 'config.json')
            if model_config:
                # Use the actual model configuration
                model_config_dict = {
                    'vocab_size': model_config.vocab_size,
                    'max_seq_length': model_config.max_seq_length,
                    'n_layers': model_config.n_layers,
                    'n_heads': model_config.n_heads,
                    'n_embd': model_config.n_embd,
                    'n_inner': model_config.n_inner
                }
            else:
                # Fallback to config defaults
                model_config_dict = {
                    'vocab_size': config.get('model', {}).get('vocab_size', 20000),
                    'max_seq_length': config.get('model', {}).get('max_seq_length', 1024),
                    'n_layers': config.get('model', {}).get('n_layers', 12),
                    'n_heads': config.get('model', {}).get('n_heads', 16),
                    'n_embd': config.get('model', {}).get('n_embd', 1024),
                    'n_inner': config.get('model', {}).get('n_inner', 4096)
                }
            
            with open(config_json_path, 'w') as f:
                json.dump(model_config_dict, f, indent=2)
            
            print(f"📄 Best model saved in generation format (pytorch_model.bin + config.json)")
        except Exception as e:
            print(f"⚠️  Failed to save generation format: {e}")
    else:
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Auto-cleanup: Keep only the last N regular checkpoints (excluding best/latest)
    max_checkpoints = config.get('training', {}).get('max_checkpoints_to_keep', 5)  # Keep last 5 checkpoints
    cleanup_old_checkpoints(save_dir, max_checkpoints)

def cleanup_old_checkpoints(save_dir: str, max_to_keep: int = 5):
    """Remove old checkpoint files, keeping only the most recent ones"""
    try:
        # Find all checkpoint files (excluding best_checkpoint.pt and latest_checkpoint.pt)
        checkpoint_pattern = os.path.join(save_dir, 'checkpoint_epoch_*_step_*.pt')
        import glob
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) > max_to_keep:
            # Sort by modification time (newest first)
            checkpoint_files.sort(key=os.path.getmtime, reverse=True)
            
            # Remove old checkpoints (keep only the newest max_to_keep)
            files_to_remove = checkpoint_files[max_to_keep:]
            
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    filename = os.path.basename(file_path)
                    print(f"🗑️  Cleaned up old checkpoint: {filename}")
                except Exception as e:
                    print(f"⚠️  Failed to remove {file_path}: {e}")
    except Exception as e:
        print(f"⚠️  Checkpoint cleanup failed: {e}")

def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    """Load training checkpoint and resume training"""
    print(f"📥 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    step = checkpoint['step'] 
    loss = checkpoint['loss']
    
    print(f"✅ Resumed from epoch {epoch}, step {step}, loss: {loss:.4f}")
    return epoch, step, loss

def main():
    """Main training function"""
    print("🚀 Starting L1 GPU Training with RTX 5060 Ti Support!")
    print("="*60)
    
    # Load configuration
    config_path = "./configs/train_config_gpu.yaml"
    if not os.path.exists(config_path):
        config_path = "./configs/train_config.yaml"
    
    config = load_config(config_path)
    
    # Get device information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_info = get_gpu_info()
    
    if gpu_info:
        print("🎮 GPU Information:")
        print(f"   ├── Available GPUs: {gpu_info['count']}")
        print(f"   ├── Current GPU: {gpu_info['name']}")
        print(f"   ├── GPU Memory: {gpu_info['memory_gb']:.1f} GB")
        print(f"   └── CUDA Version: {torch.version.cuda}")
    
    print(f"🎯 Training device: {device}")
    print("="*60)
    
    # Load tokenizer
    tokenizer_path = config.get('data', {}).get('tokenizer_path', './data/processed/tokenizer.json')
    print(f"📝 Loading BPE tokenizer from {tokenizer_path}")
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        print("💡 Please run: python prepare_large_dataset.py <your_data_file> --vocab-size 32000")
        return
    
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"✅ Loaded BPE tokenizer with {len(tokenizer.vocab)} tokens")
    
    # Calculate vocab size
    vocab_size = len(tokenizer.vocab)
    tokenizer.vocab_size = vocab_size  # Add vocab_size attribute
    
    # Create model configuration with memory optimizations for RTX 5060 Ti
    model_config_base = config.get('model', {})
    
    # Reduce model size if necessary for memory constraints
    max_seq_length = model_config_base.get('max_seq_length', 2048)
    if max_seq_length > 1024:
        print(f"⚠️  Reducing sequence length from {max_seq_length} to 1024 for memory optimization")
        max_seq_length = 1024
        
    model_config = L1Config(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        n_layers=model_config_base.get('n_layers', 12),
        n_heads=model_config_base.get('n_heads', 16),
        n_embd=model_config_base.get('n_embd', 1024),
        n_inner=model_config_base.get('n_inner', 4096),
        **{k: v for k, v in model_config_base.items() if k not in ['vocab_size', 'max_seq_length', 'n_layers', 'n_heads', 'n_embd', 'n_inner']}
    )
    
    print("🏗️  Creating model...")
    model = L1Model(model_config).to(device)
    
    # Calculate and display model statistics
    model_stats = calculate_model_size(model)
    print("📊 Model Statistics:")
    print(f"   ├── Total Parameters: {model_stats['total_params']:,}")
    print(f"   ├── Trainable Parameters: {model_stats['trainable_params']:,}")
    print(f"   ├── Model Size: {model_stats['model_size_mb']:.1f} MB")
    print(f"   └── Memory Usage (training): ~{model_stats['estimated_training_memory_mb']:.1f} MB")
    
    # Apply optimizations
    optimizations, scaler, use_amp = apply_model_optimizations(model, config, device)
    print("="*60)
    
    # Create dataset and dataloader
    print("📊 Loading training data...")
    train_data_path = config.get('data', {}).get('train_data_path', 'data/processed/train.txt')
    
    if not os.path.exists(train_data_path):
        print(f"❌ Training data not found at {train_data_path}")
        print("💡 Please prepare training data first!")
        return
    
    # Load dataset
    with open(train_data_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    dataset = SimpleTextDataset(texts, tokenizer, max_length=model_config.max_seq_length)
    print(f"✅ Dataset loaded: {len(dataset):,} samples")
    
    # Create DataLoader with memory-optimized settings
    batch_size = config.get('training', {}).get('batch_size', 8)
    
    # For RTX 5060 Ti 16GB, use smaller batch size to avoid OOM
    if batch_size > 8:
        print(f"⚠️  Reducing batch size from {batch_size} to 8 for RTX 5060 Ti memory constraints")
        batch_size = 8
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(2, os.cpu_count()),  # Reduce workers to save memory
        pin_memory=device.type == 'cuda',
        drop_last=True
    )
    
    print("📦 DataLoader configured:")
    print(f"   ├── Batch size: {batch_size}")
    print(f"   ├── Workers: {min(2, os.cpu_count())}")
    print(f"   └── Pin memory: {device.type == 'cuda'}")
    print("="*60)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    
    # Training configuration
    num_epochs = config.get('training', {}).get('num_epochs', 3)
    total_steps = len(dataloader) * num_epochs
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    print("🎓 Training Configuration:")
    print(f"   ├── Epochs: {num_epochs}")
    print(f"   ├── Total steps: {total_steps}")
    print(f"   ├── Checkpoint every: {config.get('training', {}).get('checkpoint_every_steps', 100)} steps (~{config.get('training', {}).get('checkpoint_every_steps', 100) * 0.18:.0f} min)")
    print(f"   ├── Keep checkpoints: {config.get('training', {}).get('max_checkpoints_to_keep', 5)} latest")
    print(f"   ├── Mixed precision: {use_amp}")
    print(f"   └── Optimizer: {type(optimizer).__name__}")
    print("="*60)
    
    # Setup output directory
    output_dir = config.get('training', {}).get('output_dir', './models/l1-gpu-compatible')
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    
    # Get the specific logger for training
    logger = logging.getLogger('l1_training')
    
    # Log training start
    logger.info("="*60)
    logger.info("TRAINING SESSION STARTED")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: {config.get('model', {}).get('n_layers', 'Unknown')} layers, {config.get('model', {}).get('n_heads', 'Unknown')} heads")
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Training configuration: {config.get('training', {})}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    
    # Check for existing checkpoint to resume from
    resume_path = os.path.join(output_dir, 'latest_checkpoint.pt')
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    
    if os.path.exists(resume_path):
        try:
            checkpoint_epoch, global_step, resume_loss = load_checkpoint(resume_path, model, optimizer, device)
            best_loss = resume_loss
            
            # Try to load the best loss from existing best checkpoint if it exists
            best_checkpoint_path = os.path.join(output_dir, 'best_checkpoint.pt')
            if os.path.exists(best_checkpoint_path):
                try:
                    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
                    if 'loss' in best_checkpoint:
                        best_loss = min(best_loss, best_checkpoint['loss'])
                        print(f"📊 Found existing best checkpoint with loss: {best_checkpoint['loss']:.4f}")
                except Exception as e:
                    print(f"⚠️  Could not load best checkpoint: {e}")
            
            # The checkpoint saves epoch as 1-based (human-readable), but our loop is 0-based
            # So if we're resuming from "epoch 1" (first epoch), we continue from epoch 0 in the loop
            start_epoch = checkpoint_epoch - 1
            
            print(f"🔄 Resuming training from epoch {checkpoint_epoch}, step {global_step}")
            print(f"📍 Continuing epoch {checkpoint_epoch} from step {global_step}")
            
            # Log resume information
            logger.info(f"RESUMING TRAINING from epoch {checkpoint_epoch}, step {global_step}")
            logger.info(f"Resumed loss: {resume_loss:.4f}")
            logger.info(f"Best loss so far: {best_loss:.4f}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("🚀 Starting fresh training...")
            start_epoch = 0
            global_step = 0
            best_loss = float('inf')
    else:
        print("🚀 No existing checkpoint found. Starting fresh training...")
        
        # Log fresh start
        logger.info("STARTING FRESH TRAINING - No existing checkpoint found")
        logger.info(f"Epochs planned: {num_epochs}")
        logger.info(f"Initial best loss: {best_loss}")
    
    # Initialize best checkpoint loss tracking (using explicit variable instead of function attributes)
    best_checkpoint_loss = best_loss
    
    # Training loop with memory management
    print("🏁 Starting training...")
    
    # Clear GPU cache before training
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("🧹 GPU cache cleared")
    
    for epoch in range(start_epoch, num_epochs):
        current_epoch_display = epoch + 1
        print(f"\n📚 Epoch {current_epoch_display}/{num_epochs}")
        
        # Check if this is a resumed training
        if epoch == start_epoch and global_step > 0:
            print(f"   └── Continuing from step {global_step}")
        
        # Train one epoch
        epoch_metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=current_epoch_display,  # Pass the display epoch number (1-based)
            config=config,
            scaler=scaler,
            use_amp=use_amp,
            scheduler=scheduler,
            save_checkpoint_fn=save_checkpoint,
            output_dir=output_dir,
            model_config=model_config,
            resumed_global_step=global_step if epoch == start_epoch else 0,  # Pass resumed step only for first resumed epoch
            best_checkpoint_loss=best_checkpoint_loss  # Pass the best checkpoint loss state
        )
        
        # Update best checkpoint loss from epoch results
        best_checkpoint_loss = epoch_metrics['best_checkpoint_loss']
        
        # Update global step counter
        global_step = current_epoch_display * len(dataloader)
        
        # Log metrics
        print(f"✅ Epoch {current_epoch_display} completed:")
        print(f"   ├── Average Loss: {epoch_metrics['loss']:.4f}")
        print(f"   └── Learning Rate: {epoch_metrics['learning_rate']:.2e}")
        
        # Log epoch completion
        log_training_metrics(global_step, current_epoch_display, epoch_metrics['loss'], epoch_metrics['learning_rate'], 
                           best_checkpoint_loss, is_best=False, 
                           additional_info="EPOCH_COMPLETE", avg_loss=epoch_metrics.get('avg_loss'))
        
        # Clear GPU cache after each epoch to prevent memory accumulation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Always save latest checkpoint (for resuming)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=current_epoch_display,  # Use the 1-based epoch number
            step=global_step,
            loss=epoch_metrics['loss'],
            save_dir=output_dir,
            is_best=False,
            model_config=model_config
        )
        
        # Save best checkpoint if loss improved (simplified tracking)
        current_epoch_loss = epoch_metrics['loss']
        
        # Only track and update epoch-level best loss at epoch boundary
        if current_epoch_loss < best_loss:
            previous_best = best_loss
            best_loss = current_epoch_loss
            
            # Log best epoch detection with correct previous best
            logger = logging.getLogger('l1_training')
            logger.info(f"NEW BEST EPOCH! Epoch: {current_epoch_display}, Loss: {current_epoch_loss:.4f}, Previous best: {previous_best:.4f}")
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=current_epoch_display,  # Use the 1-based epoch number
                step=global_step,
                loss=current_epoch_loss,
                save_dir=output_dir,
                is_best=True,
                model_config=model_config
            )
        
        # Save tokenizer (copy the original file)
        import shutil
        tokenizer_save_path = os.path.join(output_dir, 'tokenizer.json')
        if os.path.exists(tokenizer_path):
            shutil.copy2(tokenizer_path, tokenizer_save_path)
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"🏆 Best epoch loss: {best_loss:.4f}")
    if best_checkpoint_loss != float('inf'):
        print(f"⭐ Best checkpoint loss: {best_checkpoint_loss:.4f}")
        print(f"📋 Note: The best checkpoint (saved every 1000 steps) may be better than the best epoch!")
    else:
        print(f"📋 Checkpoint-level best tracking: Not available")
    
    # Log training completion
    logger.info("="*60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Final best epoch loss: {best_loss:.4f}")
    if best_checkpoint_loss != float('inf'):
        logger.info(f"Final best checkpoint loss: {best_checkpoint_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
