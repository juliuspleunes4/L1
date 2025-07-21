#!/usr/bin/env python3
"""
GPU-compatible training script for L1 model.
Fixed for RTX 5060 Ti (sm_120) compatibility issues.
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
from train_minimal import L1Config, L1Model, SimpleTokenizer, SimpleTextDataset

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(output_dir: str):
    """Setup logging configuration"""
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
        return {
            'count': gpu_count,
            'current': current_gpu,
            'name': gpu_name,
            'memory_gb': gpu_memory
        }
    return None

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
    scheduler=None
) -> Dict[str, float]:
    """Train for one epoch - compatible version"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    # Training configuration
    gradient_accumulation_steps = config.get('training', {}).get('gradient_accumulation_steps', 1)
    max_grad_norm = config.get('training', {}).get('max_grad_norm', 1.0)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
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
            current_lr = optimizer.param_groups[0]['lr']
            
            progress_bar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
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
    
    return {
        'loss': total_loss / num_batches,
        'learning_rate': optimizer.param_groups[0]['lr']
    }

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: dict,
    epoch: int,
    step: int,
    loss: float,
    save_dir: str
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest checkpoint
    latest_path = os.path.join(save_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")

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
    print(f"📝 Loading tokenizer from {tokenizer_path}")
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        print("💡 Please run data preparation first!")
        return
    
    tokenizer = SimpleTokenizer(tokenizer_path)
    
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
        **{k: v for k, v in model_config_base.items() if k not in ['max_seq_length', 'n_layers', 'n_heads', 'n_embd', 'n_inner']}
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
    train_data_path = config.get('data', {}).get('train_path', 'data/processed/train.txt')
    
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
    print(f"   ├── Mixed precision: {use_amp}")
    print(f"   └── Optimizer: {type(optimizer).__name__}")
    print("="*60)
    
    # Setup output directory
    output_dir = config.get('output', {}).get('model_dir', './models/l1-gpu-compatible')
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    
    # Training loop with memory management
    print("🏁 Starting training...")
    
    # Clear GPU cache before training
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("🧹 GPU cache cleared")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
        
        # Train one epoch
        epoch_metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch+1,
            config=config,
            scaler=scaler,
            use_amp=use_amp,
            scheduler=scheduler
        )
        
        # Log metrics
        print(f"✅ Epoch {epoch+1} completed:")
        print(f"   ├── Average Loss: {epoch_metrics['loss']:.4f}")
        print(f"   └── Learning Rate: {epoch_metrics['learning_rate']:.2e}")
        
        # Clear GPU cache after each epoch to prevent memory accumulation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Save checkpoint if loss improved
        if epoch_metrics['loss'] < best_loss:
            best_loss = epoch_metrics['loss']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch+1,
                step=(epoch+1) * len(dataloader),
                loss=best_loss,
                save_dir=output_dir
            )
        
        # Save tokenizer (copy the original file)
        import shutil
        tokenizer_save_path = os.path.join(output_dir, 'tokenizer.json')
        if os.path.exists(tokenizer_path):
            shutil.copy2(tokenizer_path, tokenizer_save_path)
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"🏆 Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
