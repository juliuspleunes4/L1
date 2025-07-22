"""
@file       : trainer.py
@package    : src.training
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Trainer class for L1 model.
@details    : This script manages the training process for the L1 model, including
              dataset preparation, model training, and evaluation.
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
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any
from tqdm import tqdm

from .config import TrainingConfig
from .optimizer import get_optimizer, get_scheduler, GradientClipping
from .loss import LanguageModelingLoss, PerplexityMetric
from ..utils.logging import get_logger
from ..utils.device import get_device, move_to_device


class Trainer:
    """Trainer class for L1 model.
    
    Args:
        model: L1 model to train
        config: Training configuration
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader (optional)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Setup device
        self.device = get_device(config.device)
        self.model = move_to_device(self.model, self.device)
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        # Setup optimizer and scheduler
        self.optimizer = get_optimizer(model, config)
        
        # Calculate total training steps
        self.num_training_steps = self._calculate_num_training_steps()
        self.scheduler = get_scheduler(
            self.optimizer, 
            config, 
            self.num_training_steps
        )
        
        # Setup loss function and metrics
        self.criterion = LanguageModelingLoss(
            vocab_size=model.config.vocab_size,
            ignore_index=model.config.pad_token_id
        )
        
        # Gradient clipping
        self.grad_clipper = GradientClipping(config.max_grad_norm)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Setup logging
        self.logger = get_logger("Trainer")
        self.writer = SummaryWriter(
            log_dir=os.path.join(config.output_dir, "tensorboard")
        )
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        self.logger.info(f"Trainer initialized")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        self.logger.info(f"Training steps: {self.num_training_steps}")
        self.logger.info(f"Device: {self.device}")
    
    def _calculate_num_training_steps(self) -> int:
        """Calculate total number of training steps."""
        if self.config.max_steps is not None:
            return self.config.max_steps
        
        steps_per_epoch = len(self.train_dataloader)
        return steps_per_epoch * self.config.num_epochs
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        self.model.train()
        train_start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_steps = 0
            
            # Training loop
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch + 1}",
                disable=False
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Forward pass
                loss = self.training_step(batch)
                epoch_loss += loss
                epoch_steps += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    'step': self.global_step
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self.log_metrics({
                        'train/loss': loss,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'train/epoch': epoch,
                    })
                
                # Evaluation
                if (self.config.eval_strategy == "steps" and 
                    self.eval_dataloader is not None and
                    self.global_step % self.config.eval_steps == 0 and
                    self.global_step > 0):
                    
                    eval_results = self.evaluate()
                    self.log_metrics(eval_results)
                    
                    # Save best model
                    if eval_results['eval/loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_results['eval/loss']
                        self.save_checkpoint(is_best=True)
                        self.logger.info(f"New best model saved (eval_loss: {self.best_eval_loss:.4f})")
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Check if max steps reached
                if (self.config.max_steps is not None and 
                    self.global_step >= self.config.max_steps):
                    self.logger.info(f"Reached max_steps ({self.config.max_steps})")
                    return
            
            # End of epoch evaluation
            if (self.config.eval_strategy == "epoch" and 
                self.eval_dataloader is not None):
                
                eval_results = self.evaluate()
                self.log_metrics(eval_results)
                
                # Save best model
                if eval_results['eval/loss'] < self.best_eval_loss:
                    self.best_eval_loss = eval_results['eval/loss']
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"New best model saved (eval_loss: {self.best_eval_loss:.4f})")
            
            # Log epoch metrics
            avg_epoch_loss = epoch_loss / epoch_steps
            epoch_time = time.time() - epoch_start_time
            
            self.logger.info(
                f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - "
                f"Average loss: {avg_epoch_loss:.4f}"
            )
            
            self.log_metrics({
                'epoch/loss': avg_epoch_loss,
                'epoch/time': epoch_time,
            })
        
        # Final checkpoint
        self.save_checkpoint()
        
        total_time = time.time() - train_start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        # Load best model if requested
        if self.config.load_best_model_at_end:
            best_model_path = os.path.join(self.config.output_dir, "best_model.pt")
            if os.path.exists(best_model_path):
                self.load_checkpoint(best_model_path)
                self.logger.info("Loaded best model for final evaluation")
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Loss value
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        if self.config.fp16:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask')
                )
                loss = self.criterion(outputs['logits'], batch['input_ids'])
        else:
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask')
            )
            loss = self.criterion(outputs['logits'], batch['input_ids'])
        
        # Backward pass
        if self.config.fp16:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = self.grad_clipper.clip_gradients(self.model)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            grad_norm = self.grad_clipper.clip_gradients(self.model)
            
            # Optimizer step
            self.optimizer.step()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Update global step
        self.global_step += 1
        
        return loss.item()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set.
        
        Returns:
            Evaluation metrics
        """
        if self.eval_dataloader is None:
            raise ValueError("No evaluation dataloader provided")
        
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        perplexity_metric = PerplexityMetric()
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                if self.config.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch.get('attention_mask')
                        )
                        loss = self.criterion(outputs['logits'], batch['input_ids'])
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask')
                    )
                    loss = self.criterion(outputs['logits'], batch['input_ids'])
                
                total_loss += loss.item()
                total_steps += 1
                
                # Update perplexity
                num_tokens = (batch['input_ids'] != self.model.config.pad_token_id).sum().item()
                perplexity_metric.update(loss, num_tokens)
        
        # Calculate metrics
        avg_loss = total_loss / total_steps
        perplexity = perplexity_metric.compute()
        
        self.model.train()
        
        return {
            'eval/loss': avg_loss,
            'eval/perplexity': perplexity,
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config.to_dict(),
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.output_dir, 
            f"checkpoint-{self.global_step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_eval_loss = checkpoint['best_eval_loss']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if self.config.save_total_limit <= 0:
            return
        
        checkpoint_files = []
        for file in os.listdir(self.config.output_dir):
            if file.startswith("checkpoint-") and file.endswith(".pt"):
                checkpoint_files.append(file)
        
        # Sort by step number
        checkpoint_files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
        
        # Remove old checkpoints
        while len(checkpoint_files) > self.config.save_total_limit:
            old_checkpoint = checkpoint_files.pop(0)
            old_path = os.path.join(self.config.output_dir, old_checkpoint)
            os.remove(old_path)
            self.logger.info(f"Removed old checkpoint: {old_path}")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to tensorboard and console.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.global_step)
        
        # Log to console
        if self.global_step % self.config.logging_steps == 0:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Step {self.global_step} | {metric_str}")
    
    def save_final_model(self):
        """Save the final trained model for deployment."""
        import json
        
        # Create model config dict
        model_config = {
            'vocab_size': self.model.config.vocab_size,
            'max_seq_length': self.model.config.max_seq_length,
            'n_layers': self.model.config.n_layers,
            'n_heads': self.model.config.n_heads,
            'n_embd': self.model.config.n_embd,
            'n_inner': self.model.config.n_inner,
            'dropout': self.model.config.dropout,
            'layer_norm_epsilon': self.model.config.layer_norm_epsilon,
            'initializer_range': self.model.config.initializer_range,
            'use_cache': self.model.config.use_cache,
            'pad_token_id': self.model.config.pad_token_id,
            'eos_token_id': self.model.config.eos_token_id,
            'bos_token_id': self.model.config.bos_token_id,
        }
        
        # Save model config
        config_path = os.path.join(self.config.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save model weights
        model_path = os.path.join(self.config.output_dir, 'pytorch_model.bin')
        torch.save(self.model.state_dict(), model_path)
        
        # Save training config
        training_config_path = os.path.join(self.config.output_dir, 'training_args.json')
        with open(training_config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        self.logger.info(f"Final model saved to {self.config.output_dir}")
        self.logger.info(f"Model config: {config_path}")
        self.logger.info(f"Model weights: {model_path}")
        self.logger.info(f"Training config: {training_config_path}")
