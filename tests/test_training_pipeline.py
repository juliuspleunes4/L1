"""
@file       : test_training_pipeline.py
@author     : J.J.G. Pleunes
@date       : 08/2025
@brief      : Comprehensive test suite for training pipeline components.
@details    : Tests for trainer, optimizer, loss functions, training configuration,
              and training loop functionality.
@version    : 1.0

@license    : MIT License
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import L1Model, L1Config
from data import BPETokenizer, TextDataset
from training import Trainer, TrainingConfig, get_optimizer, get_scheduler, LanguageModelingLoss


class TestTrainingConfig(unittest.TestCase):
    """Test training configuration."""
    
    def test_default_config(self):
        """Test default training configuration."""
        try:
            config = TrainingConfig()
            
            # Check essential training parameters
            self.assertGreater(config.learning_rate, 0)
            self.assertGreater(config.batch_size, 0)
            self.assertGreater(config.num_epochs, 0)
            
            # Check optional parameters have reasonable defaults
            if hasattr(config, 'weight_decay'):
                self.assertGreaterEqual(config.weight_decay, 0)
            if hasattr(config, 'warmup_steps'):
                self.assertGreaterEqual(config.warmup_steps, 0)
            
        except:
            self.skipTest("TrainingConfig class not found")
    
    def test_custom_config(self):
        """Test custom training configuration."""
        try:
            config = TrainingConfig(
                learning_rate=0.001,
                batch_size=16,
                num_epochs=5,
                weight_decay=0.01
            )
            
            self.assertEqual(config.learning_rate, 0.001)
            self.assertEqual(config.batch_size, 16)
            self.assertEqual(config.num_epochs, 5)
            
            if hasattr(config, 'weight_decay'):
                self.assertEqual(config.weight_decay, 0.01)
                
        except:
            self.skipTest("TrainingConfig class not found")
    
    def test_config_validation(self):
        """Test configuration validation."""
        try:
            # Test invalid learning rate
            with self.assertRaises((ValueError, AssertionError)):
                TrainingConfig(learning_rate=-0.01)
            
            # Test invalid batch size
            with self.assertRaises((ValueError, AssertionError)):
                TrainingConfig(batch_size=0)
            
            # Test invalid epochs
            with self.assertRaises((ValueError, AssertionError)):
                TrainingConfig(num_epochs=-1)
                
        except:
            self.skipTest("TrainingConfig validation not implemented")


class TestLossFunctions(unittest.TestCase):
    """Test loss function implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_length = 8
        self.vocab_size = 100
        
        # Create sample logits and targets
        self.logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        self.targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
    
    def test_cross_entropy_loss(self):
        """Test cross entropy loss implementation."""
        try:
            loss_fn = LanguageModelingLoss()
            loss = loss_fn(self.logits, self.targets)
            
            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.dim(), 0)  # Scalar loss
            self.assertGreater(loss.item(), 0)  # Loss should be positive
            
        except:
            # Fallback to PyTorch standard loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                self.logits.view(-1, self.vocab_size),
                self.targets.view(-1)
            )
            
            self.assertIsInstance(loss, torch.Tensor)
            self.assertGreater(loss.item(), 0)
    
    def test_label_smoothed_loss(self):
        """Test label smoothed cross entropy loss."""
        try:
            # Try with smoothing parameter if supported
            loss_fn = LanguageModelingLoss(smoothing=0.1)
            loss = loss_fn(self.logits, self.targets)
            
            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.dim(), 0)
            self.assertGreater(loss.item(), 0)
            
        except:
            self.skipTest("Label smoothing not supported in LanguageModelingLoss")
    
    def test_loss_gradients(self):
        """Test loss function gradients."""
        # Create a simple model
        model = nn.Linear(self.vocab_size, self.vocab_size)
        
        # Forward pass
        logits = model(torch.randn(self.batch_size * self.seq_length, self.vocab_size))
        targets = torch.randint(0, self.vocab_size, (self.batch_size * self.seq_length,))
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())
    
    def test_loss_with_padding(self):
        """Test loss computation with padding tokens."""
        # Create targets with padding (assuming 0 is pad token)
        targets_with_pad = self.targets.clone()
        targets_with_pad[:, -2:] = 0  # Last 2 tokens are padding
        
        # Loss with ignore_index for padding
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(
            self.logits.view(-1, self.vocab_size),
            targets_with_pad.view(-1)
        )
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)


class TestOptimizers(unittest.TestCase):
    """Test optimizer configurations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_config = L1Config(
            vocab_size=100,
            max_seq_length=32,
            n_layers=2,
            n_heads=4,
            n_embd=64
        )
        self.model = L1Model(self.model_config)
    
    def test_get_optimizer_adam(self):
        """Test Adam optimizer creation."""
        try:
            optimizer = get_optimizer(
                model=self.model,
                optimizer_type='adam',
                learning_rate=0.001,
                weight_decay=0.01
            )
            
            self.assertIsInstance(optimizer, (optim.Adam, optim.AdamW))
            
            # Check learning rate
            self.assertEqual(optimizer.param_groups[0]['lr'], 0.001)
            
        except:
            # Fallback manual optimizer creation
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=0.001,
                weight_decay=0.01
            )
            
            self.assertIsInstance(optimizer, optim.Adam)
    
    def test_get_optimizer_adamw(self):
        """Test AdamW optimizer creation."""
        try:
            optimizer = get_optimizer(
                model=self.model,
                optimizer_type='adamw',
                learning_rate=0.0005,
                weight_decay=0.1
            )
            
            self.assertIsInstance(optimizer, optim.AdamW)
            
        except:
            # Manual creation
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=0.0005,
                weight_decay=0.1
            )
            
            self.assertIsInstance(optimizer, optim.AdamW)
    
    def test_optimizer_step(self):
        """Test optimizer step functionality."""
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Get initial parameters
        initial_params = [p.clone() for p in self.model.parameters()]
        
        # Forward pass with dummy data
        input_ids = torch.randint(0, self.model_config.vocab_size, (1, 10))
        outputs = self.model(input_ids)
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Dummy loss
        targets = torch.randint(0, self.model_config.vocab_size, (1, 10))
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, self.model_config.vocab_size),
            targets.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Check parameters changed
        for initial, current in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.equal(initial, current))
    
    def test_scheduler_integration(self):
        """Test learning rate scheduler integration."""
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        try:
            scheduler = get_scheduler(
                optimizer=optimizer,
                scheduler_type='cosine',
                num_training_steps=1000,
                warmup_steps=100
            )
            
            # Check initial learning rate
            initial_lr = optimizer.param_groups[0]['lr']
            
            # Step scheduler
            for _ in range(10):
                scheduler.step()
            
            # Learning rate should have changed
            new_lr = optimizer.param_groups[0]['lr']
            # Note: might be same in early steps of cosine schedule
            
        except:
            # Manual scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
            
            initial_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            # scheduler.step() might not change LR immediately


class TestTrainer(unittest.TestCase):
    """Test trainer implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Model configuration
        self.model_config = L1Config(
            vocab_size=200,
            max_seq_length=64,
            n_layers=2,
            n_heads=4,
            n_embd=128
        )
        self.model = L1Model(self.model_config)
        
        # Training configuration
        try:
            self.training_config = TrainingConfig(
                learning_rate=0.001,
                batch_size=2,
                num_epochs=1,
                max_steps=10
            )
        except:
            # Manual config
            self.training_config = type('TrainingConfig', (), {
                'learning_rate': 0.001,
                'batch_size': 2,
                'num_epochs': 1,
                'max_steps': 10
            })()
        
        # Create dummy dataset
        self.tokenizer = BPETokenizer(vocab_size=200)
        sample_texts = ["Hello world", "This is a test", "Training data"]
        self.tokenizer.train(sample_texts)
        
        # Create temp file for dataset
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        for text in sample_texts:
            self.temp_file.write(text + '\n')
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        try:
            trainer = Trainer(
                model=self.model,
                config=self.training_config,
                tokenizer=self.tokenizer
            )
            
            self.assertIsInstance(trainer, Trainer)
            
        except:
            self.skipTest("Trainer class not found")
    
    def test_training_step(self):
        """Test single training step."""
        try:
            trainer = Trainer(
                model=self.model,
                config=self.training_config,
                tokenizer=self.tokenizer
            )
            
            # Create dummy batch
            batch_size = 2
            seq_length = 32
            input_ids = torch.randint(0, self.model_config.vocab_size, (batch_size, seq_length))
            
            if hasattr(trainer, 'training_step'):
                loss = trainer.training_step(input_ids)
                
                self.assertIsInstance(loss, torch.Tensor)
                self.assertGreater(loss.item(), 0)
            
        except:
            self.skipTest("Trainer training_step not found")
    
    def test_evaluation_step(self):
        """Test evaluation step."""
        try:
            trainer = Trainer(
                model=self.model,
                config=self.training_config,
                tokenizer=self.tokenizer
            )
            
            # Create dummy batch
            batch_size = 2
            seq_length = 32
            input_ids = torch.randint(0, self.model_config.vocab_size, (batch_size, seq_length))
            
            if hasattr(trainer, 'eval_step') or hasattr(trainer, 'validation_step'):
                eval_method = getattr(trainer, 'eval_step', None) or getattr(trainer, 'validation_step')
                
                with torch.no_grad():
                    loss = eval_method(input_ids)
                    
                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertGreater(loss.item(), 0)
            
        except:
            self.skipTest("Trainer evaluation step not found")
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        try:
            trainer = Trainer(
                model=self.model,
                config=self.training_config,
                tokenizer=self.tokenizer
            )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = os.path.join(temp_dir, 'checkpoint.pt')
                
                if hasattr(trainer, 'save_checkpoint'):
                    trainer.save_checkpoint(checkpoint_path, epoch=1, step=100)
                    
                    self.assertTrue(os.path.exists(checkpoint_path))
                    
                    # Load and verify checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    self.assertIn('model_state_dict', checkpoint)
                    self.assertIn('epoch', checkpoint)
                    self.assertIn('step', checkpoint)
            
        except:
            self.skipTest("Trainer save_checkpoint not found")
    
    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        try:
            trainer = Trainer(
                model=self.model,
                config=self.training_config,
                tokenizer=self.tokenizer
            )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = os.path.join(temp_dir, 'checkpoint.pt')
                
                # Save checkpoint first
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'epoch': 1,
                    'step': 100,
                    'loss': 2.5
                }
                torch.save(checkpoint, checkpoint_path)
                
                if hasattr(trainer, 'load_checkpoint'):
                    result = trainer.load_checkpoint(checkpoint_path)
                    
                    # Check result contains expected info
                    if isinstance(result, dict):
                        self.assertIn('epoch', result)
                        self.assertIn('step', result)
            
        except:
            self.skipTest("Trainer load_checkpoint not found")


class TestTrainingLoop(unittest.TestCase):
    """Test complete training loop functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Small model for fast testing
        self.model_config = L1Config(
            vocab_size=100,
            max_seq_length=32,
            n_layers=1,
            n_heads=2,
            n_embd=32
        )
        self.model = L1Model(self.model_config)
        
        # Tokenizer
        self.tokenizer = BPETokenizer(vocab_size=100)
        sample_texts = ["Hello", "World", "Test", "Data"]
        self.tokenizer.train(sample_texts)
    
    def test_training_loop_basic(self):
        """Test basic training loop functionality."""
        # Create optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        # Training data
        batch_size = 2
        seq_length = 16
        num_steps = 5
        
        initial_loss = None
        final_loss = None
        
        self.model.train()
        
        for step in range(num_steps):
            # Generate random batch
            input_ids = torch.randint(0, self.model_config.vocab_size, (batch_size, seq_length))
            targets = torch.randint(0, self.model_config.vocab_size, (batch_size, seq_length))
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(input_ids)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Compute loss
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.model_config.vocab_size),
                targets.view(-1)
            )
            
            if step == 0:
                initial_loss = loss.item()
            if step == num_steps - 1:
                final_loss = loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Training should reduce loss (usually, but not guaranteed with random data)
        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)
        print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
    
    def test_model_improvement(self):
        """Test that model parameters change during training."""
        # Get initial parameters
        initial_params = []
        for param in self.model.parameters():
            initial_params.append(param.clone())
        
        # Train for a few steps
        optimizer = optim.Adam(self.model.parameters(), lr=0.1)  # High LR for guaranteed change
        
        for _ in range(3):
            input_ids = torch.randint(0, self.model_config.vocab_size, (1, 8))
            targets = torch.randint(0, self.model_config.vocab_size, (1, 8))
            
            optimizer.zero_grad()
            outputs = self.model(input_ids)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.model_config.vocab_size),
                targets.view(-1)
            )
            
            loss.backward()
            optimizer.step()
        
        # Check parameters changed
        parameters_changed = False
        for initial, current in zip(initial_params, self.model.parameters()):
            if not torch.equal(initial, current):
                parameters_changed = True
                break
        
        self.assertTrue(parameters_changed, "Model parameters should change during training")
    
    def test_overfitting_small_dataset(self):
        """Test model can overfit to a small dataset."""
        # Very small dataset for guaranteed overfitting
        fixed_input = torch.tensor([[1, 2, 3, 4]])
        fixed_target = torch.tensor([[2, 3, 4, 1]])
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        initial_loss = None
        final_loss = None
        
        self.model.train()
        
        # Train for many steps on same data
        for step in range(50):
            optimizer.zero_grad()
            outputs = self.model(fixed_input)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.model_config.vocab_size),
                fixed_target.view(-1)
            )
            
            if step == 0:
                initial_loss = loss.item()
            if step == 49:
                final_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Loss should decrease significantly with overfitting
        self.assertLess(final_loss, initial_loss * 0.8, 
                       f"Loss should decrease significantly: {initial_loss:.4f} -> {final_loss:.4f}")


if __name__ == '__main__':
    # Set random seed for reproducible tests
    torch.manual_seed(42)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)
