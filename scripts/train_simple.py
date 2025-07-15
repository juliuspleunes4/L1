#!/usr/bin/env python3
"""
Simple training script for L1 model.
This version works around import issues by using direct imports.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_dir)

# Direct imports
import models.config as model_config
import models.transformer as transformer_model
import training.trainer as trainer_module
import training.config as training_config
import data.tokenizer as tokenizer_module
import data.dataset as dataset_module
import utils.logging as logging_utils
import utils.seed as seed_utils


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_tokenizer(config: dict):
    """Create or load tokenizer."""
    # First check if tokenizer exists in data/processed
    prepared_tokenizer_path = './data/processed/tokenizer.json'
    output_tokenizer_path = os.path.join(config['training']['output_dir'], 'tokenizer.json')
    
    if os.path.exists(prepared_tokenizer_path):
        print(f"Loading existing tokenizer from {prepared_tokenizer_path}")
        tokenizer = tokenizer_module.BPETokenizer.load(prepared_tokenizer_path)
        
        # Copy tokenizer to output directory
        os.makedirs(os.path.dirname(output_tokenizer_path), exist_ok=True)
        tokenizer.save(output_tokenizer_path)
        print(f"Tokenizer copied to {output_tokenizer_path}")
        
    else:
        print("Training new tokenizer...")
        # Load training data
        train_texts = []
        with open(config['data']['train_data_path'], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    train_texts.append(line)
        
        # Train tokenizer
        tokenizer = tokenizer_module.BPETokenizer(vocab_size=config['data']['vocab_size'])
        tokenizer.train(train_texts[:10000])
        
        # Save tokenizer
        os.makedirs(os.path.dirname(output_tokenizer_path), exist_ok=True)
        tokenizer.save(output_tokenizer_path)
        print(f"Tokenizer saved to {output_tokenizer_path}")
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description='Train L1 model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility
    seed_utils.set_seed(config['training']['seed'])
    
    # Setup logging
    logging_utils.setup_logging(config['training']['output_dir'])
    
    # Create tokenizer
    tokenizer = create_tokenizer(config)
    
    # Update model config with tokenizer settings
    model_cfg = model_config.L1Config(**config['model'])
    model_cfg.vocab_size = len(tokenizer.vocab)
    model_cfg.pad_token_id = tokenizer.pad_token_id
    model_cfg.eos_token_id = tokenizer.eos_token_id
    model_cfg.bos_token_id = tokenizer.bos_token_id
    
    # Create model
    model = transformer_model.L1Model(model_cfg)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data module
    data_module = dataset_module.TextDataModule(
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],
        train_texts=config['data']['train_data_path'],
        val_texts=config['data']['val_data_path'],
        test_texts=config['data']['test_data_path'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        seed=config['training']['seed']
    )
    
    # Create training configuration
    train_cfg = training_config.TrainingConfig(**config['training'])
    
    # Create trainer
    trainer = trainer_module.Trainer(
        model=model,
        config=train_cfg,
        train_dataloader=data_module.train_dataloader(
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.dataloader_num_workers
        ),
        eval_dataloader=data_module.val_dataloader(
            batch_size=train_cfg.eval_batch_size,
            num_workers=train_cfg.dataloader_num_workers
        ) if data_module.val_dataset else None
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed training from {args.resume}")
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model for deployment
    trainer.save_final_model()
    
    print("Training completed!")
    print(f"Model saved to: {train_cfg.output_dir}")
    print("Use scripts/generate_new.py to generate text with your trained model!")


if __name__ == '__main__':
    main()
