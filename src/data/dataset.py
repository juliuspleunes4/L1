"""
@file       : dataset.py
@package    : src.data
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Dataset management for L1 model.
@details    : This module provides dataset classes for handling text data,
              including tokenization, padding, and batching for training and evaluation.
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
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Union, Tuple
import random

from .tokenizer import Tokenizer


class TextDataset(Dataset):
    """Dataset for text data with tokenization.
    
    Args:
        texts: List of text strings or path to text file
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        stride: Stride for sliding window (for long texts)
    """
    
    def __init__(
        self,
        texts: Union[List[str], str],
        tokenizer: Tokenizer,
        max_length: int = 1024,
        stride: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length // 2
        
        # Load texts
        if isinstance(texts, str):
            # Load from file
            self.texts = self._load_texts_from_file(texts)
        else:
            self.texts = texts
        
        # Tokenize all texts
        self.examples = self._create_examples()
    
    def _load_texts_from_file(self, file_path: str) -> List[str]:
        """Load texts from file."""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        return texts
    
    def _create_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Create training examples from texts."""
        examples = []
        
        for text in self.texts:
            # Tokenize text
            token_ids = self.tokenizer.encode(text)
            
            # Create sliding windows for long texts
            if len(token_ids) <= self.max_length:
                # Pad if necessary
                if len(token_ids) < self.max_length:
                    token_ids.extend([self.tokenizer.pad_token_id] * 
                                   (self.max_length - len(token_ids)))
                
                examples.append({
                    'input_ids': torch.tensor(token_ids, dtype=torch.long),
                    'attention_mask': torch.ones(len(token_ids), dtype=torch.long)
                })
            else:
                # Use sliding window
                for i in range(0, len(token_ids) - self.max_length + 1, self.stride):
                    window = token_ids[i:i + self.max_length]
                    examples.append({
                        'input_ids': torch.tensor(window, dtype=torch.long),
                        'attention_mask': torch.ones(len(window), dtype=torch.long)
                    })
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class TextDataModule:
    """Data module for handling train/validation/test splits.
    
    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        train_texts: Training texts
        val_texts: Validation texts
        test_texts: Test texts
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int = 1024,
        train_texts: Optional[Union[List[str], str]] = None,
        val_texts: Optional[Union[List[str], str]] = None,
        test_texts: Optional[Union[List[str], str]] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        
        # Handle data splits
        if train_texts is not None and val_texts is None and test_texts is None:
            # Auto-split single dataset
            if isinstance(train_texts, str):
                all_texts = self._load_texts_from_file(train_texts)
            else:
                all_texts = train_texts
            
            self.train_texts, self.val_texts, self.test_texts = self._split_texts(all_texts)
        else:
            self.train_texts = train_texts
            self.val_texts = val_texts
            self.test_texts = test_texts
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self._setup_datasets()
    
    def _load_texts_from_file(self, file_path: str) -> List[str]:
        """Load texts from file."""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        return texts
    
    def _split_texts(self, texts: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Split texts into train/val/test sets."""
        random.seed(self.seed)
        random.shuffle(texts)
        
        n_total = len(texts)
        n_test = int(n_total * self.test_split)
        n_val = int(n_total * self.val_split)
        n_train = n_total - n_test - n_val
        
        train_texts = texts[:n_train]
        val_texts = texts[n_train:n_train + n_val]
        test_texts = texts[n_train + n_val:]
        
        return train_texts, val_texts, test_texts
    
    def _setup_datasets(self):
        """Setup train/val/test datasets."""
        if self.train_texts:
            self.train_dataset = TextDataset(
                self.train_texts, 
                self.tokenizer, 
                self.max_length
            )
        
        if self.val_texts:
            self.val_dataset = TextDataset(
                self.val_texts, 
                self.tokenizer, 
                self.max_length
            )
        
        if self.test_texts:
            self.test_dataset = TextDataset(
                self.test_texts, 
                self.tokenizer, 
                self.max_length
            )
    
    def train_dataloader(self, batch_size: int = 8, num_workers: int = 4) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            raise ValueError("No training dataset available")
        
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self, batch_size: int = 8, num_workers: int = 4) -> DataLoader:
        """Get validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("No validation dataset available")
        
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self, batch_size: int = 8, num_workers: int = 4) -> DataLoader:
        """Get test dataloader."""
        if self.test_dataset is None:
            raise ValueError("No test dataset available")
        
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
