"""
@file       : preprocessing.py
@package    : src.data
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Data preprocessing utilities for L1 model.
@details    : This script provides various utilities for preprocessing text data,
              including cleaning, tokenization, and dataset management.
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

import torch
from typing import List, Dict, Any, Optional


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    def __init__(self):
        pass
    
    def clean_text(self, text: str) -> str:
        """Clean input text."""
        # Basic cleaning
        text = text.strip()
        # Remove multiple whitespaces
        text = ' '.join(text.split())
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = []
        for sentence in text.split('.'):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence + '.')
        return sentences


class DataCollator:
    """Data collator for batching."""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples."""
        # Get maximum length
        max_length = max(example['input_ids'].size(0) for example in batch)
        
        # Pad all sequences to max length
        input_ids = []
        attention_masks = []
        
        for example in batch:
            seq_len = example['input_ids'].size(0)
            
            # Pad input_ids
            padded_input = torch.cat([
                example['input_ids'],
                torch.full((max_length - seq_len,), self.pad_token_id, dtype=torch.long)
            ])
            input_ids.append(padded_input)
            
            # Create attention mask
            attention_mask = torch.cat([
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(max_length - seq_len, dtype=torch.long)
            ])
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks)
        }
