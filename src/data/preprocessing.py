"""
Data preprocessing utilities for L1 model.
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
