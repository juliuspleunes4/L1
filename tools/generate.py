#!/usr/bin/env python3
"""
@file       : generate_simple.py
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Simple text generation script for L1 model.
@details    : This script loads a pre-trained L1 model and generates text based on a
              prompt provided by the user.
@version    : 3.3

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
import json
import argparse
import sys
import regex as re
from typing import List, Dict, Optional, Union, Tuple
from collections import Counter, defaultdict

# Load the BPE tokenizer class definition directly
# (copying essential parts to avoid complex imports)

class BPETokenizer:
    """Simplified BPE tokenizer for text generation."""
    
    def __init__(self, vocab_size: int = 50257, special_tokens: Optional[Dict[str, int]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            '<pad>': 0,
            '<unk>': 1, 
            '<bos>': 2,
            '<eos>': 3,
        }
        
        # Initialize vocabulary with special tokens
        self.vocab = {token: idx for token, idx in self.special_tokens.items()}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        # BPE merges
        self.merges = {}
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Patterns for text preprocessing
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """Create a mapping from bytes to unicode characters."""
        bs = list(range(ord("!"), ord("~")+1)) + \
             list(range(ord("¡"), ord("¬")+1)) + \
             list(range(ord("®"), ord("ÿ")+1))
        
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        """Get all pairs of consecutive symbols in word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _bpe(self, token: str) -> str:
        """Apply BPE to a token."""
        if token in self.vocab:
            return token
        
        word = tuple(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
            if bigram not in self.merges:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word = tuple(new_word)
            word = new_word
            
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)
        
        return ' '.join(word)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        bpe_tokens = []
        
        for token in re.findall(self.pat, text):
            token_bytes = token.encode('utf-8')
            token_unicode = ''.join(self.byte_encoder[b] for b in token_bytes)
            
            bpe_token = self._bpe(token_unicode)
            bpe_tokens.extend(bpe_token.split(' '))
        
        # Convert to IDs
        token_ids = []
        for token in bpe_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['<unk>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                # Skip special tokens in output (except unk which we'll handle)
                if token not in ['<pad>', '<bos>', '<eos>']:
                    tokens.append(token)
            else:
                tokens.append('<unk>')
        
        # Join all BPE tokens
        text = ''.join(tokens)
        
        # Decode from unicode back to bytes, then to text
        try:
            text_bytes = bytearray()
            for c in text:
                if c in self.byte_decoder:
                    text_bytes.append(self.byte_decoder[c])
                else:
                    # Fallback for unknown characters
                    text_bytes.extend(c.encode('utf-8'))
            
            decoded_text = text_bytes.decode('utf-8', errors='replace')
            
            # More aggressive <unk> cleanup since we fixed the vocab
            import re
            
            # Replace most <unk> tokens with spaces (they're usually separators)
            decoded_text = re.sub(r'<unk>', ' ', decoded_text)
            
            # Clean up multiple spaces
            decoded_text = re.sub(r'\s+', ' ', decoded_text).strip()
            
            return decoded_text
            
        except (KeyError, UnicodeDecodeError):
            # Fallback: return as-is if byte decoding fails, but apply smart <unk> handling
            import re
            
            # Smart replacement: only replace <unk> when it's clearly a separator
            clean_text = re.sub(r'(?<=\w)<unk>(?=\w)', ' ', text)
            clean_text = re.sub(r'^<unk>(?=\w)', ' ', clean_text)
            clean_text = re.sub(r'(?<=\w)<unk>$', ' ', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            return clean_text
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # Handle different tokenizer formats
        special_tokens = tokenizer_data.get('special_tokens', {
            '<pad>': 0,
            '<unk>': 1, 
            '<bos>': 2,
            '<eos>': 3,
        })
        
        vocab_size = tokenizer_data.get('vocab_size', len(tokenizer_data['vocab']))
        
        tokenizer = cls(
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
        
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.id_to_token = {int(idx): token for token, idx in tokenizer.vocab.items()}
        
        # Handle merges if present (for full BPE) or empty dict for simple vocab
        if 'merges' in tokenizer_data:
            tokenizer.merges = {tuple(k.split(' ', 1)): v for k, v in tokenizer_data['merges'].items()}
        else:
            tokenizer.merges = {}
        
        return tokenizer
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.special_tokens.get('<pad>', 0)
    
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.special_tokens.get('<unk>', 1)
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.special_tokens.get('<bos>', 2)
    
    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        return self.special_tokens.get('<eos>', 3)

# Copy the model classes from training script
class L1Config:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 1000)
        self.max_seq_length = kwargs.get('max_seq_length', 512)
        self.n_layers = kwargs.get('n_layers', 6)
        self.n_heads = kwargs.get('n_heads', 8)
        self.n_embd = kwargs.get('n_embd', 512)
        self.n_inner = kwargs.get('n_inner', 2048)
        self.dropout = kwargs.get('dropout', 0.1)
        self.layer_norm_epsilon = kwargs.get('layer_norm_epsilon', 1e-5)
        self.initializer_range = kwargs.get('initializer_range', 0.02)
        self.use_cache = kwargs.get('use_cache', True)
        self.pad_token_id = kwargs.get('pad_token_id', 0)
        self.eos_token_id = kwargs.get('eos_token_id', 3)
        self.bos_token_id = kwargs.get('bos_token_id', 2)

# [Copy the same model classes from train_minimal.py - L1Model, TransformerBlock, etc.]
import torch.nn as nn
import math

class L1Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Output projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states = token_embeds + position_embeds
        
        # Apply transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        return type('ModelOutput', (), {'logits': logits})()

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.ln1(x))
        # MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_heads
        
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.output = nn.Linear(config.n_embd, config.n_embd)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        batch_size, seq_length, n_embd = x.shape
        
        # Get Q, K, V
        q = self.query(x).view(batch_size, seq_length, self.n_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.n_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.n_heads, self.head_size).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        scores.masked_fill_(mask.to(scores.device), float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, n_embd
        )
        
        return self.output(attention_output)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_inner)
        self.fc2 = nn.Linear(config.n_inner, config.n_embd)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_tokenizer(model_dir):
    """Load the BPE tokenizer from the model directory."""
    tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    print(f"Loading BPE tokenizer from {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")
    
    return tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device
    
    # Encode the prompt with BOS token
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)
    
    generated_ids = input_ids.copy()
    
    print(f"🔍 Debug: Prompt tokens: {input_ids[:10]}...")  # Debug first 10 tokens
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Get model output
            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]  # Last token
            
            # Apply temperature
            logits = logits / temperature
            
            # Use top-k sampling for better quality
            top_k = 50
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Debug: show what token was selected
            if i < 5:  # Show first 5 tokens
                token_text = tokenizer.id_to_token.get(next_token, f"ID:{next_token}")
                print(f"🔍 Generated token {i+1}: {next_token} -> '{token_text}'")
            
            # Add to sequence
            generated_ids.append(next_token)
            
            # Stop if EOS
            if next_token == tokenizer.eos_token_id:
                print("🛑 Hit EOS token, stopping generation")
                break
            
            # Update input tensor
            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=device)
            ], dim=1)
            
            # Trim if too long
            if input_tensor.size(1) > 100:  # Keep last 100 tokens
                input_tensor = input_tensor[:, -100:]
    
    # Decode and return, but filter out special tokens for cleaner output
    full_text = tokenizer.decode(generated_ids)
    
    # Try to extract just the generated part (after the prompt)
    try:
        prompt_text = tokenizer.decode([tokenizer.bos_token_id] + tokenizer.encode(prompt))
        if full_text.startswith(prompt_text):
            generated_part = full_text[len(prompt_text):]
            return prompt + generated_part
        else:
            return full_text
    except:
        return full_text


def main():
    parser = argparse.ArgumentParser(description='Generate text with L1 model')
    parser.add_argument('--model_path', type=str, default='./models/l1-v1',
                       help='Path to trained model directory')
    parser.add_argument('--prompt', type=str, default='the',
                       help='Text prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    
    args = parser.parse_args()
    
    print(f"🚀 Loading L1 model from: {args.model_path}")
    
    # Load config
    with open(os.path.join(args.model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    config = L1Config(**config_dict)
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.model_path)
    
    # Load model
    model = L1Model(config)
    
    # Try to load from checkpoint first (for actively training models)
    checkpoint_path = os.path.join(args.model_path, 'latest_checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.model_path, 'checkpoint_epoch_1_step_16000.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded from step {checkpoint.get('step', 'unknown')}")
    else:
        # Fallback to final model
        state_dict = torch.load(os.path.join(args.model_path, 'pytorch_model.bin'), 
                               map_location='cpu')
        model.load_state_dict(state_dict)
        print("✅ Loaded from pytorch_model.bin")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 Device: {device}")
    print(f"📝 Vocabulary size: {len(tokenizer.vocab)}")
    
    print(f"\n🎭 Generating text with prompt: '{args.prompt}'")
    print("=" * 60)
    
    # Generate text
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    print(generated_text)
    print("=" * 60)
    print(f"🎉 Generated {len(generated_text.split())} words!")


if __name__ == '__main__':
    main()
