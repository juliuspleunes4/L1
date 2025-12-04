"""
@file       : tokenizer.py
@package    : src.data
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Tokenizer package for L1 project.
@details    : This script implements various tokenization strategies for the L1 model,
              including byte pair encoding (BPE) and word-level tokenization.
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

import json
import pickle
import regex as re
from typing import List, Dict, Optional, Union, Tuple
from collections import Counter, defaultdict


class Tokenizer:
    """Base tokenizer class."""
    
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.special_tokens = {}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        raise NotImplementedError
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        raise NotImplementedError
    
    def save(self, path: str):
        """Save tokenizer to file."""
        raise NotImplementedError
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from file."""
        raise NotImplementedError


class BPETokenizer(Tokenizer):
    """Byte Pair Encoding tokenizer implementation.
    
    Args:
        vocab_size: Target vocabulary size
        special_tokens: Dictionary of special tokens
    """
    
    def __init__(
        self, 
        vocab_size: int = 50257,
        special_tokens: Optional[Dict[str, int]] = None
    ):
        super().__init__()
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
    
    def train(self, texts: List[str], min_frequency: int = 2) -> 'BPETokenizer':
        """
        Train BPE tokenizer on text corpus.
        
        Args:
            texts: List of texts to train on
            min_frequency: Minimum frequency for a word to be included (default: 2)
            
        Returns:
            Trained tokenizer
        """
        print("Training BPE tokenizer...")
        
        # Tokenize texts into words
        word_freqs = Counter()
        for text in texts:
            words = re.findall(self.pat, text)
            for word in words:
                word_bytes = word.encode('utf-8')
                word_unicode = ''.join(self.byte_encoder[b] for b in word_bytes)
                word_freqs[word_unicode] += 1
        
        # Filter rare words to speed up training dramatically
        if min_frequency > 1:
            original_count = len(word_freqs)
            word_freqs = Counter({word: freq for word, freq in word_freqs.items() 
                                 if freq >= min_frequency})
            print(f"Filtered words: {original_count:,} → {len(word_freqs):,} (min_freq={min_frequency})")
        
        # Initialize vocabulary with character-level tokens
        vocab = list(self.special_tokens.keys())
        
        # Add ALL byte-level tokens first (256 possible bytes encoded as unicode)
        # This ensures complete coverage and no <unk> tokens for any input
        byte_tokens = sorted(set(self.byte_encoder.values()))
        for token in byte_tokens:
            if token not in vocab:
                vocab.append(token)
        
        # Add all unique characters from training corpus
        chars = set()
        for word in word_freqs:
            chars.update(word)
        # Only add chars not already in vocab (byte tokens cover most)
        new_chars = sorted(chars - set(vocab))
        vocab.extend(new_chars)
        
        # Initialize word representations
        word_splits = {
            word: tuple(word) for word in word_freqs
        }
        
        # BPE training loop
        num_merges = self.vocab_size - len(vocab)
        merges = {}
        
        print(f"Starting BPE training: {num_merges:,} merges needed")
        print(f"Processing {len(word_freqs):,} unique word types")
        
        # Cache pair stats for each word to avoid recomputing
        pair_stats = {}
        for word, freq in word_freqs.items():
            pair_stats[word] = defaultdict(int)
            pairs_in_word = self._get_pairs(word_splits[word])
            for pair in pairs_in_word:
                pair_stats[word][pair] = freq
        
        for merge_idx in range(num_merges):
            # Count pairs from cache
            pairs = defaultdict(int)
            for word in word_freqs:
                for pair, freq in pair_stats[word].items():
                    pairs[pair] += freq
            
            if not pairs:
                print(f"No more pairs to merge at step {merge_idx}")
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair in affected words only
            new_word_splits = {}
            affected_words = [word for word in word_freqs if best_pair in pair_stats[word]]
            
            for word in affected_words:
                new_word = []
                idx = 0
                word_tokens = word_splits[word]
                
                while idx < len(word_tokens):
                    try:
                        j = word_tokens.index(best_pair[0], idx)
                        new_word.extend(word_tokens[idx:j])
                        idx = j
                    except ValueError:
                        new_word.extend(word_tokens[idx:])
                        break
                    
                    if (idx < len(word_tokens) - 1 and 
                        word_tokens[idx + 1] == best_pair[1]):
                        new_word.append(best_pair[0] + best_pair[1])
                        idx += 2
                    else:
                        new_word.append(word_tokens[idx])
                        idx += 1
                
                new_word_splits[word] = tuple(new_word)
                
                # Update pair cache for this word
                pair_stats[word] = defaultdict(int)
                new_pairs = self._get_pairs(new_word_splits[word])
                for pair in new_pairs:
                    pair_stats[word][pair] = word_freqs[word]
            
            # Update word_splits
            for word in affected_words:
                word_splits[word] = new_word_splits[word]
            merges[best_pair] = len(vocab)
            vocab.append(best_pair[0] + best_pair[1])
            
            # Progress indicator
            if (merge_idx + 1) % 500 == 0:
                progress = (merge_idx + 1) / num_merges * 100
                print(f"Progress: {merge_idx + 1:,}/{num_merges:,} merges ({progress:.1f}%) - Last: '{best_pair[0]}' + '{best_pair[1]}'")
            elif (merge_idx + 1) % 100 == 0:
                print(f"  {merge_idx + 1:,}/{num_merges:,} merges...", end='\r')
        
        # Update tokenizer state
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.merges = merges
        
        print(f"Training complete. Vocabulary size: {len(self.vocab)}")
        return self
    
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
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
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
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                # Skip special tokens in output
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
            
            # Post-process: Clean up <unk> tokens that are typically separators
            # This handles cases where spaces/punctuation become <unk> during generation
            import re
            decoded_text = re.sub(r'<unk>', ' ', decoded_text)
            decoded_text = re.sub(r'\s+', ' ', decoded_text).strip()
            
            return decoded_text
            
        except (KeyError, UnicodeDecodeError):
            # Fallback: return as-is if byte decoding fails
            import re
            clean_text = re.sub(r'<unk>', ' ', text)
            return re.sub(r'\s+', ' ', clean_text).strip()
    
    def save(self, path: str):
        """Save tokenizer to file."""
        # Convert tuple keys to strings for JSON serialization
        merges_serializable = {f"{k[0]} {k[1]}": v for k, v in self.merges.items()}
        
        tokenizer_data = {
            'vocab': self.vocab,
            'merges': merges_serializable,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = cls(
            vocab_size=tokenizer_data['vocab_size'],
            special_tokens=tokenizer_data['special_tokens']
        )
        
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.id_to_token = {int(idx): token for token, idx in tokenizer.vocab.items()}
        
        # Handle merges if they exist (for custom format), otherwise initialize empty
        if 'merges' in tokenizer_data:
            # Convert string keys back to tuples
            tokenizer.merges = {tuple(k.split(' ', 1)): v for k, v in tokenizer_data['merges'].items()}
        else:
            # No merges available (HuggingFace format), initialize empty
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
