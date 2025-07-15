"""
Tokenizer implementations for L1 model.
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
    
    def train(self, texts: List[str]) -> 'BPETokenizer':
        """
        Train BPE tokenizer on text corpus.
        
        Args:
            texts: List of texts to train on
            
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
        
        # Initialize vocabulary with character-level tokens
        vocab = list(self.special_tokens.keys())
        
        # Add all unique characters
        chars = set()
        for word in word_freqs:
            chars.update(word)
        vocab.extend(sorted(chars))
        
        # Initialize word representations
        word_splits = {
            word: tuple(word) for word in word_freqs
        }
        
        # BPE training loop
        num_merges = self.vocab_size - len(vocab)
        merges = {}
        
        for i in range(num_merges):
            # Count pairs
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                word_tokens = word_splits[word]
                pairs_in_word = self._get_pairs(word_tokens)
                for pair in pairs_in_word:
                    pairs[pair] += freq
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair in all words
            new_word_splits = {}
            for word in word_freqs:
                new_word = []
                i = 0
                word_tokens = word_splits[word]
                
                while i < len(word_tokens):
                    try:
                        j = word_tokens.index(best_pair[0], i)
                        new_word.extend(word_tokens[i:j])
                        i = j
                    except ValueError:
                        new_word.extend(word_tokens[i:])
                        break
                    
                    if (i < len(word_tokens) - 1 and 
                        word_tokens[i + 1] == best_pair[1]):
                        new_word.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_word.append(word_tokens[i])
                        i += 1
                
                new_word_splits[word] = tuple(new_word)
            
            word_splits = new_word_splits
            merges[best_pair] = len(vocab)
            vocab.append(best_pair[0] + best_pair[1])
            
            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1}/{num_merges} merges")
        
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
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append('<unk>')
        
        text = ''.join(tokens)
        
        # Decode bytes
        try:
            text_bytes = bytearray([self.byte_decoder[c] for c in text])
            return text_bytes.decode('utf-8', errors='replace')
        except KeyError:
            return text
    
    def save(self, path: str):
        """Save tokenizer to file."""
        tokenizer_data = {
            'vocab': self.vocab,
            'merges': self.merges,
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
        tokenizer.merges = {tuple(k.split()): v for k, v in tokenizer_data['merges'].items()}
        
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
