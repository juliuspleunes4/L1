#!/usr/bin/env python3
"""
@file       : fix_existing_tokenizer.py
@author     : J.J.G. Pleunes
@date       : 08/2025
@brief      : Fix existing tokenizers that have missing essential tokens.
@details    : This script patches existing tokenizer.json files to include
              essential tokens like spaces and punctuation that may be missing
              from older trained tokenizers.
@version    : 1.0
"""

import os
import json
import shutil
from typing import List

def fix_tokenizer_file(tokenizer_path: str) -> bool:
    """
    Fix an existing tokenizer file by adding missing essential tokens.
    
    Args:
        tokenizer_path: Path to the tokenizer.json file
        
    Returns:
        True if fixes were applied, False if no fixes needed
    """
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        return False
    
    # Backup the original
    backup_path = tokenizer_path.replace('.json', '_backup.json')
    if not os.path.exists(backup_path):
        shutil.copy(tokenizer_path, backup_path)
        print(f"💾 Backed up original to {backup_path}")
    
    # Load existing tokenizer
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    vocab = tokenizer_data.get('vocab', {})
    current_max_id = max(vocab.values()) if vocab else 3
    
    # Essential tokens that are commonly missing
    essential_tokens = [
        ' ',           # space
        '.',           # period
        ',',           # comma
        '!',           # exclamation
        '?',           # question
        ':',           # colon
        ';',           # semicolon
        '"',           # quote
        "'",           # apostrophe
        '(',           # parentheses
        ')',
        '-',           # dash
        '\n',          # newline
        '\t',          # tab
    ]
    
    added_count = 0
    for token in essential_tokens:
        if token not in vocab:
            current_max_id += 1
            vocab[token] = current_max_id
            added_count += 1
            print(f"   ➕ Added missing token: '{repr(token)}' -> {current_max_id}")
    
    if added_count > 0:
        # Update the tokenizer data
        tokenizer_data['vocab'] = vocab
        tokenizer_data['vocab_size'] = len(vocab)
        
        # Save the updated tokenizer
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Fixed tokenizer: added {added_count} essential tokens")
        print(f"📊 New vocab size: {len(vocab)}")
        return True
    else:
        print("✅ Tokenizer already contains all essential tokens")
        return False

def main():
    """Find and fix tokenizers in the project."""
    print("🔧 L1 Tokenizer Fix Utility")
    print("=" * 50)
    
    # Common tokenizer locations
    possible_paths = [
        "models/l1-gpu-compatible/tokenizer.json",
        "models/l1-v1/tokenizer.json",
        "data/processed/tokenizer.json",
    ]
    
    fixed_count = 0
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\n🔍 Checking {path}...")
            if fix_tokenizer_file(path):
                fixed_count += 1
        else:
            print(f"⏭️  Skipping {path} (not found)")
    
    print(f"\n🎉 Complete! Fixed {fixed_count} tokenizer(s)")
    
    if fixed_count > 0:
        print("\n📝 What was fixed:")
        print("   • Added missing space, punctuation, and newline tokens")
        print("   • Updated vocabulary size")
        print("   • Improved text generation quality")
        print("\n🧪 Test your model with: python generate_simple.py")

if __name__ == '__main__':
    main()
