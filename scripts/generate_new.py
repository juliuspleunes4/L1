#!/usr/bin/env python3
"""
Text generation script for L1 model.

Usage:
    python scripts/generate.py --model_path models/l1-v1/
    python scripts/generate.py --model_path models/l1-v1/ --prompt "The future of AI"
"""

import os
import sys
import argparse
import torch
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import L1Model, L1Config
from data import BPETokenizer


def load_model_and_tokenizer(model_path: str):
    """Load trained model and tokenizer."""
    # Load config
    config_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = L1Config(**config_dict)
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_path, 'tokenizer.json')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    # Load model
    model = L1Model(config)
    
    # Look for model weights
    model_weights_paths = [
        os.path.join(model_path, 'pytorch_model.bin'),
        os.path.join(model_path, 'model.safetensors'),
        os.path.join(model_path, 'model.pt'),
    ]
    
    model_weights_path = None
    for path in model_weights_paths:
        if os.path.exists(path):
            model_weights_path = path
            break
    
    if model_weights_path is None:
        # Look for latest checkpoint
        checkpoint_dirs = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]
        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
            checkpoint_path = os.path.join(model_path, latest_checkpoint)
            
            checkpoint_weights_paths = [
                os.path.join(checkpoint_path, 'pytorch_model.bin'),
                os.path.join(checkpoint_path, 'model.safetensors'),
                os.path.join(checkpoint_path, 'model.pt'),
            ]
            
            for path in checkpoint_weights_paths:
                if os.path.exists(path):
                    model_weights_path = path
                    break
    
    if model_weights_path is None:
        raise FileNotFoundError(f"No model weights found in {model_path}")
    
    print(f"Loading model weights from: {model_weights_path}")
    
    # Load weights
    if model_weights_path.endswith('.safetensors'):
        # Handle safetensors if needed
        state_dict = torch.load(model_weights_path, map_location='cpu')
    else:
        state_dict = torch.load(model_weights_path, map_location='cpu')
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, tokenizer, config


def generate_text(model, tokenizer, config, prompt: str, max_new_tokens: int = 50, 
                 temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9):
    """Generate text using the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        generated_ids = input_ids[0].tolist()
        
        for _ in range(max_new_tokens):
            # Get model output
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits[top_k_indices] = top_k_logits
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated sequence
            generated_ids.append(next_token.item())
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Update input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Trim if too long
            if input_ids.size(1) > config.max_seq_length:
                input_ids = input_ids[:, -config.max_seq_length:]
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text with L1 model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--prompt', type=str, default="The future of artificial intelligence",
                       help='Text prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p sampling parameter')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    
    try:
        # Load model and tokenizer
        model, tokenizer, config = load_model_and_tokenizer(args.model_path)
        print(f"Model loaded successfully!")
        print(f"Vocabulary size: {len(tokenizer.vocab)}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        print(f"\nGenerating text with prompt: '{args.prompt}'")
        print("=" * 60)
        
        # Generate text
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            config=config,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        print(generated_text)
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
