#!/usr/bin/env python3
"""
Text generation script for L1 model.

Usage:
    python scripts/generate.py --model checkpoints/best_model.pt --prompt "Hello, world!"
"""

import os
import sys
import argparse
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import L1Model, L1Config
from data import BPETokenizer


def load_model_and_tokenizer(checkpoint_path: str):
    """Load trained model and tokenizer."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load config
    config_dict = checkpoint['config']
    model_config = L1Config.from_dict(config_dict['model']) if 'model' in config_dict else L1Config()
    
    # Create model
    model = L1Model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(checkpoint_path), 'tokenizer.json')
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    return model, tokenizer


def generate_text(
    model: L1Model,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "auto"
):
    """Generate text using the model."""
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    print(f"Prompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens...")
    print("-" * 50)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    
    print(f"Generated text:\n{generated_text}")
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text with L1 model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Input prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=50,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p sampling parameter')
    parser.add_argument('--device', type=str, default="auto",
                       help='Device to use for generation')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Generate text
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device
    )


if __name__ == '__main__':
    main()
