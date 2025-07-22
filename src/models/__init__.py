"""
@file       : __init__.py
@package    : src.models
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : L1 model architecture package.
@details    : This package contains the model architecture components for the L1 language model,
              including the transformer blocks, attention mechanisms, and embedding layers.
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

from .transformer import L1Model, TransformerBlock, MultiHeadAttention
from .config import L1Config
from .embeddings import TokenEmbedding, PositionalEmbedding

__all__ = [
    "L1Model",
    "L1Config",
    "TransformerBlock", 
    "MultiHeadAttention",
    "TokenEmbedding",
    "PositionalEmbedding",
]
