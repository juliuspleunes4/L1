"""
@file       : __init__.py
@package    : src
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : L1 package initialization.
@details    : This package contains the core components of the L1 language model,
              including model architecture, training routines, and inference utilities.
              This is a PyTorch based implementation of a transformer language model
              with comprehensive training and inference capabilities.
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

__version__ = "0.1.0"
__author__ = "L1 Development Team"
__email__ = "l1@example.com"

from .models import L1Model, L1Config
from .training import Trainer, TrainingConfig
from .inference import InferenceEngine
from .data import TextDataset, Tokenizer

__all__ = [
    "L1Model",
    "L1Config", 
    "Trainer",
    "TrainingConfig",
    "InferenceEngine",
    "TextDataset",
    "Tokenizer",
]
