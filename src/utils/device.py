"""
@file       : device.py
@package    : src.utils
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Device utilities for L1 project.
@details    : This script provides utilities for managing devices (CPU/GPU)
              in the L1 project, including functions for device selection
              and tensor movement.

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
from typing import Union


def get_device(device_spec: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device_spec: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)
        
    Returns:
        PyTorch device object
    """
    if device_spec == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_spec)
    
    return device


def move_to_device(
    obj: Union[torch.Tensor, torch.nn.Module], 
    device: torch.device
) -> Union[torch.Tensor, torch.nn.Module]:
    """
    Move tensor or model to specified device.
    
    Args:
        obj: Tensor or model to move
        device: Target device
        
    Returns:
        Object moved to device
    """
    return obj.to(device)
