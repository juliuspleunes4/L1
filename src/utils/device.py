"""
Device utilities for L1 project.
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
