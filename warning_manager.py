#!/usr/bin/env python3
"""
@file       : warning_manager.py
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Warning management utility for L1 training.
@details    : This module provides a way to manage and suppress
              warnings during the training process, ensuring that
              only relevant warnings are displayed to the user.
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

import warnings
import logging
import sys
from functools import wraps

class WarningManager:
    """Manages warnings for L1 training"""
    
    def __init__(self, suppress_level="medium"):
        """
        Initialize warning manager
        
        Args:
            suppress_level: "none", "low", "medium", "high"
                - none: Show all warnings
                - low: Suppress only very minor warnings
                - medium: Suppress most training-related warnings (default)
                - high: Suppress almost all warnings (not recommended)
        """
        self.suppress_level = suppress_level
        self.setup_warnings()
    
    def setup_warnings(self):
        """Configure warning filters based on suppress level"""
        
        if self.suppress_level == "none":
            # Show all warnings
            warnings.filterwarnings("default")
            return
        
        # Common non-critical warnings to suppress
        minor_warnings = [
            # PyTorch compilation warnings (usually not critical)
            ("ignore", ".*torch._dynamo.*", UserWarning),
            ("ignore", ".*_inductor.*", UserWarning),
            
            # Pandas future warnings (usually not critical for training)
            ("ignore", ".*pandas.*", FutureWarning),
            
            # NumPy precision warnings 
            ("ignore", ".*invalid value encountered.*", RuntimeWarning),
            
            # Threading warnings from data loading
            ("ignore", ".*Multiprocessing.*", UserWarning),
        ]
        
        medium_warnings = [
            # Mixed precision warnings when running on CPU
            ("ignore", ".*GradScaler.*CUDA.*not available.*", UserWarning),
            ("ignore", ".*autocast.*CUDA.*not available.*", UserWarning),
            
            # Model compilation warnings
            ("ignore", ".*torch.compile.*", UserWarning),
            ("ignore", ".*Compiler.*not found.*", RuntimeWarning),
            
            # DataLoader warnings  
            ("ignore", ".*pin_memory.*", UserWarning),
        ]
        
        aggressive_warnings = [
            # Suppress most FutureWarnings
            ("ignore", ".*", FutureWarning),
            
            # Suppress most UserWarnings 
            ("ignore", ".*", UserWarning),
        ]
        
        # Apply warning filters based on level
        if self.suppress_level in ["low", "medium", "high"]:
            for action, message, category in minor_warnings:
                warnings.filterwarnings(action, message, category)
        
        if self.suppress_level in ["medium", "high"]:
            for action, message, category in medium_warnings:
                warnings.filterwarnings(action, message, category)
                
        if self.suppress_level == "high":
            for action, message, category in aggressive_warnings:
                warnings.filterwarnings(action, message, category)
    
    def with_warning_context(self, func):
        """Decorator to run function with specific warning context"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                self.setup_warnings()
                return func(*args, **kwargs)
        return wrapper

def setup_training_warnings(level="medium"):
    """
    Setup warnings for L1 training
    
    Args:
        level: Warning suppression level
            - "none": Show all warnings (for debugging)
            - "low": Suppress only very minor warnings  
            - "medium": Suppress most training warnings (recommended)
            - "high": Suppress almost all warnings
    """
    manager = WarningManager(level)
    
    # Also setup logging to show important messages
    if level != "none":
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
        
        print(f"ℹ️  Warning suppression level: {level}")
        if level == "medium":
            print("   (Hiding non-critical training warnings, showing important errors)")
        elif level == "high":
            print("   (Hiding most warnings - only critical errors will show)")
    
    return manager

# Specific warning suppressors for common L1 issues
def suppress_torch_warnings():
    """Suppress common PyTorch warnings that don't affect training"""
    warnings.filterwarnings("ignore", ".*torch._dynamo.*")
    warnings.filterwarnings("ignore", ".*_inductor.*") 
    warnings.filterwarnings("ignore", ".*torch.compile.*")

def suppress_amp_warnings():
    """Suppress mixed precision warnings when running on CPU"""
    warnings.filterwarnings("ignore", ".*GradScaler.*CUDA.*not available.*")
    warnings.filterwarnings("ignore", ".*autocast.*CUDA.*not available.*")

def suppress_dataloader_warnings():
    """Suppress DataLoader related warnings"""
    warnings.filterwarnings("ignore", ".*pin_memory.*")
    warnings.filterwarnings("ignore", ".*num_workers.*")

if __name__ == "__main__":
    # Demo of warning management
    print("🔧 L1 Warning Manager Demo")
    print("Available suppression levels:")
    print("  - none: Show all warnings")
    print("  - low: Suppress minor warnings")  
    print("  - medium: Suppress most training warnings (recommended)")
    print("  - high: Suppress almost all warnings")
    
    level = input("\nEnter suppression level (default: medium): ").strip() or "medium"
    
    print(f"\nSetting up warnings with level: {level}")
    manager = setup_training_warnings(level)
    
    print("✅ Warning management configured!")
    print("\nTo use in your training scripts:")
    print("from warning_manager import setup_training_warnings")
    print("setup_training_warnings('medium')  # Add at top of script")
