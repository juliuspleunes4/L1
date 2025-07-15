"""
Logging utilities for L1 project.
"""

import os
import logging
import sys
from typing import Optional
from datetime import datetime


def setup_logging(
    output_dir: str, 
    log_level: str = "INFO",
    log_to_file: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        output_dir: Directory to save log files
        log_level: Logging level
        log_to_file: Whether to log to file
        
    Returns:
        Configured logger
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    logger = logging.getLogger("L1")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"training_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance."""
    if name is None:
        name = "L1"
    return logging.getLogger(name)
