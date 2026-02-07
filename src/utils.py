"""
Utility functions for the training pipeline.

This module provides logging configuration that writes:
- Detailed logs to file (debug level) for troubleshooting
- Summary logs to console (info level) for user visibility
"""

import logging  # Python's standard logging module
import sys  # Access system streams (stdout, stderr)
from pathlib import Path  # Cross-platform file path handling
from datetime import datetime  # Generate timestamps for log files

def setup_logging(log_dir="logs", log_level=logging.INFO):
    """
    Configure logging with both file and console handlers.
    
    Creates a logs directory and writes:
    1. Detailed logs to file with timestamp (for debugging)
    2. Summary logs to console (for user feedback)
    
    Args:
        log_dir (str): Directory to save log files. Default: "logs"
        log_level (int): Console logging level. Default: logging.INFO
                        - logging.DEBUG: Most detailed (all messages)
                        - logging.INFO: Key milestones and progress
                        - logging.WARNING: Issues that might be problems
                        - logging.ERROR: Serious problems
    
    Returns:
        logging.Logger: Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    # parents=True: create parent directories if needed
    # exist_ok=True: don't error if directory already exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Get root logger (logs from all modules will go here)
    logger = logging.getLogger()
    
    # Set root logger to DEBUG so handlers can filter as needed
    logger.setLevel(log_level)
    
    # Clear any existing handlers (prevents duplicate logs)
    logger.handlers.clear()
    
    # Generate timestamp for unique log filename
    # Format: training_YYYYMMDD_HHMMSS.log (e.g., training_20260208_143022.log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"training_{timestamp}.log"
    
    # ===== CREATE FORMATTERS =====
    # Formatters determine what information is logged
    
    # Detailed format for file: includes timestamp, module name, level
    # Example: "2026-02-08 14:30:22 - src.dataset - INFO - Loaded 500 samples"
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simple format for console: just level and message
    # Example: "INFO - Model loaded successfully"
    simple_formatter = logging.Formatter(
        fmt='%(levelname)s - %(message)s'
    )
    
    # ===== CREATE FILE HANDLER =====
    # Writes ALL logs to file (maximum detail for debugging)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Capture all messages (including DEBUG)
    file_handler.setFormatter(detailed_formatter)  # Use detailed format
    logger.addHandler(file_handler)
    
    # ===== CREATE CONSOLE HANDLER =====
    # Writes logs to console (limited detail for user visibility)
    # Force UTF-8 encoding to support emojis and special characters on Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)  # Use specified level (default: INFO)
    console_handler.setFormatter(simple_formatter)  # Use simple format
    logger.addHandler(console_handler)
    
    # Log initial message to confirm setup
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger
