"""
Custom PyTorch Dataset class for article topic classification.

This module handles:
- Reading articles from JSONL format (JSON Lines: one JSON object per line)
- Tokenizing text using DistilBERT tokenizer
- Creating attention masks for padding tokens
- Mapping topic labels to numeric IDs
"""

import json  # Parse JSON data
import logging  # Log dataset operations
from torch.utils.data import Dataset  # Base class for PyTorch datasets

logger = logging.getLogger(__name__)

class NewsDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing news articles.
    
    Features:
    - Loads articles from JSONL file
    - Tokenizes text (converts words to token IDs)
    - Pads/truncates to fixed length (256 tokens)
    - Creates attention masks (tells model which tokens are real)
    - Maps topic strings to numeric labels
    
    Attributes:
        samples (list): List of JSON objects from JSONL file
        tokenizer: Tokenizer instance (DistilBERT WordPiece tokenizer)
        max_length (int): Maximum sequence length (256 tokens)
        label2id (dict): Mapping from topic name to numeric ID
    """
    
    def __init__(self, path, tokenizer, max_length):
        """
        Initialize dataset by loading and storing samples.
        
        Args:
            path (str): Path to JSONL file (articles)
            tokenizer: Pre-trained tokenizer (e.g., DistilBERT)
            max_length (int): Maximum token sequence length
        """
        logger.info(f"Initializing NewsDataset from {path}")
        
        # Initialize empty list to store all samples
        self.samples = []
        
        # Read JSONL file: each line is a complete JSON object
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                # Parse JSON line and add to samples
                self.samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.samples)} samples")
        
        # Store tokenizer and max length as instance variables
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create mapping: topic string -> numeric label (for training)
        # These IDs are used as targets for the classification task
        self.label2id = {
            "World": 0,      # World news
            "Sports": 1,     # Sports articles
            "Business": 2,   # Business/Finance articles
            "Sci/Tech": 3,   # Science and Technology articles
        }
        logger.debug(f"Label mapping: {self.label2id}")
        logger.info(f"Dataset initialized with max_length={max_length}")

    def __len__(self):
        """
        Return total number of samples in dataset.
        
        Returns:
            int: Number of articles
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get and preprocess a single sample at given index.
        
        This method:
        1. Retrieves article at index
        2. Concatenates title and body (the article text)
        3. Tokenizes using DistilBERT tokenizer
        4. Pads/truncates to max_length
        5. Creates attention mask (1 for real tokens, 0 for padding)
        6. Converts topic label to numeric ID
        
        Args:
            idx (int): Index of sample to retrieve
            
        Returns:
            dict: Dictionary with keys:
                - input_ids: Token IDs (numeric representation of text)
                - attention_mask: Mask indicating real vs padding tokens
                - labels: Numeric topic ID (0-3)
        """
        # Get sample at index
        s = self.samples[idx]
        
        # Combine title and body into single text
        # Both are important: title is concise summary, body has full content
        text = s["title"] + " " + s["body"]

        # Tokenize the text:
        # - WordPiece tokenization breaks text into subword units
        # - Adds [CLS] token at start, [SEP] token at end (BERT convention)
        # - Truncates if longer than max_length
        # - Pads with [PAD] token if shorter than max_length
        enc = self.tokenizer(
            text,
            truncation=True,  # Truncate text longer than max_length
            padding="max_length",  # Pad text shorter than max_length
            max_length=self.max_length,  # Target length: 256 tokens
            return_tensors="pt",  # Return PyTorch tensors (not lists)
        )

        # Return dictionary with input_ids, attention_mask, and label
        # This format matches what the model expects in train_distilbert.py
        return {
            "input_ids": enc["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": enc["attention_mask"].squeeze(0),  # Remove batch dimension
            "labels": self.label2id[s["topic"]],  # Convert topic string to ID
        }
