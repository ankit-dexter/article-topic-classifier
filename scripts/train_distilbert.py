"""
Training script for DistilBERT-based article topic classifier.

This script performs the following steps:
1. Load configuration from YAML file
2. Initialize tokenizer and pre-trained DistilBERT model
3. Load and preprocess the training dataset
4. Fine-tune the model on article topic classification task
5. Save the trained model and tokenizer for inference
"""

# Import required libraries
import yaml  # Parse YAML configuration files
import torch  # PyTorch deep learning framework
import logging  # Log training progress and status
from pathlib import Path  # Handle file paths cross-platform
from torch.utils.data import DataLoader  # Batch data for training
from torch.optim import AdamW  # AdamW optimizer (PyTorch version, better than transformers version)

# Import from Hugging Face transformers library
from transformers import (
    AutoTokenizer,  # Automatically load the correct tokenizer
    AutoModelForSequenceClassification,  # Load classification model
    get_scheduler,  # Configure learning rate scheduler
)

# Import custom modules from src directory
from src.dataset import NewsDataset  # Custom dataset class that loads and tokenizes data
from src.utils import setup_logging  # Configure logging to file and console

def main():
    """
    Main training function that orchestrates the entire training pipeline.
    """
    
    # Initialize logging system (creates logs directory and log files)
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Print training start banner
    logger.info("=" * 80)
    logger.info("Starting training pipeline")
    logger.info("=" * 80)
    
    # ========== LOAD CONFIGURATION ==========
    # Load hyperparameters and settings from YAML configuration file
    logger.info("Loading configuration from config/train.yaml")
    with open("config/train.yaml") as f:
        cfg = yaml.safe_load(f)  # Parse YAML into Python dictionary
    logger.info(f"Config loaded successfully")
    logger.debug(f"Config: {cfg}")

    # ========== SET UP DEVICE (GPU or CPU) ==========
    # Check if NVIDIA GPU (CUDA) is available, use CPU as fallback
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========== LOAD TOKENIZER ==========
    # Tokenizer converts raw text into token IDs that the model understands
    # DistilBERT uses WordPiece tokenization
    logger.info(f"Loading tokenizer: {cfg['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    logger.info("Tokenizer loaded successfully")
    
    # ========== LOAD PRE-TRAINED MODEL ==========
    # Load DistilBERT: a lightweight BERT model (40% smaller, 60% faster)
    # Add classification head on top for 4-class topic classification
    # Model learns general language understanding + fine-tunes for our task
    logger.info(f"Loading model: {cfg['model']['name']} with {cfg['model']['num_labels']} labels")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],  # 4 topic classes
    ).to(device)  # Move model to GPU for faster computation
    logger.info(f"Model moved to {device}")

    # ========== LOAD DATASET ==========
    # Create dataset instance that:
    # - Reads JSONL file (JSON Lines format - one JSON object per line)
    # - Tokenizes text (converts to numeric representations)
    # - Creates attention masks (tells model which tokens are real vs padding)
    # - Maps topic labels to numeric IDs
    logger.info(f"Loading dataset from {cfg['data']['train_jsonl']}")
    dataset = NewsDataset(
        cfg["data"]["train_jsonl"],
        tokenizer,
        cfg["training"]["max_length"],  # Truncate/pad to 256 tokens
    )
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # ========== CREATE DATA LOADER ==========
    # DataLoader handles:
    # - Batching: groups samples together for efficient GPU processing
    # - Shuffling: randomizes order to prevent overfitting
    # - Parallel loading: can load data while GPU processes previous batch
    logger.info(f"Creating DataLoader with batch_size={cfg['training']['batch_size']}")
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],  # 16 samples per batch
        shuffle=True,  # Randomize batch order each epoch
    )
    logger.info(f"DataLoader created: {len(loader)} batches")

    # ========== SETUP OPTIMIZER ==========
    # AdamW optimizer: adaptive learning rate for each parameter
    # Weight decay (L2 regularization): prevents overfitting by penalizing large weights
    logger.info(f"Setting up optimizer (lr={cfg['training']['learning_rate']}, weight_decay={cfg['training']['weight_decay']})")
    optimizer = AdamW(
        model.parameters(),  # Which parameters to update
        lr=cfg["training"]["learning_rate"],  # Learning rate: 0.00002 (small for fine-tuning)
        weight_decay=cfg["training"]["weight_decay"],  # L2 penalty: 0.01
    )

    # ========== SETUP LEARNING RATE SCHEDULER ==========
    # Scheduler adjusts learning rate during training:
    # - Warmup phase (first 10% of steps): gradually increase LR to prevent instability
    # - Linear decay (remaining 90%): slowly decrease LR for finer-grained updates
    # This helps model converge better and generalize
    num_training_steps = cfg["training"]["epochs"] * len(loader)
    warmup_steps = int(cfg["training"]["warmup_ratio"] * num_training_steps)
    logger.info(f"Total training steps: {num_training_steps}, Warmup steps: {warmup_steps}")
    
    scheduler = get_scheduler(
        "linear",  # Linear decay after warmup
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    logger.info("Scheduler configured")

    # ========== TRAINING LOOP ==========
    logger.info("=" * 80)
    logger.info(f"Starting training for {cfg['training']['epochs']} epochs")
    logger.info("=" * 80)
    
    # Set model to training mode (enables dropout, updates batch norm stats, etc.)
    model.train()
    
    # Outer loop: iterate through epochs (full passes through data)
    for epoch in range(cfg["training"]["epochs"]):
        logger.info(f"\nEpoch {epoch+1}/{cfg['training']['epochs']}")
        total_loss = 0
        num_batches = 0
        
        # Inner loop: iterate through batches
        for batch_idx, batch in enumerate(loader):
            # Move batch tensors to GPU if available
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # ===== FORWARD PASS =====
            # Feed input_ids, attention_mask to model
            # Model outputs: logits (raw predictions) for each class
            # Cross-entropy loss: measures how wrong predictions are vs true labels
            out = model(**batch)  # Equivalent to model(input_ids=..., attention_mask=..., labels=...)
            loss = out.loss
            
            # ===== BACKWARD PASS (Gradient Computation) =====
            # Compute gradients: how much each weight contributed to the loss
            loss.backward()
            
            # ===== PARAMETER UPDATE =====
            # Use gradients to update model weights (gradient descent)
            optimizer.step()
            
            # ===== UPDATE LEARNING RATE =====
            # Adjust learning rate according to schedule
            scheduler.step()
            
            # ===== RESET GRADIENTS =====
            # Clear gradients to prevent them from accumulating
            optimizer.zero_grad()

            # Accumulate loss for epoch average
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress every ~20% of batches (for visibility without too much logging)
            if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
                avg_loss = total_loss / num_batches
                logger.info(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {avg_loss:.4f}")

        # Calculate and log epoch average loss
        epoch_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch+1} completed | Avg Loss: {epoch_loss:.4f}")

    # ========== SAVE TRAINED MODEL ==========
    logger.info("\n" + "=" * 80)
    logger.info(f"Training completed! Saving model to {cfg['output']['dir']}")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    
    # Save model weights and configuration (can be loaded with from_pretrained)
    model.save_pretrained(cfg["output"]["dir"])
    
    # Save tokenizer (needed for inference to tokenize new text)
    tokenizer.save_pretrained(cfg["output"]["dir"])
    logger.info("[SUCCESS] Model and tokenizer saved successfully")

if __name__ == "__main__":
    main()
