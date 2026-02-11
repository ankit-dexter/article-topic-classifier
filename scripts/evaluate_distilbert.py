"""
Evaluation script for DistilBERT article topic classifier.

This script:
1. Loads a trained model and tokenizer
2. Evaluates model performance on test dataset
3. Computes classification metrics: precision, recall, F1-score
4. Generates confusion matrix to show misclassifications
5. Reports model confidence statistics
"""

import json  # Parse JSON if needed
from scripts.calibration_analysis import compute_ece
import torch  # PyTorch deep learning framework
import logging  # Logging for evaluation progress
import numpy as np  # Numerical computations
from sklearn.metrics import classification_report, confusion_matrix  # Evaluation metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Load model
from scripts.calibration_analysis import coverage_accuracy_curve
from src.dataset import NewsDataset  # Custom dataset class

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------- EVALUATION CONFIGURATION -------- #
MODEL_DIR = "artifacts/distilbert"     # Directory containing trained model and tokenizer
DATA_PATH = "data/test.jsonl"     # Path to evaluation dataset (JSONL format)
BATCH_SIZE = 32                         # Batch size for evaluation (higher = faster)
MAX_LENGTH = 256                        # Maximum token sequence length (match training)
# ------------------------------------------ #

# Topic labels mapping (must match training labels)
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

def main():
    """
    Main evaluation function that:
    1. Loads trained model and tokenizer
    2. Evaluates on dataset
    3. Computes and prints metrics
    """

    # ========== SETUP DEVICE ==========
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========== LOAD MODEL & TOKENIZER ==========
    # Load pre-trained tokenizer from saved model directory
    logger.info(f"Loading tokenizer from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    logger.info("Tokenizer loaded successfully")
    
    # Load fine-tuned classification model
    logger.info(f"Loading model from {MODEL_DIR}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)  # Move to GPU if available
    model.eval()  # Set to evaluation mode (disable dropout, freeze batch norm)
    logger.info("Model loaded and set to evaluation mode")

    # ========== LOAD EVALUATION DATASET ==========
    # Create dataset instance that loads and tokenizes articles
    logger.info(f"Loading dataset from {DATA_PATH}")
    dataset = NewsDataset(DATA_PATH, tokenizer, MAX_LENGTH)
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Create DataLoader for batching (no shuffle needed for evaluation)
    logger.info(f"Creating DataLoader with batch_size={BATCH_SIZE}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    logger.info(f"DataLoader created: {len(loader)} batches")

    # ========== INITIALIZE PREDICTION CONTAINERS ==========
    # Lists to store predictions, true labels, and confidence scores
    all_preds = []       # Predicted topic IDs (0-3)
    all_labels = []      # True topic IDs from dataset
    all_probs = []       # Confidence scores (max softmax probability)

    # ========== EVALUATION LOOP ==========
    logger.info("\n" + "=" * 80)
    logger.info("Starting evaluation...")
    logger.info("=" * 80)
    
    # torch.no_grad() disables gradient computation (faster, less memory)
    # Since we're not training, we don't need gradients
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Extract true labels BEFORE moving to device
            # (labels are not part of model input, only for evaluation)
            labels = batch["labels"].numpy()
            
            # Move batch tensors to GPU if available
            batch = {k: v.to(device) for k, v in batch.items()}

            # ===== FORWARD PASS (NO GRADIENTS) =====
            # Feed tokenized text to model
            # Note: we don't pass 'labels' to model (no loss computation needed)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            # outputs.logits shape: [batch_size, 4] - raw scores for 4 classes
            # Example: [[0.5, 2.3, 0.1, -0.2], [1.2, 0.3, 1.5, 0.4], ...]

            # ===== COMPUTE PROBABILITIES =====
            # Softmax converts logits to probabilities (sum to 1)
            # dim=1: apply softmax across the 4 class dimensions
            probs = torch.softmax(outputs.logits, dim=1)
            # Shape: [batch_size, 4], values in range [0, 1]
            
            # Get predicted class: argmax of probabilities
            # Returns class ID with highest probability (0-3)
            preds = torch.argmax(probs, dim=1)

            # ===== COLLECT PREDICTIONS & CONFIDENCE =====
            # Store predictions (convert from GPU tensor to CPU numpy)
            all_preds.extend(preds.cpu().numpy())
            
            # Store true labels
            all_labels.extend(labels)
            
            # Store maximum probability (confidence) for each prediction
            # probs.max(dim=1) returns (values, indices)
            # We only want the values (confidence scores)
            all_probs.extend(probs.cpu().numpy())
            
            # Log progress every batch
            if (batch_idx + 1) % max(1, len(loader) // 4) == 0:
                logger.info(f"  Processed batch {batch_idx+1}/{len(loader)}")

    # ========== COMPUTE & DISPLAY METRICS ==========
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete - Computing Metrics")
    logger.info("=" * 80)
    
    # ===== CLASSIFICATION REPORT =====
    # Precision: TP / (TP + FP) - How many predicted positives were correct?
    # Recall: TP / (TP + FN) - How many actual positives did we find?
    # F1-score: Harmonic mean of precision and recall
    logger.info("\n=== CLASSIFICATION REPORT ===")
    report = classification_report(all_labels, all_preds, target_names=LABELS)
    print(report)
    logger.info(report)

    # ===== CONFUSION MATRIX =====
    # Shows misclassifications: rows = true labels, columns = predicted labels
    # Diagonal = correct predictions, off-diagonal = errors
    logger.info("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    logger.info(f"Confusion Matrix:\n{cm}")

    # ===== CONFIDENCE STATISTICS =====
    # Analyze model's certainty in its predictions
    logger.info("\n=== CONFIDENCE STATS ===")
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    confidences = np.max(all_probs, axis=1)

    avg_conf = np.mean(confidences)
    min_conf = np.min(confidences)
    max_conf = np.max(confidences)
    

    logger.info(f"Average confidence: {avg_conf:.3f} (higher = more certain)")
    logger.info(f"Min confidence: {min_conf:.3f} (least certain prediction)")
    logger.info(f"Max confidence: {max_conf:.3f} (most certain prediction)")
    
    print(f"Average confidence: {avg_conf:.3f}")
    print(f"Min confidence: {min_conf:.3f}")
    print(f"Max confidence: {max_conf:.3f}")
    
    logger.info("=" * 80)
    logger.info("Evaluation completed successfully!")
    logger.info("=" * 80)

    
    ece = compute_ece(np.array(all_probs), np.array(all_labels))
    print(f"Validation ECE: {ece:.4f}")
    curve = coverage_accuracy_curve(all_probs, all_labels)

    print("\nThreshold | Coverage | Accuracy")
    print("----------------------------------")
    for t, cov, acc in curve:
        print(f"{t:.2f}      | {cov:.3f}    | {acc:.3f}")


if __name__ == "__main__":
    main()
