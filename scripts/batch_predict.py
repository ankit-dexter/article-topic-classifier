import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.dataset import NewsDataset

# -------- CONFIG -------- #
# Directory where the fine-tuned DistilBERT model is stored
MODEL_DIR = "artifacts/distilbert"

# Input test data (JSONL format: one sample per line)
INPUT_JSONL = "data/test.jsonl"

# Output file where predictions will be written
OUTPUT_JSONL = "data/predictions_test.jsonl"

# Inference batch size
BATCH_SIZE = 32

# Maximum token length for model inputs
MAX_LENGTH = 256

# Ordered list of labels corresponding to model outputs
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# Confidence thresholds for automated decision-making
AUTO_ACCEPT_CONF = 0.85   # High confidence threshold
AUTO_ACCEPT_GAP = 0.20    # Minimum gap between top-1 and top-2 predictions
REVIEW_CONF = 0.60        # Minimum confidence to send for human review
# ------------------------ #


def apply_confidence_rules(prob_map):
    """
    Apply business rules on prediction probabilities to decide whether
    a prediction should be auto-accepted, reviewed, or rejected.

    Args:
        prob_map (dict): Mapping of label -> probability

    Returns:
        decision (str): auto_accept | needs_review | reject
        top2_label (str): Second most likely label
        top2_gap (float): Confidence gap between top-1 and top-2 labels
    """
    # Sort labels by probability (highest first)
    ranked = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)

    # Extract top-1 and top-2 predictions
    (top1_label, top1_p), (top2_label, top2_p) = ranked[0], ranked[1]

    # Difference between top-1 and top-2 confidence
    top2_gap = round(top1_p - top2_p, 4)

    # Decision logic based on confidence thresholds
    if top1_p >= AUTO_ACCEPT_CONF and top2_gap >= AUTO_ACCEPT_GAP:
        decision = "auto_accept"
    elif top1_p >= REVIEW_CONF:
        decision = "needs_review"
    else:
        decision = "reject"

    return decision, top2_label, top2_gap


def main():
    # Select GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model and tokenizer...")
    # Load tokenizer and sequence classification model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # Move model to selected device and set evaluation mode
    model.to(device)
    model.eval()

    print(f"Loading input data: {INPUT_JSONL}")
    # Create dataset and dataloader for batch inference
    dataset = NewsDataset(INPUT_JSONL, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    print(f"Running batch inference (batch_size={BATCH_SIZE})")
    results = []

    # Disable gradient computation for faster inference
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            # Labels are not used during inference but kept for analysis/debugging
            labels = batch["labels"]

            # Move all tensors in batch to the selected device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass through the model
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            # Convert logits to probabilities
            probs = torch.softmax(outputs.logits, dim=1)

            # Process each sample in the batch
            for i in range(probs.size(0)):
                # Build a label -> probability mapping
                prob_map = {
                    LABELS[j]: round(probs[i, j].item(), 4)
                    for j in range(len(LABELS))
                }

                # Predicted label with highest confidence
                pred_label = max(prob_map, key=prob_map.get)
                confidence = prob_map[pred_label]

                # Apply confidence-based decision rules
                decision, top2_label, top2_gap = apply_confidence_rules(prob_map)

                # Store prediction result
                results.append({
                    "prediction": pred_label,
                    "confidence": confidence,
                    "decision": decision,
                    "top2_label": top2_label,
                    "top2_gap": top2_gap,
                    "probabilities": prob_map,
                })

            # Log progress every 5 batches
            if batch_idx % 5 == 0:
                print(f"  Processed batch {batch_idx}/{len(loader)}")

    print(f"Writing results to {OUTPUT_JSONL}")
    # Write predictions in JSONL format
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("✅ Batch inference completed successfully")


if __name__ == "__main__":
    # Entry point for script execution
    main()
