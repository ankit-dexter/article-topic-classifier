"""
Article Topic Classification - Prediction Script

This module loads a pre-trained DistilBERT model and uses it to classify
article topics into one of four categories: World, Sports, Business, or Sci/Tech.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------- CONFIG -------- #
# Path to the pre-trained model artifacts (weights, config, tokenizer)
MODEL_DIR = "artifacts/distilbert"

# Maximum sequence length for tokenization (longer texts will be truncated)
MAX_LENGTH = 256

# Topic classification labels (must match training setup)
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
# ------------------------ #

def predict(title: str, body: str):
    """
    Classify an article into one of four topic categories.
    
    Args:
        title (str): Article headline/title
        body (str): Article content/body text
    
    Returns:
        dict: Contains:
            - 'prediction': The predicted topic label (str)
            - 'confidence': Confidence score for the prediction (float, 0-1)
            - 'all_probabilities': Probability scores for all labels (dict)
    """
    # Detect and use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained tokenizer and model from artifacts
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    
    # Move model to the appropriate device (GPU/CPU)
    model.to(device)
    # Set model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()

    # Combine title and body into a single text input
    text = title + " " + body

    # Tokenize the input text for model consumption
    inputs = tokenizer(
        text,
        truncation=True,  # Truncate text longer than MAX_LENGTH
        padding="max_length",  # Pad shorter sequences to MAX_LENGTH
        max_length=MAX_LENGTH,
        return_tensors="pt",  # Return PyTorch tensors
    )

    # Move tokenized inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference without gradient computation (faster, less memory)
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply softmax to convert logits into probability distribution (shape: 4,)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    # Build a dictionary mapping each label to its probability score
    prob_map = {
        LABELS[i]: round(probs[i].item(), 4)
        for i in range(len(LABELS))
    }

    # Find the index of the highest probability (predicted class)
    pred_idx = torch.argmax(probs).item()

    # Return structured prediction results
    return {
        "prediction": LABELS[pred_idx],
        "confidence": round(prob_map[LABELS[pred_idx]], 4),
        "all_probabilities": prob_map,
    }

if __name__ == "__main__":
    # Example article to demonstrate classification
    # This is a real-world test case: tech company earnings (ambiguous Business/Sci/Tech)
    
    title = "Tech firms report strong quarterly earnings"
    body = (
        "Several major technology companies reported better-than-expected earnings "
        "this quarter, driven by strong cloud computing demand and cost-cutting measures."
    )

    # Run the prediction function
    result = predict(title, body)

    # Display results in a readable format
    print("\nPrediction Result:")
    print(f"Predicted topic : {result['prediction']}")
    print(f"Confidence      : {result['confidence']}")
    print("\nProbabilities:")
    for label, prob in result["all_probabilities"].items():
        print(f"  {label:10s} → {prob}")
