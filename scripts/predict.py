#!/usr/bin/env python3
"""
Article Topic Classification - Standalone Prediction Script

This script demonstrates how to use the DistilBERT model for classifying article topics.
It loads a pre-trained model, applies confidence-based decision rules, and outputs predictions
with metadata for production-style routing (auto-accept, needs-review, or reject).

Usage:
    python predict.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------- CONFIG -------- #
# Model and tokenizer location
MODEL_DIR = "artifacts/distilbert"
# Maximum sequence length for tokenization
MAX_LENGTH = 256

# Article topic categories (must match training labels)
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# Confidence thresholds for decision routing
# HIGH confidence + CLEAR MARGIN -> auto-accept
AUTO_ACCEPT_CONF = 0.85  # Minimum confidence for auto-accept
AUTO_ACCEPT_GAP = 0.20   # Minimum gap between top 2 predictions
# MEDIUM confidence -> needs human review
REVIEW_CONF = 0.60       # Minimum confidence for review queue
# Below REVIEW_CONF -> reject
# ------------------------ #

def apply_confidence_rules(prob_map: dict):
    """
    Apply confidence-based decision rules for production routing.
    
    Args:
        prob_map (dict): Dictionary mapping labels to prediction probabilities
        
    Returns:
        dict: Decision metadata containing decision, reason, top2_label, and top2_gap
    """
    # Sort labels by probability (descending)
    ranked = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    (top1_label, top1_p), (top2_label, top2_p) = ranked[0], ranked[1]
    top2_gap = round(top1_p - top2_p, 4)

    if top1_p >= AUTO_ACCEPT_CONF and top2_gap >= AUTO_ACCEPT_GAP:
        return {
            "decision": "auto_accept",
            "reason": f"high_confidence_and_clear_margin (gap={top2_gap})",
            "top2_label": top2_label,
            "top2_gap": top2_gap,
        }

    if top1_p >= REVIEW_CONF:
        return {
            "decision": "needs_review",
            "reason": f"medium_confidence_or_ambiguous (gap={top2_gap})",
            "top2_label": top2_label,
            "top2_gap": top2_gap,
        }

    return {
        "decision": "reject",
        "reason": f"low_confidence (gap={top2_gap})",
        "top2_label": top2_label,
        "top2_gap": top2_gap,
    }


def predict(title: str, body: str):
    """
    Predict the topic of an article using DistilBERT model.
    
    Args:
        title (str): Article headline
        body (str): Article content/body text
        
    Returns:
        dict: Prediction result with confidence, all probabilities, and decision routing
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    text = title + " " + body

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]  # (4,)

    prob_map = {LABELS[i]: round(probs[i].item(), 4) for i in range(len(LABELS))}
    pred_label = max(prob_map, key=prob_map.get)
    confidence = prob_map[pred_label]

    decision_meta = apply_confidence_rules(prob_map)

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "all_probabilities": prob_map,
        **decision_meta,
    }


if __name__ == "__main__":
    # Try a few examples quickly by changing these:
    title = "Further sanctions on Russian banks impair Venezuela's access to hard currency"
    body = (
        "Context: "
        "Venezuela's export earnings stand to benefit from relatively high oil prices in the near term, "
        "partly driven by global geopolitical uncertainty. However, a moderate risk remains that the country "
        "will struggle to access its oil revenue, as a portion of these funds is channelled through the Russian "
        "financial system. The removal of several Russian banks from the SWIFT global payments network following "
        "the war in Ukraine could further restrict the Maduro regime's ability to access these proceeds. "
        "Trigger: "
        "Venezuela's access to oil revenue could face greater restrictions if Western powers impose further "
        "sanctions on Russia's financial system in an effort to intensify its economic isolation and end the war "
        "in Ukraine. "
        "Impact: "
        "Stronger sanctions on Russian banks would be likely to limit Venezuela's access to foreign exchange, "
        "leading to a drop in international reserves and tighter capital controls. This could trigger a new "
        "currency crisis and a possible return to recession, driving up exchange-rate volatility and fuelling "
        "renewed inflation. "
        "Mitigation: "
        "Businesses reliant on imported goods should factor in the risk of higher input costs when planning budgets "
        "and pricing strategies, particularly if tighter capital controls and currency depreciation materialise."
    )

    result = predict(title, body)

    print("\nPrediction Result:")
    print(f"Predicted topic : {result['prediction']}")
    print(f"Confidence      : {result['confidence']}")
    print(f"Decision        : {result['decision']}")
    print(f"Reason          : {result['reason']}")
    print(f"Top2 gap        : {result['top2_gap']} (runner-up: {result['top2_label']})")

    print("\nProbabilities:")
    for label, prob in result["all_probabilities"].items():
        print(f"  {label:10s} → {prob}")
