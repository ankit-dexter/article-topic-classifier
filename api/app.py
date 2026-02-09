#!/usr/bin/env python3
"""
Article Topic Classification API

A FastAPI-based REST API for classifying article topics using DistilBERT.

This service performs **inference only** (no training) and applies
confidence-based rules to decide how predictions should be routed:
- auto_accept   → High confidence, safe to use directly
- needs_review  → Medium confidence, requires human validation
- reject        → Low confidence, prediction should not be trusted

Endpoints:
    POST /predict        - Classify a single article
    POST /batch_predict  - Classify multiple articles in one request

Run locally:
    uvicorn app:app --reload
"""

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Global objects (initialized once at startup to avoid reloading per request)
tokenizer = None
model = None
device = None

# ---------------- CONFIG ---------------- #
# Path to the fine-tuned DistilBERT model
MODEL_DIR = "artifacts/distilbert"

# Maximum token length for inputs
MAX_LENGTH = 256

# Output labels in the same order as the model logits
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# Confidence thresholds for routing predictions
AUTO_ACCEPT_CONF = 0.85   # Very confident prediction
AUTO_ACCEPT_GAP = 0.20    # Clear separation from second-best class
REVIEW_CONF = 0.60        # Minimum confidence for human review
# --------------------------------------- #

# ---------- Lifespan (startup / shutdown) ---------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager.

    Runs once when the app starts:
    - Selects device (GPU / CPU)
    - Loads tokenizer and model into memory
    - Switches model to evaluation mode

    Ensures model is NOT reloaded on every request.
    """
    global tokenizer, model, device

    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and trained model from disk
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # Move model to selected device and disable training-specific layers
    model.to(device)
    model.eval()

    print(f"✅ Model loaded on {device}")

    yield  # ---- API is running and serving requests here ----

    # Optional cleanup logic during shutdown
    print("🛑 Shutting down API")

# ---------- App ---------- #

app = FastAPI(
    title="Article Topic Classification API",
    description="DistilBERT-based topic classifier with confidence rules",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------- Request / Response Models ---------- #

class Article(BaseModel):
    """
    Input schema for article classification.
    Title and body are combined and fed to the model.
    """
    title: str
    body: str

class Prediction(BaseModel):
    """
    Output schema returned by the API.
    Includes prediction confidence and routing decision.
    """
    prediction: str
    confidence: float
    decision: str
    top2_label: str
    top2_gap: float
    probabilities: Dict[str, float]

# ---------- Confidence Rules ---------- #

def apply_confidence_rules(prob_map):
    """
    Apply business logic on model probabilities to decide routing.

    Args:
        prob_map (dict): Mapping of label -> probability

    Returns:
        decision (str): auto_accept | needs_review | reject
        top2_label (str): Second most probable label
        top2_gap (float): Confidence gap between top-1 and top-2
    """
    # Sort labels by confidence (highest first)
    ranked = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    (top1_label, top1_p), (top2_label, top2_p) = ranked[0], ranked[1]

    # Difference between top-1 and top-2 predictions
    top2_gap = round(top1_p - top2_p, 4)

    if top1_p >= AUTO_ACCEPT_CONF and top2_gap >= AUTO_ACCEPT_GAP:
        decision = "auto_accept"
    elif top1_p >= REVIEW_CONF:
        decision = "needs_review"
    else:
        decision = "reject"

    return decision, top2_label, top2_gap

# ---------- Core Prediction Logic ---------- #

def run_prediction(article: Article) -> Prediction:
    """
    End-to-end prediction pipeline:
    - Text preprocessing
    - Tokenization
    - Model inference (no gradients)
    - Confidence-based routing decision
    """
    # Merge title and body into a single input string
    text = article.title + " " + article.body

    # Convert text to model-ready tensors
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Move input tensors to CPU/GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference mode: no gradients, forward pass only
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    # Build label → probability mapping
    prob_map = {LABELS[i]: round(probs[i].item(), 4) for i in range(len(LABELS))}

    # Highest confidence label
    pred_label = max(prob_map, key=prob_map.get)
    confidence = prob_map[pred_label]

    # Apply routing rules
    decision, top2_label, top2_gap = apply_confidence_rules(prob_map)

    return Prediction(
        prediction=pred_label,
        confidence=confidence,
        decision=decision,
        top2_label=top2_label,
        top2_gap=top2_gap,
        probabilities=prob_map,
    )

# ---------- API Endpoints ---------- #

@app.post("/predict", response_model=Prediction)
def predict(article: Article):
    """
    Classify a single article.

    Example:
        POST /predict
        {
          "title": "Tech stocks rally",
          "body": "Markets reacted positively..."
        }
    """
    return run_prediction(article)

@app.post("/batch_predict", response_model=List[Prediction])
def batch_predict(articles: List[Article]):
    """
    Classify multiple articles in one request (batch inference).

    Useful for:
    - Bulk scoring jobs
    - Offline pipelines
    - Human-in-the-loop workflows
    """
    return [run_prediction(article) for article in articles]
