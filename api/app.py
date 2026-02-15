#!/usr/bin/env python3
"""
Article Topic Classification API

This service performs inference using a fine-tuned DistilBERT model.
It supports two model loading modes:

1. Local filesystem (development / initial EC2 deployment)
2. S3-based model registry (production ECS deployment)

In production, the model is stored in S3 as a versioned artifact:
    s3://bucket/models/topicclf/<version>/model.tar.gz

A pointer file (latest.json) determines which model is active:
    s3://bucket/models/topicclf/latest.json

This allows:
- Versioned model management
- Zero Docker image rebuild for model updates
- Clean separation of inference and training
"""

import os
import json
import tarfile
import shutil
from pathlib import Path
from urllib.parse import urlparse

import torch
import boto3
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==============================
# Configuration
# ==============================

# MODEL_URI can be:
# - Local directory (default: artifacts/distilbert)
# - s3://bucket/path/model.tar.gz
# - s3://bucket/path/latest.json
MODEL_URI = os.getenv("MODEL_URI", "artifacts/distilbert")

# Local directory where S3 models are downloaded and extracted
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/topicclf_model")

# Maximum input token length
MAX_LENGTH = 256

# Label order must match training configuration
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# Confidence routing thresholds
AUTO_ACCEPT_CONF = 0.85
AUTO_ACCEPT_GAP = 0.20
REVIEW_CONF = 0.60


# ==============================
# Global Objects (loaded once)
# ==============================

tokenizer = None
model = None
device = None

# S3 client (credentials resolved via IAM role in ECS/EC2)
s3 = boto3.client("s3")


# ==============================
# Model Download Utilities
# ==============================

def is_s3_uri(uri: str) -> bool:
    """Check whether URI is an S3 path."""
    return uri.startswith("s3://")


def download_s3_object(bucket: str, key: str, destination: Path):
    """
    Download a single object from S3 to local filesystem.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(destination))


def prepare_model_from_uri(model_uri: str) -> str:
    """
    Resolve MODEL_URI into a local filesystem path
    containing HuggingFace model files.

    Supports:
    - Local directory
    - S3 model artifact (.tar.gz)
    - S3 latest.json pointer

    Returns:
        str → local path to model directory
    """

    # -----------------------------
    # Case 1: Local directory
    # -----------------------------
    if not is_s3_uri(model_uri):
        if not Path(model_uri).exists():
            raise FileNotFoundError(f"Local MODEL_URI not found: {model_uri}")
        return model_uri

    # -----------------------------
    # Case 2: S3 path
    # -----------------------------
    parsed = urlparse(model_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    cache_root = Path(MODEL_CACHE_DIR)
    cache_root.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Case 2A: latest.json pointer
    # -----------------------------
    if key.endswith(".json"):
        local_json = cache_root / "latest.json"
        download_s3_object(bucket, key, local_json)

        with open(local_json, "r") as f:
            metadata = json.load(f)

        # latest.json must contain:
        # { "artifact_uri": "s3://bucket/.../model.tar.gz" }
        artifact_uri = metadata.get("artifact_uri")
        if not artifact_uri:
            raise ValueError("latest.json missing 'artifact_uri' field")

        # Recursively resolve artifact URI
        return prepare_model_from_uri(artifact_uri)

    # -----------------------------
    # Case 2B: Direct tar.gz artifact
    # -----------------------------
    if not (key.endswith(".tar.gz") or key.endswith(".tgz")):
        raise ValueError("S3 MODEL_URI must point to .tar.gz or latest.json")

    local_tar = cache_root / "model.tar.gz"
    download_s3_object(bucket, key, local_tar)

    extract_dir = cache_root / "extracted"

    # Clean extraction directory to avoid mixing versions
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Extract model archive
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    # Find directory containing HuggingFace model files
    for root, dirs, files in os.walk(extract_dir):
        if "config.json" in files:
            return root

    raise RuntimeError("Could not locate HuggingFace model files after extraction")


# ==============================
# FastAPI Lifespan Manager
# ==============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifecycle hook.

    Runs once on startup:
    - Selects device (GPU if available)
    - Resolves MODEL_URI (local or S3)
    - Loads tokenizer + model into memory
    - Switches model to evaluation mode

    Ensures:
    - Model is loaded once
    - No reloading per request
    """

    global tokenizer, model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔄 Loading model from: {MODEL_URI}")

    local_model_path = prepare_model_from_uri(MODEL_URI)

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)

    model.to(device)
    model.eval()

    print(f"✅ Model successfully loaded on {device}")

    yield

    print("🛑 API shutting down")


# ==============================
# FastAPI App Definition
# ==============================

app = FastAPI(
    title="Article Topic Classification API",
    description="DistilBERT-based topic classifier with confidence-based routing",
    version="1.1.0",
    lifespan=lifespan,
)


# ==============================
# Request / Response Schemas
# ==============================

class Article(BaseModel):
    title: str
    body: str


class Prediction(BaseModel):
    prediction: str
    confidence: float
    decision: str
    top2_label: str
    top2_gap: float
    probabilities: Dict[str, float]


# ==============================
# Confidence Routing Logic
# ==============================

def apply_confidence_rules(prob_map):
    """
    Apply business rules to determine routing decision.
    """

    ranked = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    (top1_label, top1_p), (top2_label, top2_p) = ranked[0], ranked[1]

    top2_gap = round(top1_p - top2_p, 4)

    if top1_p >= AUTO_ACCEPT_CONF and top2_gap >= AUTO_ACCEPT_GAP:
        decision = "auto_accept"
    elif top1_p >= REVIEW_CONF:
        decision = "needs_review"
    else:
        decision = "reject"

    return decision, top2_label, top2_gap


# ==============================
# Core Inference Logic
# ==============================

def run_prediction(article: Article) -> Prediction:
    """
    End-to-end inference pipeline:
    - Merge title + body
    - Tokenize
    - Run forward pass
    - Apply routing rules
    """

    text = article.title + " " + article.body

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
        probs = torch.softmax(outputs.logits, dim=1)[0]

    prob_map = {LABELS[i]: round(probs[i].item(), 4) for i in range(len(LABELS))}

    pred_label = max(prob_map, key=prob_map.get)
    confidence = prob_map[pred_label]

    decision, top2_label, top2_gap = apply_confidence_rules(prob_map)

    return Prediction(
        prediction=pred_label,
        confidence=confidence,
        decision=decision,
        top2_label=top2_label,
        top2_gap=top2_gap,
        probabilities=prob_map,
    )


# ==============================
# API Endpoints
# ==============================

@app.get("/health")
def health():
    """Health check endpoint (used by ECS)."""
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(article: Article):
    """Classify a single article."""
    return run_prediction(article)


@app.post("/batch_predict", response_model=List[Prediction])
def batch_predict(articles: List[Article]):
    """Batch inference endpoint."""
    return [run_prediction(article) for article in articles]
