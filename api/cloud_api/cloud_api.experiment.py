#!/usr/bin/env python3
"""
ECS Fargate Inference Service
Loads model from S3 registry using latest.json pointer.
"""

import os
import json
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict

import torch
import boto3
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==============================
# Configuration
# ==============================

REGISTRY_BUCKET = os.getenv("REGISTRY_BUCKET")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/topicclf_model")

MAX_LENGTH = 256
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

AUTO_ACCEPT_CONF = 0.85
AUTO_ACCEPT_GAP = 0.20
REVIEW_CONF = 0.60

tokenizer = None
model = None
device = None

s3 = boto3.client("s3")


# ==============================
# Model Loading Logic
# ==============================

def download_model_from_registry():
    """
    1. Read pointers/latest.json
    2. Extract current_version
    3. Download models/<version>/ recursively
    """

    if not REGISTRY_BUCKET:
        raise RuntimeError("REGISTRY_BUCKET env variable not set")

    # Read latest.json
    obj = s3.get_object(
        Bucket=REGISTRY_BUCKET,
        Key="pointers/latest.json"
    )
    latest = json.loads(obj["Body"].read())
    version = latest.get("current_version")

    if not version:
        raise RuntimeError("latest.json has no current_version")

    prefix = f"models/{version}/"
    local_dir = Path(MODEL_CACHE_DIR) / version

    # Clean previous cache
    if local_dir.exists():
        shutil.rmtree(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=REGISTRY_BUCKET,
        Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            filename = key.split("/")[-1]
            local_path = local_dir / filename

            s3.download_file(
                REGISTRY_BUCKET,
                key,
                str(local_path)
            )

    if not (local_dir / "config.json").exists():
        raise RuntimeError("Downloaded model is invalid")

    return str(local_dir)


# ==============================
# FastAPI Lifespan
# ==============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🔄 Loading model from S3 registry")

    local_model_path = download_model_from_registry()

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)

    model.to(device)
    model.eval()

    print(f"✅ Model loaded: {local_model_path} on {device}")

    yield

    print("🛑 API shutting down")


app = FastAPI(
    title="Article Topic Classification API",
    version="2.0.0",
    lifespan=lifespan
)


# ==============================
# Schemas
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
# Business Logic
# ==============================

def apply_confidence_rules(prob_map):
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


def run_prediction(article: Article) -> Prediction:
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

    prob_map = {
        LABELS[i]: round(probs[i].item(), 4)
        for i in range(len(LABELS))
    }

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
# Endpoints
# ==============================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(article: Article):
    return run_prediction(article)


@app.post("/batch_predict", response_model=List[Prediction])
def batch_predict(articles: List[Article]):
    return [run_prediction(a) for a in articles]