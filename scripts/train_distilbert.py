"""
Production Training Script with:
- Incremental fine-tuning
- Evaluation + promotion gating
- Registry update with metrics
- CloudWatch structured logging
"""

# ========= FIX FOR LOCAL EXECUTION =========
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# ============================================

import os
import json
import yaml
import time
import torch
import boto3
import logging
import numpy as np
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import NewsDataset
from src.utils import setup_logging
from src.registry import ModelRegistry


PROMOTION_THRESHOLD = 0.80  # Adjust as needed


# =====================================================
# DATASET DOWNLOAD (CLOUD)
# =====================================================
def download_snapshot_from_s3(raw_bucket, logger):
    s3 = boto3.client("s3")
    key = "processed/train.jsonl"
    local_path = "/tmp/train.jsonl"

    logger.info(f"[DATASET][CLOUD] Downloading {key}")
    s3.download_file(raw_bucket, key, local_path)

    return local_path


# =====================================================
# EVALUATION
# =====================================================
def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predicted = torch.argmax(logits, dim=1)

            preds.extend(predicted.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    return acc, f1


# =====================================================
# MAIN
# =====================================================
def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE STARTED")
    logger.info("=" * 80)

    start_time = time.time()

    # -------------------------------------------------
    # Load Config
    # -------------------------------------------------
    with open("config/train.yaml") as f:
        cfg = yaml.safe_load(f)

    logger.info("[CONFIG] Loaded")

    raw_bucket = os.environ.get("RAW_BUCKET")
    registry_bucket = os.environ.get("REGISTRY_BUCKET")

    cloud_mode = raw_bucket and registry_bucket

    if cloud_mode:
        logger.info("[MODE] CLOUD")
        cfg["data"]["train_jsonl"] = download_snapshot_from_s3(
            raw_bucket, logger
        )
    else:
        logger.info("[MODE] LOCAL")

    # -------------------------------------------------
    # Device
    # -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[DEVICE] Using {device}")

    torch.set_num_threads(os.cpu_count())

    # -------------------------------------------------
    # Tokenizer
    # -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    logger.info("[MODEL] Tokenizer loaded")

    # -------------------------------------------------
    # Registry
    # -------------------------------------------------
    registry = ModelRegistry(registry_bucket, logger)

    existing_model_path = registry.load_existing_model_path(
        cfg["output"]["dir"]
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    if existing_model_path:
        logger.info(f"[MODEL] Fine-tuning {existing_model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            existing_model_path
        ).to(device)
    else:
        logger.info("[MODEL] Training from base")
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model"]["name"],
            num_labels=cfg["model"]["num_labels"],
        ).to(device)

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------
    dataset = NewsDataset(
        cfg["data"]["train_jsonl"],
        tokenizer,
        cfg["training"]["max_length"],
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )

    logger.info(f"[DATASET] {len(dataset)} samples")

    # -------------------------------------------------
    # Optimizer + Scheduler
    # -------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = cfg["training"]["epochs"] * len(loader)
    warmup_steps = int(cfg["training"]["warmup_ratio"] * total_steps)

    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------
    model.train()

    for epoch in range(cfg["training"]["epochs"]):
        logger.info(f"[TRAINING] Epoch {epoch+1}")

        total_loss = 0

        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if (step + 1) % 20 == 0:
                logger.info(
                    f"[TRAINING] Epoch {epoch+1} "
                    f"Step {step+1}/{len(loader)} "
                    f"Loss={loss.item():.4f}"
                )

        avg_loss = total_loss / len(loader)
        logger.info(
            f"[TRAINING] Epoch {epoch+1} Avg Loss={avg_loss:.4f}"
        )

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------
    logger.info("[EVAL] Starting evaluation")

    accuracy, f1 = evaluate_model(model, loader, device)

    logger.info(f"[EVAL] Accuracy={accuracy:.4f}")
    logger.info(f"[EVAL] F1={f1:.4f}")

    # -------------------------------------------------
    # Save Model Locally
    # -------------------------------------------------
    output_dir = cfg["output"]["dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"[SAVE] Model saved to {output_dir}")

    training_duration = time.time() - start_time
    logger.info(f"[STATS] Training duration: {training_duration:.2f}s")

    # -------------------------------------------------
    # Promotion Gating
    # -------------------------------------------------
    if accuracy >= PROMOTION_THRESHOLD:
        logger.info("[PROMOTION] Threshold met — promoting model")

        registry.save_new_version(
            output_dir,
            accuracy=accuracy,
            f1=f1,
            training_time=training_duration,
            dataset_size=len(dataset),
        )
    else:
        logger.warning(
            "[PROMOTION] Threshold NOT met — skipping promotion"
        )

    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()