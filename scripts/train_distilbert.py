"""
Production-ready training script for DistilBERT topic classifier.

Supports:
- Local training
- Cloud training (S3 registry)
- Incremental fine-tuning
- Safe S3 model promotion
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
import torch
import boto3
import logging
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)

from src.dataset import NewsDataset
from src.utils import setup_logging
from src.registry import ModelRegistry


# =====================================================
# DATASET DOWNLOAD (CLOUD MODE)
# =====================================================
def download_latest_jsonl_from_s3(raw_bucket, logger):
    s3 = boto3.client("s3")

    logger.info("[DATASET][CLOUD] Searching processed/train_*.jsonl")

    response = s3.list_objects_v2(
        Bucket=raw_bucket,
        Prefix="processed/",
    )

    jsonl_files = [
        obj for obj in response.get("Contents", [])
        if obj["Key"].startswith("processed/train_")
        and obj["Key"].endswith(".jsonl")
    ]

    if not jsonl_files:
        raise RuntimeError("[DATASET][CLOUD] No processed train_*.jsonl found")

    latest_file = max(jsonl_files, key=lambda x: x["LastModified"])

    local_path = "/tmp/train.jsonl"

    s3.download_file(raw_bucket, latest_file["Key"], local_path)

    logger.info(
        f"[DATASET][CLOUD] Using dataset: {latest_file['Key']}"
    )

    return local_path


# =====================================================
# MAIN
# =====================================================
def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE STARTED")
    logger.info("=" * 80)

    # -------------------------------------------------
    # Load Config
    # -------------------------------------------------
    with open("config/train.yaml") as f:
        cfg = yaml.safe_load(f)

    logger.info("[CONFIG] Configuration loaded")

    raw_bucket = os.environ.get("RAW_BUCKET")
    registry_bucket = os.environ.get("REGISTRY_BUCKET")

    cloud_mode = raw_bucket and registry_bucket

    if cloud_mode:
        logger.info("[MODE] Running in CLOUD mode")
        cfg["data"]["train_jsonl"] = download_latest_jsonl_from_s3(
            raw_bucket,
            logger,
        )
    else:
        logger.info("[MODE] Running in LOCAL mode")

    # -------------------------------------------------
    # Device
    # -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[DEVICE] Using {device}")

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
    # Model Loading
    # -------------------------------------------------
    if existing_model_path:
        logger.info(f"[MODEL] Fine-tuning existing model: {existing_model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            existing_model_path
        ).to(device)
    else:
        logger.info("[MODEL] Training from base model")
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model"]["name"],
            num_labels=cfg["model"]["num_labels"],
        ).to(device)

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------
    logger.info(f"[DATASET] Loading {cfg['data']['train_jsonl']}")

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

    logger.info(f"[DATASET] {len(dataset)} samples | {len(loader)} batches")

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

    logger.info(
        f"[TRAINING] Steps={total_steps} | Warmup={warmup_steps}"
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

            if (step + 1) % 10 == 0:
                logger.info(
                    f"[TRAINING] Epoch {epoch+1} "
                    f"Step {step+1}/{len(loader)} "
                    f"Loss={loss.item():.4f}"
                )

        avg_loss = total_loss / len(loader)
        logger.info(
            f"[TRAINING] Epoch {epoch+1} completed | Avg Loss={avg_loss:.4f}"
        )

    # -------------------------------------------------
    # Save Model (Local Directory)
    # -------------------------------------------------
    output_dir = cfg["output"]["dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"[SAVE] Model saved to {output_dir}")

    # -------------------------------------------------
    # Registry Upload (Cloud Only)
    # -------------------------------------------------
    registry.save_new_version(output_dir)

    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()