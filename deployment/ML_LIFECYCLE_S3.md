# 🧠 End-to-End ML Lifecycle -- Complete System Flow

This document explains the full lifecycle of the Article Topic
Classification system, including how data flows through S3, ECS tasks,
model registry, and inference.

------------------------------------------------------------------------

# 🏗 High-Level Architecture

S3 (raw XML) ↓ Conversion Task (ECS RunTask) ↓ processed/train.jsonl
(active snapshot) + backups (7-day retention) ↓ Training Task (ECS
RunTask) ↓ Model Registry (S3 versioned models) ↓ Inference Service
(ECS + ALB)

The system is composed of three layers:

1.  Data Layer (Raw bucket)
2.  Model Layer (Registry bucket)
3.  Compute Layer (ECS tasks + service)

------------------------------------------------------------------------

# 🪣 1️⃣ S3 Strategy

## 🔹 Raw Data Bucket

topicclf-raw-data-ankit

Purpose: - Stores XML files - Stores processed training snapshot - Keeps
short-term dataset backups

Structure:

raw/ processed/ train.jsonl ← Active snapshot (always overwritten)
backups/ train_YYYYMMDD_HHMMSS.jsonl ← Auto-expire after 7 days

Strategy: - train.jsonl is always overwritten. - Backups are retained
for 7 days using S3 lifecycle rules. - Training always reads exactly one
file: processed/train.jsonl. - This ensures deterministic behavior and
no file scanning logic.

------------------------------------------------------------------------

## 🔹 Model Registry Bucket

topicclf-ml-registry-ankit

Purpose: - Stores immutable model versions - Controls production
promotion - Tracks metadata and metrics

Structure:

models/ v20260221_185512/ v20260222_091200/

pointers/ latest.json metadata.json

latest.json:

{ "current_version": "v20260221_185512" }

metadata.json (example):

{ "last_trained_at": "...", "last_full_train_at": "...",
"current_version": "v20260221_185512", "total_versions": 3,
"training_mode": "incremental", "metrics": { "accuracy": 0.91, "f1":
0.89 }, "training_time_seconds": 384.21, "dataset_size": 6080 }

Strategy: - Model folders are immutable. - Each training run creates a
new version folder. - Inference never scans models. - Deployment is
controlled only via latest.json. - Rollback = update latest.json.

------------------------------------------------------------------------

# ⚙️ 2️⃣ ECS Compute Layer

## 🔹 Conversion Task (ECS RunTask)

Purpose: - Read XML files from raw/ - Create JSONL snapshot - Upload: -
processed/train.jsonl -
processed/backups/train\_`<timestamp>`{=html}.jsonl

Which XML files are processed?

All .xml files under raw/ WHERE LastModified \>
metadata\["last_trained_at"\]

If last_trained_at is null: → All XML files are included.

If not null: → Only newly uploaded XML files are processed.

Key Characteristics: - Stateless - Short-lived - Zero idle cost - No
load balancer - Pure batch processing

------------------------------------------------------------------------

## 🔹 Training Task (ECS RunTask)

Purpose: - Download processed/train.jsonl - Load previous model (if
exists) - Fine-tune incrementally - Evaluate performance - Promote model
if metrics pass threshold - Upload new model version - Update metadata
and pointer

Training Behavior:

1.  Read pointers/latest.json
2.  If current_version exists:
    -   Download previous model
    -   Fine-tune
3.  If null:
    -   Train from base DistilBERT

After training: - Compute Accuracy + Macro F1 - Compare with promotion
threshold (e.g. 0.80)

If threshold met: - Upload new version - Update metadata.json - Update
latest.json

If threshold not met: - Model saved locally - Not promoted - latest.json
unchanged

Key Characteristics: - Stateless - Zero idle cost - Memory-optimized -
Registry-driven promotion - Controlled deployment

------------------------------------------------------------------------

## 🔹 Inference Service (ECS Service + ALB)

Purpose: - Serve predictions via FastAPI - Load model on startup - Use
latest.json pointer

Startup Flow:

1.  Read pointers/latest.json
2.  Get current_version
3.  Download models/`<version>`{=html}/
4.  Load HuggingFace model
5.  Serve predictions

Key Characteristics: - Always running - Behind ALB - Stateless - Model
loaded once - No Docker rebuild needed for new models

------------------------------------------------------------------------

# 🔁 Complete Lifecycle Flow

1️⃣ XML uploaded to raw/ 2️⃣ Conversion RunTask triggered 3️⃣ Snapshot
created → processed/train.jsonl 4️⃣ Training RunTask triggered 5️⃣ Model
fine-tuned + evaluated 6️⃣ If approved → new version uploaded 7️⃣
latest.json updated 8️⃣ Inference container loads new version

------------------------------------------------------------------------

# 🔒 Why This Architecture Is Production-Grade

Separation of Concerns: - Conversion handles data - Training handles
model lifecycle - Inference handles serving

Immutable Model Artifacts: models/v1/ models/v2/ models/v3/

Pointer-Based Promotion: - Change latest.json - Inference reads
pointer - Deployment is controlled

Zero Idle Cost for Heavy Compute: - Conversion runs only when
triggered - Training runs only when triggered - Inference is lightweight

Deterministic Behavior: - Training always reads processed/train.jsonl -
Inference always reads pointers/latest.json

------------------------------------------------------------------------

# 🏁 Summary

This system implements:

-   Incremental data ingestion
-   Incremental model fine-tuning
-   Evaluation-based promotion gating
-   Versioned model registry
-   Snapshot lifecycle management
-   ECS Fargate deployment
-   Cost-optimized compute
-   Controlled production promotion

It represents a complete, registry-driven MLOps lifecycle suitable for
production demonstration and scalable evolution.
