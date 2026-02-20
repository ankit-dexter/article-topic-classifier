# 📰 Article Topic Classification -- Production-Grade NLP System

[![Python
3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0%2B-brightgreen.svg)](https://huggingface.co/transformers/)
[![Deployed on AWS
EC2](https://img.shields.io/badge/Deployment-AWS%20EC2-orange.svg)]()

A production-oriented **end-to-end NLP system** that classifies news
articles into four topics using a fine-tuned **DistilBERT** model.

This project demonstrates real-world ML engineering practices: - Clean
data pipelines\
- Proper train/validation/test isolation\
- Confidence-aware decision routing\
- Calibration & selective classification\
- Batch and real-time inference\
- Dockerized deployment\
- Secure AWS EC2 hosting (Free Tier safe)

------------------------------------------------------------------------

## 🎯 Problem Statement

Given a news article (title + body), predict its primary topic:

-   World
-   Sports
-   Business
-   Sci/Tech

The system supports:

-   Incremental fine-tuning
-   Evaluation-based promotion gating
-   Versioned model registry
-   Zero-idle-cost training
-   ECS Fargate inference service
-   CloudWatch observability

------------------------------------------------------------------------

# 🧠 Model Choice

**DistilBERT (distilbert-base-uncased)**

Why DistilBERT?

-   Retains \~95% of BERT performance\
-   \~40% smaller\
-   \~60% faster\
-   Production-friendly latency\
-   Widely adopted in real NLP systems

Framework: PyTorch + Hugging Face Transformers

Training Hardware: Local GPU (RTX 3060)

------------------------------------------------------------------------

## 🏗 Architecture Overview

XML Ingestion (S3 raw/) ↓ Conversion (ECS RunTask) ↓
processed/train.jsonl (active snapshot + 7-day backups) ↓ Training (ECS
RunTask with evaluation gating) ↓ Model Registry (S3 versioned models) ↓
Inference (ECS Fargate Service behind ALB)

------------------------------------------------------------------------

# 🏗 System Architecture

Raw Data (CSV)\
↓\
Data Conversion Pipeline (CSV → XML / JSONL / Parquet)\
↓\
Train / Validation / Test Split\
↓\
Model Training (DistilBERT Fine-Tuning)\
↓\
Evaluation on Unseen Test Set\
↓\
Inference Layer\
├─ Single Prediction\
├─ Batch Inference\
└─ FastAPI REST Service\
↓\
Docker Container\
↓\
AWS EC2 Deployment

------------------------------------------------------------------------

# 📁 Project Structure

article-topic-classifier/\
├── api/ \# FastAPI service\
├── artifacts/ \# Saved model + tokenizer\
├── config/ \# Training config\
├── data/ \# train / val / test JSONL\
├── scripts/ \# Training & evaluation scripts\
├── src/ \# Dataset class\
├── requirements-dev.txt\
├── requirements-prod.txt\
├── Dockerfile\
├── DEPLOYMENT.md\
└── README.md

------------------------------------------------------------------------

# 🔍 Confidence-Aware Decision Logic

The system does not blindly trust predictions.

Decision rules:

auto_accept\
- confidence ≥ 0.85\
- top1 - top2 probability gap ≥ 0.20

needs_review\
- confidence ≥ 0.60 but ambiguous

reject\
- confidence \< 0.60

These thresholds were validated using a **coverage vs accuracy curve**
on the validation set.

Example result:

Threshold 0.86 →\
- \~85% coverage\
- \~95% accuracy on auto-accepted predictions

This mirrors real production risk-management systems.

------------------------------------------------------------------------

# 📊 Evaluation

Evaluation is performed strictly on unseen test data.

Metrics reported: - Accuracy (\~90%) - Precision / Recall / F1 -
Confusion matrix - Confidence statistics - Expected Calibration Error
(ECE) - Coverage vs Accuracy curve

This transforms the system from a simple classifier into a **selective
classification system**.

------------------------------------------------------------------------

# 🧪 Inference

Single prediction:

python -m scripts.predict

Batch prediction:

python -m scripts.batch_predict

Outputs include: - prediction - confidence - all probabilities -
decision label

------------------------------------------------------------------------

# 🌐 API Service

Run locally:

uvicorn api.app:app --reload

Interactive docs:

http://54.83.125.240:8000/docs#/

Endpoints:

POST /predict\
POST /batch_predict

The model loads once at startup and serves inference requests
efficiently.

------------------------------------------------------------------------

# 🐳 Dockerized Production Build

Production dependencies are separated from development dependencies.

requirements-prod.txt contains only inference essentials.

Dockerfile installs CPU-only PyTorch and minimal runtime packages.

Build:

docker build -t article-topic-classifier .

Run:

docker run -d -p 8000:8000 article-topic-classifier

------------------------------------------------------------------------

## 🪣 S3 Buckets

### Raw Data Bucket

topicclf-raw-data-ankit

processed/ - train.jsonl (active snapshot) - backups/ (auto-expire after
7 days)

### Model Registry Bucket

topicclf-ml-registry-ankit

models/ - vYYYYMMDD_HHMMSS/

pointers/ - latest.json - metadata.json

------------------------------------------------------------------------

## 🔁 Incremental Training Flow

1.  Read latest.json
2.  If version exists → download & fine-tune
3.  Else → train from base model
4.  Evaluate (Accuracy + Macro F1)
5.  Promote only if threshold met
6.  Update metadata.json and latest.json

Promotion threshold example:

PROMOTION_THRESHOLD = 0.80

------------------------------------------------------------------------

## 📊 Metadata Tracking

metadata.json stores:

-   last_trained_at
-   current_version
-   total_versions
-   training_mode
-   metrics (accuracy, f1)
-   training_time_seconds
-   dataset_size

------------------------------------------------------------------------

## 🚀 Inference Service

-   ECS Fargate Service
-   Application Load Balancer
-   Swagger endpoint: /docs
-   Loads model using latest.json pointer

------------------------------------------------------------------------

## 💰 Cost Strategy

-   ECS RunTask for training (no idle cost)
-   ECS RunTask for conversion
-   ECS Service for inference
-   No SageMaker
-   No Lambda
-   No NAT Gateway

------------------------------------------------------------------------

## 🔄 Rollback Capability

Rollback is instant by updating:

pointers/latest.json

Inference loads previous version on restart.

------------------------------------------------------------------------

## 📁 Project Structure

article-topic-classifier/ ├── api/ ├── artifacts/ ├── config/ ├──
scripts/ ├── src/ │ ├── dataset.py │ ├── registry.py │ └── utils.py ├──
Dockerfile ├── Dockerfile.train └── README.md

------------------------------------------------------------------------


# 🛠 Real-World Issues Resolved

-   SSH timeouts due to dynamic IP changes\
-   Disk overflow from CUDA dependencies\
-   Pip index override causing install failures\
-   Windows SSH private key permission errors\
-   Accidental private key upload to server

Each issue was diagnosed using Docker logs, AWS security rules, and
system-level debugging.

------------------------------------------------------------------------

# 🔐 Security Practices

-   SSH locked to current IP\
-   No password authentication\
-   Private key removed from EC2\
-   Billing budget guard enabled\
-   Minimal production dependencies\
-   No unnecessary open ports

------------------------------------------------------------------------

# 🔮 Future Improvements

-   HTTPS via Let's Encrypt\
-   Elastic IP\
-   Reverse proxy (Nginx)\
-   ECS Fargate migration\
-   CI/CD pipeline\
-   Monitoring & drift detection

------------------------------------------------------------------------

# 📄 License

MIT License

------------------------------------------------------------------------

# Summary

This repository demonstrates how to design, validate, deploy, and secure
a production-aligned NLP classification system.

It reflects modern ML engineering practices across:

-   Data pipeline design\
-   Model training\
-   Calibration & risk management\
-   API serving\
-   Containerization\
-   Cloud deployment
