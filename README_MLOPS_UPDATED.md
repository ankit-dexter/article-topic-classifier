# 📰 Article Topic Classification --- End-to-End MLOps System (AWS Fargate)

A production-grade end-to-end NLP + MLOps system that classifies news
articles into four topics using an incrementally fine-tuned DistilBERT
model.

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

## 🏗 Architecture Overview

XML Ingestion (S3 raw/) ↓ Conversion (ECS RunTask) ↓
processed/train.jsonl (active snapshot + 7-day backups) ↓ Training (ECS
RunTask with evaluation gating) ↓ Model Registry (S3 versioned models) ↓
Inference (ECS Fargate Service behind ALB)

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

## Summary

This repository demonstrates a fully managed MLOps lifecycle on AWS
Fargate, including incremental training, evaluation gating, versioned
registry, and production-grade deployment.
