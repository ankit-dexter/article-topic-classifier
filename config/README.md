# Article Topic Classifier – Architecture & System Design

## 1. Purpose of This Document

This document explains **how the system is built and why it is built this way**.
It focuses on **architecture, data flow, and engineering decisions**, not on step-by-step tutorials.

The goal is to show how a **production-style NLP system** is designed end-to-end.

---

## 2. Problem Definition

Given a news article consisting of a **title** and **body**, the system predicts one of four topics:

* **World** – international / geopolitical news
* **Sports** – sports events and commentary
* **Business** – markets, finance, companies
* **Sci/Tech** – science and technology

In addition to the prediction, the system must:

* return calibrated confidence scores
* expose probabilities for all classes
* decide whether the prediction can be auto-accepted or requires human review

---

## 3. High-Level System Architecture

```
Raw Data (CSV)
   ↓
Data Conversion Pipeline
(CSV → XML / JSONL / Parquet)
   ↓
Train / Validation / Test Split
   ↓
Model Training (DistilBERT)
   ↓
Evaluation on Unseen Data
   ↓
Inference Layer
   ├─ Single Prediction
   ├─ Batch Inference
   └─ FastAPI Service
```

This separation mirrors how ML systems are built in industry:

* ingestion is independent of modeling
* training is isolated from inference
* serving logic does not depend on training code

---

## 4. Project Structure (Logical View)

```
article-topic-classifier/
├── api/                     # Online serving (FastAPI)
│   └── app.py
├── artifacts/               # Model artifacts
│   └── distilbert/
├── config/                  # Training configuration
│   └── train.yaml
├── data/                    # Canonical datasets
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── scripts/                 # Offline jobs
│   ├── data_sanity_check.py
│   ├── split_dataset.py
│   ├── train_distilbert.py
│   ├── evaluate_distilbert.py
│   ├── predict.py
│   └── batch_predict.py
├── src/                     # Reusable ML components
│   └── dataset.py
└── README.md
```

---

## 5. Data Flow & Dataset Design

### Canonical Dataset Format

All ML code consumes **JSONL** files with a stable schema:

```json
{
  "title": "...",
  "body": "...",
  "topic": "Business"
}
```

Why JSONL:

* streamable
* easy to shard
* human-readable
* widely used in NLP pipelines

### Train / Validation / Test Split

Data is split explicitly into:

* `train.jsonl`
* `val.jsonl`
* `test.jsonl`

This prevents data leakage and enables honest evaluation.

---

## 6. Model Architecture

### Why DistilBERT

DistilBERT is a **Transformer encoder** distilled from BERT:

* ~40% smaller
* ~60% faster
* ~95% of BERT’s accuracy

It is a common **production default** for text classification tasks.

### Architecture Overview

```
Input Text (title + body)
   ↓
Tokenizer (WordPiece)
   ↓
Token IDs + Attention Mask
   ↓
Embedding Layer (768-dim)
   ↓
6 Transformer Encoder Layers
   ↓
[CLS] Token Representation
   ↓
Classification Head (768 → 4)
   ↓
Softmax Probabilities
```

The `[CLS]` token acts as a learned summary representation of the entire article.

---

## 7. Training Architecture

### Training Loop (Conceptual)

```
for epoch:
  for batch:
    forward pass
    compute loss (cross-entropy)
    backward pass
    optimizer step
    scheduler step
```

Key design choices:

* **Fine-tuning**, not training from scratch
* **Low learning rate** (2e-5) to preserve pretrained knowledge
* **Warmup + linear decay** for stable convergence
* **GPU acceleration** via PyTorch

---

## 8. Evaluation Architecture

Evaluation is performed on **unseen test data** and produces:

* accuracy
* precision / recall / F1 per class
* confusion matrix
* confidence statistics

This step answers:

> “Does the model generalize beyond training data?”

---

## 9. Inference Architecture

### Single Prediction

```
Article
   ↓
Tokenizer
   ↓
Model (eval mode, no gradients)
   ↓
Softmax
   ↓
Probabilities for all classes
```

### Batch Inference

```
JSONL File
   ↓
Batch DataLoader
   ↓
Model (loaded once)
   ↓
Predictions + confidence + decisions
   ↓
JSONL Output
```

Batch inference is preferred for:

* daily ingestion
* reprocessing
* analytics pipelines

---

## 10. Confidence-Aware Decision Layer

Predictions are routed using explicit rules:

* **auto_accept**

  * confidence ≥ 0.85
  * top-1 vs top-2 probability gap ≥ 0.20

* **needs_review**

  * confidence ≥ 0.60 but ambiguous

* **reject**

  * confidence < 0.60

This layer converts raw predictions into **safe, actionable decisions**.

---

## 11. API Architecture (FastAPI)

The model is exposed as a REST service using **FastAPI** with **ASGI lifespan** management.

### Design Principles

* model loaded once at startup
* GPU/CPU automatically selected
* stateless request handling
* JSON-in / JSON-out

### Endpoints

* `POST /predict` – single article
* `POST /batch_predict` – multiple articles

This architecture supports both real-time and batch clients.

---

## 12. Why PyTorch + Transformers

* PyTorch provides:

  * dynamic computation graphs
  * efficient GPU execution
  * fine-grained control over training

* Hugging Face Transformers provides:

  * pretrained models
  * standardized tokenization
  * reliable model loading & saving

Together, they form the **de facto industry stack for NLP**.

---

## 13. Production Readiness Considerations

This system demonstrates:

* clear separation of concerns
* reproducible training
* honest evaluation
* confidence-aware decisions
* batch and online inference paths

Common next steps in real systems:

* monitoring & drift detection
* Dockerization
* CI/CD integration

---

## 14. Summary

This architecture represents a **realistic, production-grade NLP system**, not a toy model.
It reflects how modern ML systems are designed, evaluated, and deployed in industry.
