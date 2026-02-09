# 📰 Article Topic Classification – Production-Style NLP System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0%2B-brightgreen.svg)](https://huggingface.co/transformers/)

A production-oriented **end-to-end NLP system** that classifies news articles into four topics using a fine‑tuned **DistilBERT** model.
This project goes beyond model training and demonstrates **real industry ML practices**: clean data pipelines, proper evaluation, confidence‑aware decision logic, batch processing, and a deployable FastAPI service.

---

## 🎯 Problem Statement

Given a news article (title + body), automatically predict its primary topic:

* **World**
* **Sports**
* **Business**
* **Sci/Tech**

The system must also:

* Return a confidence score
* Expose probabilities for all classes
* Decide whether a prediction can be auto‑accepted or needs human review

---

## ✨ Key Features

* **Modern Transformer Model**: Fine‑tuned DistilBERT (fast, lightweight, production‑friendly)
* **End‑to‑End ML Lifecycle**: Data → Training → Evaluation → Inference → API
* **Confidence‑Aware Decisions**: Auto‑accept / Needs‑review / Reject logic
* **Batch Inference Support**: Process large article sets efficiently
* **FastAPI Service**: Deployable REST API with modern ASGI lifespan handling
* **GPU‑Accelerated Training**: PyTorch + CUDA

---

## 🧠 Why DistilBERT?

DistilBERT retains ~95% of BERT’s accuracy while being ~40% smaller and ~60% faster.

**Chosen because:**

* Strong pretrained language understanding
* Lower latency and cost than full BERT
* Widely used in real production NLP systems
* Excellent fit for topic classification

---

## 🏗️ System Architecture

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

---

## 📁 Project Structure

```
article-topic-classifier/
├── api/
│   └── app.py              # FastAPI service
├── artifacts/
│   └── distilbert/         # Trained model & tokenizer
├── config/
│   └── train.yaml          # Training configuration
├── data/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── scripts/
│   ├── data_sanity_check.py
│   ├── split_dataset.py
│   ├── train_distilbert.py
│   ├── evaluate_distilbert.py
│   ├── predict.py          # Single inference
│   └── batch_predict.py    # Batch inference
├── src/
│   └── dataset.py          # PyTorch Dataset
├── logs/
│   └── training_*.log
├── requirements.txt
└── README.md
```

---

## 🔍 Confidence‑Aware Decision Logic

Predictions are not blindly trusted. The system applies routing rules:

* **auto_accept**

  * confidence ≥ 0.85
  * top‑1 vs top‑2 probability gap ≥ 0.20

* **needs_review**

  * confidence ≥ 0.60 but ambiguous

* **reject**

  * confidence < 0.60

This mirrors how real editorial and enterprise ML systems manage risk.

---

## 🚀 Training

```bash
python -m scripts.train_distilbert
```

* Uses GPU if available
* Trains DistilBERT for topic classification
* Saves model and tokenizer to `artifacts/distilbert/`

---

## 📊 Evaluation

```bash
python -m scripts.evaluate_distilbert
```

Evaluation is performed on **unseen test data** and reports:

* Accuracy
* Precision / Recall / F1 per class
* Confusion matrix
* Confidence statistics

**Example result:** ~90% accuracy with balanced class performance.

---

## 🧪 Inference

### Single Article

```bash
python -m scripts.predict
```

Returns:

* predicted topic
* confidence
* probabilities for all classes
* decision (auto_accept / needs_review / reject)

### Batch Inference

```bash
python -m scripts.batch_predict
```

Processes an entire JSONL file and produces a JSONL output with predictions and decisions for each article.

---

## 🌐 API Service (FastAPI)

Run the API:

```bash
uvicorn api.app:app --reload
```

Endpoints:

* `POST /predict` — single article
* `POST /batch_predict` — multiple articles

Interactive docs:

```
http://127.0.0.1:8000/docs
```

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* DistilBERT
* FastAPI
* scikit‑learn
* JSONL / Parquet

---

## ✅ What This Project Demonstrates

* Clean separation of data, model, and serving layers
* Proper train/validation/test isolation
* Confidence‑aware ML decision making
* Batch vs real‑time inference patterns
* Production‑ready API design

---

## 🔮 Future Extensions (V2 Ideas)

* Vector embeddings for semantic search
* LLM (LLaMA) integration for summarization
* Model monitoring & drift detection
* Dockerized deployment

---

## 📄 License

MIT License

---

## Summary

This repository demonstrates how to build a **realistic, production‑ready NLP system**, not just a model. It reflects how modern ML is designed, evaluated, and served in industry.
