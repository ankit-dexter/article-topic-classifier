# 🧪 Scripts Usage Guide

## Article Topic Classification -- Command Reference

This document describes how to run each script in the `scripts/`
directory.

All commands should be executed from the project root directory.

------------------------------------------------------------------------

# 📁 Available Scripts

    scripts/
    ├── data_sanity_check.py
    ├── split_dataset.py
    ├── train_distilbert.py
    ├── evaluate_distilbert.py
    ├── calibration_analysis.py
    ├── predict.py
    └── batch_predict.py

------------------------------------------------------------------------

# 🔍 1️⃣ Data Sanity Check

``` bash
python -m scripts.data_sanity_check
```

Verifies:

-   Total sample count\
-   Label distribution\
-   Token length statistics\
-   Dataset integrity

✔ Ensures clean and balanced input data before training.

------------------------------------------------------------------------

# ✂️ 2️⃣ Train / Validation / Test Split

``` bash
python -m scripts.split_dataset
```

Creates:

-   `train.jsonl`\
-   `val.jsonl`\
-   `test.jsonl`

✔ Maintains label distribution\
✔ Prevents data leakage

------------------------------------------------------------------------

# 🧠 3️⃣ Model Training (DistilBERT Fine-Tuning)

``` bash
python -m scripts.train_distilbert
```

What happens:

-   Loads training and validation datasets\
-   Fine-tunes DistilBERT\
-   Uses GPU automatically if available\
-   Applies AdamW optimizer + linear scheduler\
-   Saves model to `artifacts/distilbert/`\
-   Logs training progress

✔ Typical convergence: loss \~1.2 → \~0.14

------------------------------------------------------------------------

# 📊 4️⃣ Model Evaluation (Unseen Test Set)

``` bash
python -m scripts.evaluate_distilbert
```

Reports:

-   Accuracy (\~90%)\
-   Precision / Recall / F1-score\
-   Confusion matrix\
-   Confidence statistics\
-   Expected Calibration Error (ECE)\
-   Coverage vs Accuracy curve

✔ Strict evaluation on `test.jsonl` only\
✔ Validates confidence-aware routing thresholds

------------------------------------------------------------------------

# 📈 5️⃣ Calibration & Selective Classification Analysis

``` bash
python -m scripts.calibration_analysis
```

Computes:

-   Expected Calibration Error (ECE)\
-   Reliability diagnostics\
-   Coverage vs Accuracy tradeoff

✔ Used to validate production confidence threshold (\~0.85)\
✔ Enables risk-aware deployment

------------------------------------------------------------------------

# 🧪 6️⃣ Single Article Inference

``` bash
python -m scripts.predict
```

Returns structured output:

``` json
{
  "prediction": "World",
  "confidence": 0.97,
  "all_probabilities": {
    "World": 0.97,
    "Sports": 0.01,
    "Business": 0.01,
    "Sci/Tech": 0.01
  },
  "decision": "auto_accept",
  "top2_label": "Business",
  "top2_gap": 0.94
}
```

✔ Applies confidence-aware decision routing

------------------------------------------------------------------------

# 📦 7️⃣ Batch Inference

``` bash
python -m scripts.batch_predict
```

-   Reads input JSONL file\
-   Loads model once\
-   Processes articles using DataLoader batching\
-   Writes output JSONL with predictions

✔ Efficient for large-scale processing\
✔ Mimics real-world batch ML pipelines

------------------------------------------------------------------------

# 🏁 Summary

These scripts collectively demonstrate:

-   Clean ML lifecycle management\
-   Proper dataset handling\
-   Rigorous evaluation\
-   Calibration & selective classification\
-   Real-time and batch inference patterns

They form the backbone of the production-ready NLP pipeline.
