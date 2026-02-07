# Article Topic Classifier - Complete Architecture Guide

## 📋 Project Overview

This project implements a **fine-tuned DistilBERT model** that classifies news articles into 4 topics:
- 🌍 **World** - International news
- 🏆 **Sports** - Sports articles
- 💼 **Business** - Business and finance
- 🔬 **Sci/Tech** - Science and technology

The model uses **transfer learning**: starting from a pre-trained DistilBERT model and fine-tuning it on our specific task.

---

## 🏗️ Project Structure

```
article-topic-classifier/
├── config/
│   └── train.yaml              # Hyperparameters and settings
├── data/
│   ├── README.md
│   └── part-0001.jsonl         # Training articles (JSON Lines format)
├── scripts/
│   ├── data_sanity_check.py    # Validate data before training
│   └── train_distilbert.py     # Main training script
├── src/
│   ├── dataset.py              # Custom dataset class
│   ├── metrics.py              # Evaluation metrics
│   └── utils.py                # Logging configuration
├── artifacts/
│   └── distilbert/             # Saved model and tokenizer
├── notebooks/
│   └── [analysis and visualization notebooks]
├── logs/
│   └── training_*.log          # Training logs
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🔄 How Everything Works Together

### 1️⃣ **Configuration (config/train.yaml)**

```yaml
model:
  name: distilbert-base-uncased  # Pre-trained model from Hugging Face
  num_labels: 4                  # 4 topic classes

training:
  epochs: 3                      # Train for 3 full passes through data
  batch_size: 16                 # Process 16 articles at a time
  learning_rate: 0.00002         # How fast to update weights
  max_length: 256                # Truncate/pad articles to 256 tokens
  weight_decay: 0.01             # L2 regularization (prevent overfitting)
  warmup_ratio: 0.1              # Warmup for first 10% of training
```

**Why these values?**
- **DistilBERT**: 40% smaller and 60% faster than BERT while maintaining 97% of performance
- **Learning rate (0.00002)**: Very small because we're fine-tuning (weights already good)
- **Batch size (16)**: Good balance between speed and memory usage
- **Max length (256)**: Most articles fit comfortably; more tokens = more computation

---

## 📊 Data Pipeline

### Step 1: Raw Data → Dataset Class

**Input:** `data/part-0001.jsonl` (articles in JSON Lines format)

```json
{"title": "Breaking News", "body": "Full article text...", "topic": "World"}
{"title": "Sports Update", "body": "Game highlights...", "topic": "Sports"}
```

### Step 2: Tokenization

**Process:** `NewsDataset` class converts text → token IDs

```python
Text: "Breaking News. Full article"
         ↓ Tokenizer
Tokens: [101, 8149, 1038, 1012, 2440, 3720, 102, 0, 0...]
         ↑                                           ↑
       [CLS] token                                 [PAD] tokens
```

**Key Components:**
- **[CLS]** token: Special token at start (used for classification)
- **[SEP]** token: Separates sentences
- **Subword tokens**: "Breaking" → "Breaking", but "tokenization" → "token", "##ization"
- **[PAD]** tokens: Fill sequences shorter than max_length

### Step 3: Attention Mask

Tells model which tokens are real vs padding:
```
Tokens:          [101, 8149, 1012, 2440, 0, 0]
Attention Mask:  [1,   1,    1,    1,    0, 0]
                 ↑ real tokens ↑  ↑ padding ↑
```

---

## 🧠 Model Architecture

### DistilBERT Structure

```
Input Text
    ↓
Tokenizer → Token IDs [101, 8149, 1012, ...]
    ↓
Embedding Layer (converts IDs → 768-dimensional vectors)
    ↓
6 Transformer Layers (each with 12 attention heads)
    ↓
[CLS] Token Output (768-dimensional vector representing whole text)
    ↓
Classification Head (768 → 4 classes)
    ↓
Output: Probabilities for each topic
    ↓
Softmax: [0.05, 0.02, 0.88, 0.05]
         World Sports Business Sci/Tech
```

**Why [CLS] token?** It's a learned representation of the entire text (summary). Perfect for classification!

---

## 🔧 Training Process

### Training Loop Overview

```
For each Epoch:
    For each Batch (16 articles):
        1. Tokenize articles
        2. Forward pass → Get predictions
        3. Calculate loss (how wrong we are)
        4. Backward pass → Calculate gradients
        5. Update weights using optimizer
        6. Update learning rate using scheduler
        7. Log progress
```

### Detailed Step-by-Step

#### Step 1: Forward Pass
```python
output = model(
    input_ids=token_ids,           # Tokenized text
    attention_mask=attention_mask,  # Tell model what's real vs padding
    labels=topic_ids                # True labels (0-3)
)
loss = output.loss  # Cross-entropy loss
```

Cross-entropy loss measures how different predicted probabilities are from true labels:
- Perfect prediction [0, 0, 1, 0] → loss = 0
- Bad prediction [0.25, 0.25, 0.25, 0.25] → loss = high

#### Step 2: Backward Pass (Gradient Calculation)
```python
loss.backward()  # Compute ∂loss/∂weights for all layers
```

This tells us: "To reduce loss, change weight W by gradient G"

#### Step 3: Parameter Update (Gradient Descent)
```python
optimizer.step()  # weights = weights - (learning_rate × gradient)
```

Example: If gradient = 0.001, learning_rate = 0.00002
- New weight = old weight - (0.00002 × 0.001) = old weight - 0.00000002

#### Step 4: Learning Rate Scheduling
```python
scheduler.step()  # Adjust learning rate
```

**Warmup Phase (first 10%):**
- Learning rate: 0 → 0.00002 gradually
- Why? Prevent instability with pre-trained weights

**Decay Phase (remaining 90%):**
- Learning rate: 0.00002 → 0 gradually
- Why? Fine-grained updates as we approach convergence

---

## 📝 Key Training Concepts

### 1. **Fine-tuning vs Training from Scratch**

| Aspect | From Scratch | Fine-tuning |
|--------|-------------|------------|
| Learning rate | 0.001-0.01 (large) | 0.00002 (tiny) |
| Epochs | 10-100 | 2-5 |
| Data needed | 100K+ examples | 500+ examples |
| Time | Days/weeks | Minutes/hours |
| Performance | Often worse | Usually better |

We use **fine-tuning** because:
- DistilBERT already learned language patterns
- We only need to teach it our specific task
- Small learning rate prevents "forgetting" pre-trained knowledge

### 2. **Batch Processing**

Why batch 16 articles instead of 1?
- **Speed**: GPU can process 16 in parallel
- **Stability**: Gradients from 16 samples more reliable than 1
- **Memory**: 16 × 256 tokens ≈ 4KB per batch (GPU has 24GB)

### 3. **Loss Function**

We use **Cross-Entropy Loss** (standard for classification):

```
Loss = -Σ(true_label × log(predicted_probability))

Example:
- True label: Business [0, 0, 1, 0]
- Prediction: [0.05, 0.02, 0.88, 0.05]
- Loss = -log(0.88) ≈ 0.128 (lower is better)
```

---

## 📦 Dependencies & Environment

### Required Libraries

```
torch                    # Deep learning framework
transformers>=4.0       # Pre-trained models (BERT, DistilBERT, etc.)
pandas                  # Data manipulation
PyYAML                  # Load config files
scikit-learn            # Evaluation metrics
```

### Why Each?
- **torch**: PyTorch framework for neural networks
- **transformers**: Hugging Face library with pre-trained models
- **pandas**: Handle data in DataFrames
- **PyYAML**: Parse YAML config files (human-readable settings)
- **scikit-learn**: Precision, recall, F1-score calculations

---

## 📊 Monitoring Training Progress

### Log Files

Training creates logs in two places:

1. **File Logs** (`logs/training_20260208_143022.log`)
   - Detailed: includes timestamps, module names, debug info
   - Used for debugging issues

2. **Console Output**
   - Summary only: key milestones
   - Used for real-time progress monitoring

### Example Log Output

```
INFO - ================================================================================
INFO - Starting training pipeline
INFO - ================================================================================
INFO - Loading configuration from config/train.yaml
INFO - Config loaded successfully
INFO - Using device: cuda
INFO - Loading tokenizer: distilbert-base-uncased
INFO - Tokenizer loaded successfully
INFO - Loading model: distilbert-base-uncased with 4 labels
INFO - Model moved to cuda
INFO - Loading dataset from data/part-0001.jsonl
INFO - Dataset loaded: 500 samples
INFO - Creating DataLoader with batch_size=16
INFO - DataLoader created: 32 batches

INFO - Starting training for 3 epochs
INFO - 
INFO - Epoch 1/3
INFO -   Batch 6/31 | Loss: 1.3245
INFO -   Batch 12/31 | Loss: 1.1234
INFO -   Batch 19/31 | Loss: 0.9834
INFO -   Batch 25/31 | Loss: 0.8567
INFO -   Batch 31/31 | Loss: 0.7834
INFO - Epoch 1 completed | Avg Loss: 0.9456
...
INFO - Training completed! Saving model to artifacts/distilbert
INFO - ✅ Model and tokenizer saved successfully
```

---

## 💾 Output Files

### After Training

```
artifacts/distilbert/
├── config.json              # Model architecture configuration
├── pytorch_model.bin        # Model weights (268 MB)
├── tokenizer.json           # Tokenizer vocabulary and rules
├── tokenizer_config.json    # Tokenizer settings
└── vocab.txt                # List of all tokens
```

These files are all you need to make predictions on new articles!

---

## 🔮 How to Use the Trained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load saved model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("artifacts/distilbert")
model = AutoModelForSequenceClassification.from_pretrained("artifacts/distilbert")

# Predict topic for new article
text = "Breaking news from the tech industry..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
outputs = model(**inputs)

# Get predicted class
predicted_class = outputs.logits.argmax(dim=-1).item()
topics = ["World", "Sports", "Business", "Sci/Tech"]
print(f"Topic: {topics[predicted_class]}")
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)
- **Cause**: Batch size too large for GPU
- **Solution**: Reduce `batch_size` in config (8, 4)

### Issue: Loss not decreasing
- **Cause**: Learning rate wrong
- **Solution**: Try 0.00001 or 0.00005

### Issue: Model predicts same class for everything
- **Cause**: Insufficient training data or epochs
- **Solution**: Increase `epochs` to 5-10, add more data

---

## 📚 Learning Resources

### Understanding Transformers
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Hugging Face Course](https://huggingface.co/course) - Interactive learning

### Transfer Learning
- [Fine-tuning BERT](https://huggingface.co/docs/transformers/training) - Official guide

### PyTorch
- [PyTorch Documentation](https://pytorch.org/docs) - Complete reference

---

## ✅ Next Steps

1. **Run Training**: `python -m scripts.train_distilbert`
2. **Monitor Progress**: Check `logs/` directory for detailed logs
3. **Evaluate**: Create evaluation script using `src/metrics.py`
4. **Deploy**: Use saved model for inference on new articles
5. **Improve**: Experiment with hyperparameters, more data, different architectures

---

## 🎯 Summary

This project demonstrates:
- ✅ Transfer learning with pre-trained models
- ✅ Fine-tuning for custom tasks
- ✅ Proper data handling and preprocessing
- ✅ Production-quality logging and monitoring
- ✅ Clean, commented, readable code

The architecture is modular and extensible - you can easily swap DistilBERT for BERT, RoBERTa, or other models from Hugging Face!
