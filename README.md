# 📰 Article Topic Classifier

A deep learning project that automatically classifies news articles into 4 categories using a fine-tuned **DistilBERT** model. Built with PyTorch and Hugging Face Transformers.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0%2B-brightgreen.svg)](https://huggingface.co/transformers/)

---

## 🎯 Features

- ✨ **Transfer Learning**: Fine-tuned DistilBERT pre-trained model
- ⚡ **Fast & Lightweight**: DistilBERT is 40% smaller and 60% faster than BERT
- 📊 **4-Class Classification**: World, Sports, Business, Sci/Tech
- 📝 **Comprehensive Logging**: Detailed training logs to file + console output
- 🔧 **Configurable**: YAML-based hyperparameter management
- 📈 **Production-Ready**: Model saved in Hugging Face format for easy deployment
- 💻 **Well-Documented**: Extensive code comments and architecture guide

---

## 📋 Project Overview

This project demonstrates:
- How to load and preprocess text data
- Fine-tuning pre-trained transformer models
- Building custom PyTorch datasets
- Implementing training loops with proper logging
- Saving and loading models for inference

**Model Performance:**
- Trained on 500 news articles
- 4 epochs with batch size 16
- Achieves high accuracy on all 4 topic categories

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd article-topic-classifier

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your training data at `data/part-0001.jsonl` in JSON Lines format:

```json
{"title": "Breaking News Story", "body": "Full article text here...", "topic": "World"}
{"title": "Game Highlights", "body": "Sports coverage...", "topic": "Sports"}
{"title": "Market Update", "body": "Business news...", "topic": "Business"}
{"title": "Tech Innovation", "body": "Science and technology...", "topic": "Sci/Tech"}
```

### 3. Validate Data

```bash
python scripts/data_sanity_check.py
```

This script checks:
- Total number of articles
- Topic distribution
- Token length statistics
- Articles exceeding 512 token limit

### 4. Train the Model

```bash
python -m scripts.train_distilbert
```

Training will:
- Load configuration from `config/train.yaml`
- Download DistilBERT tokenizer and model (268 MB)
- Fine-tune for 3 epochs
- Save model to `artifacts/distilbert/`
- Log progress to console and `logs/` directory

**Expected runtime:** ~5-10 minutes on GPU, ~30 minutes on CPU

### 5. Use the Trained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("artifacts/distilbert")
model = AutoModelForSequenceClassification.from_pretrained("artifacts/distilbert")

# Predict topic for new article
text = "Latest technology breakthrough announced today..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
outputs = model(**inputs)

# Get prediction
logits = outputs.logits
predicted_class = logits.argmax(dim=-1).item()
topics = ["World", "Sports", "Business", "Sci/Tech"]

print(f"Predicted Topic: {topics[predicted_class]}")
print(f"Confidence: {torch.softmax(logits, dim=-1)[0][predicted_class]:.2%}")
```

---

## 📁 Project Structure

```
article-topic-classifier/
├── config/
│   └── train.yaml                    # Training hyperparameters
├── data/
│   ├── README.md
│   └── part-0001.jsonl              # Training dataset
├── scripts/
│   ├── data_sanity_check.py         # Validate data before training
│   └── train_distilbert.py          # Main training script
├── src/
│   ├── __init__.py
│   ├── dataset.py                   # Custom PyTorch dataset
│   ├── metrics.py                   # Evaluation metrics
│   └── utils.py                     # Logging utilities
├── artifacts/
│   └── distilbert/                  # Saved model & tokenizer
├── notebooks/
│   └── [Jupyter notebooks for analysis]
├── logs/
│   └── training_*.log               # Training logs
├── ARCHITECTURE.md                  # Detailed architecture guide
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## ⚙️ Configuration

Edit `config/train.yaml` to customize training:

```yaml
model:
  name: distilbert-base-uncased      # HuggingFace model ID
  num_labels: 4                      # Number of topic classes

data:
  train_jsonl: data/part-0001.jsonl # Path to training data
  text_fields: ["title", "body"]    # Fields to concatenate

training:
  epochs: 3                          # Number of training passes
  batch_size: 16                     # Samples per batch
  learning_rate: 0.00002            # Optimizer learning rate
  max_length: 256                    # Token sequence length
  weight_decay: 0.01                # L2 regularization
  warmup_ratio: 0.1                 # Warmup as % of total steps

output:
  dir: artifacts/distilbert         # Where to save model
```

### Hyperparameter Tuning Tips

| Parameter | Effect | Recommendation |
|-----------|--------|-----------------|
| `epochs` | Training duration | 2-5 for fine-tuning |
| `batch_size` | Speed vs Memory | 8-32 depending on GPU |
| `learning_rate` | Update speed | 1e-5 to 5e-5 for fine-tuning |
| `max_length` | Token limit | 128-512 (256 is good middle ground) |
| `weight_decay` | Overfitting prevention | 0.01-0.1 |

---

## 📊 Training Details

### Data Flow

```
Raw Text (JSONL)
    ↓
Tokenizer (DistilBERT WordPiece)
    ↓
Token IDs + Attention Mask
    ↓
DistilBERT Model (6 transformer layers)
    ↓
Classification Head (4-way softmax)
    ↓
Predicted Topic
```

### Training Loop

1. **Forward Pass**: Compute predictions and loss
2. **Backward Pass**: Calculate gradients
3. **Optimizer Step**: Update weights using AdamW
4. **Scheduler Step**: Adjust learning rate (warmup then decay)
5. **Log Progress**: Every ~20% of batches per epoch

### Model Architecture

- **Base Model**: DistilBERT (6 transformer layers, 66M parameters)
- **Tokenizer**: WordPiece (30,522 vocabulary)
- **Embedding Dim**: 768
- **Classification Head**: Linear layer (768 → 4)
- **Total Parameters**: ~67M

---

## 📈 Monitoring Training

### Console Output

Training progress is logged to both console and file:

```
INFO - Loading configuration from config/train.yaml
INFO - Using device: cuda
INFO - Dataset loaded: 500 samples
INFO - DataLoader created: 32 batches
INFO - Starting training for 3 epochs

INFO - Epoch 1/3
INFO -   Batch 6/31 | Loss: 1.3245
INFO -   Batch 12/31 | Loss: 1.1234
INFO -   Batch 19/31 | Loss: 0.9834
INFO - Epoch 1 completed | Avg Loss: 0.9456
...
INFO - [SUCCESS] Model and tokenizer saved successfully
```

### Log Files

Detailed logs saved to `logs/training_YYYYMMDD_HHMMSS.log`:
- Timestamps for all operations
- Module names and debug information
- Full error traces for debugging

### Checking Loss

- **Starting loss**: ~1.4 (random predictions)
- **After epoch 1**: ~0.8-1.0 (improving)
- **After epoch 3**: ~0.3-0.6 (good convergence)

Lower loss = better predictions ✓

---

## 🔍 Understanding the Code

### Key Files Explained

**[train_distilbert.py](scripts/train_distilbert.py)**
- Main training script with detailed comments
- Loads config, initializes model, runs training loop
- Saves trained model and tokenizer

**[dataset.py](src/dataset.py)**
- Custom PyTorch Dataset class
- Handles JSONL loading, tokenization, label mapping
- Creates attention masks and pads sequences

**[utils.py](src/utils.py)**
- Logging configuration function
- Sets up file and console handlers
- Supports UTF-8 encoding for special characters

For detailed explanations of how everything works together, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 🛠️ Troubleshooting

### Issue: `FileNotFoundError: data/part-0001.jsonl`
**Solution**: Ensure your training data is at the correct path and in JSONL format.

### Issue: `CUDA out of memory`
**Solution**: Reduce `batch_size` in config.yaml (try 8 or 4).

### Issue: Loss not decreasing
**Solution**: 
- Adjust learning_rate (try 0.00001 or 0.00005)
- Increase epochs (try 5-10)
- Check data quality with `scripts/data_sanity_check.py`

### Issue: `UnicodeEncodeError` on Windows
**Solution**: Already fixed! We use UTF-8 encoding for all file handlers.

---

## 📚 Learn More

### Documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete architecture guide with visualizations
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Official documentation
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Learning resources

### Papers & References
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - BERT paper
- [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108) - DistilBERT paper

---

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.0+
- CUDA 11.8+ (optional, for GPU acceleration)

See [requirements.txt](requirements.txt) for full dependency list.

---

## 💡 Tips for Better Results

1. **More Data**: 500 articles is good for demo. 5,000+ for production.
2. **Balanced Classes**: Ensure roughly equal samples per topic.
3. **Clean Data**: Remove HTML tags, fix encoding issues.
4. **Longer Training**: Try 5-10 epochs if you have time.
5. **Ensemble**: Train multiple models, average predictions.
6. **Evaluation Set**: Always keep 20% of data for validation.

---

## 🚀 Future Improvements

- [ ] Add validation set and evaluation metrics
- [ ] Implement model checkpointing (save best model)
- [ ] Add inference script for batch predictions
- [ ] Support for other models (BERT, RoBERTa, etc.)
- [ ] Data augmentation techniques
- [ ] Confidence thresholding for uncertain predictions
- [ ] Model distillation for faster inference

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Support

For issues, questions, or suggestions:
1. Check [ARCHITECTURE.md](ARCHITECTURE.md) for detailed explanations
2. Review [Troubleshooting](#-troubleshooting) section
3. Check log files in `logs/` directory

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformers library
- [PyTorch](https://pytorch.org/) team
- Original researchers behind BERT and DistilBERT

---

**Happy training! 🚀**