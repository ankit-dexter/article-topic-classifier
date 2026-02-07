import json
from collections import Counter
import pandas as pd
from transformers import AutoTokenizer

# -------- CONFIG -------- #
JSONL_PATH = "data\part-0001.jsonl"   # adjust this
MODEL_NAME = "distilbert-base-uncased"
MAX_SAMPLES = 500  # set to e.g. 5000 to limit rows
# ------------------------ #

def load_jsonl(path, max_samples=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rows.append(json.loads(line))
            if max_samples and i + 1 >= max_samples:
                break
    return pd.DataFrame(rows)

def main():
    print("Loading data...")
    df = load_jsonl(JSONL_PATH, MAX_SAMPLES)

    print("\n=== BASIC STATS ===")
    print(f"Total articles: {len(df)}")

    print("\n=== TOPIC DISTRIBUTION ===")
    topic_counts = Counter(df["topic"])
    for topic, count in topic_counts.items():
        print(f"{topic}: {count}")

    print("\n=== TOKEN LENGTH STATS ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    lengths = []
    for text in (df["title"] + " " + df["body"]):
        tokens = tokenizer.tokenize(text)
        lengths.append(len(tokens))

    print(f"Average token length: {sum(lengths) / len(lengths):.2f}")
    print(f"Max token length: {max(lengths)}")
    print(f"Min token length: {min(lengths)}")

    over_limit = sum(1 for l in lengths if l > 512)
    print(f"Articles > 512 tokens: {over_limit} ({over_limit / len(lengths) * 100:.2f}%)")

if __name__ == "__main__":
    main()
