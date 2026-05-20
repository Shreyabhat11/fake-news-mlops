"""
data_pipeline/ingest.py
=======================
Ingests raw fake/true news CSVs, cleans text, creates train/val/test splits,
and saves DVC-versioned artifacts to data/processed/.

Expects:
    data/raw/Fake.csv  — fake news articles
    data/raw/True.csv  — real news articles

Outputs:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    data/processed/dataset_stats.json
"""

import json
import logging
import os
import re
import string
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Download NLTK resources (safe to call multiple times)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# ── Constants ────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
MIN_TEXT_LENGTH = 20


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Apply a sequence of cleaning steps to raw news text.

    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove HTML tags
        4. Remove punctuation / special chars (keep alphanumeric + spaces)
        5. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove Reuters / AP dateline artifacts like "WASHINGTON (Reuters) -"
    text = re.sub(r"\([^)]{0,30}\)\s*-\s*", " ", text)

    # Keep letters, digits, spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_combined_text(row: pd.Series) -> str:
    """
    Combine title + article body into a single string for embedding.
    The title is repeated to give it extra weight in embeddings.
    """
    title = clean_text(str(row.get("title", "")))
    body = clean_text(str(row.get("text", "")))
    # Truncate body to first 400 words to avoid memory issues with embeddings
    body_words = body.split()[:400]
    body = " ".join(body_words)
    return f"{title} {title} {body}".strip()


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_raw_data(fake_path: Path, true_path: Path) -> pd.DataFrame:
    """
    Load Fake.csv and True.csv, assign labels, and concatenate.

    Label convention:
        0 = Real news
        1 = Fake news
    """
    logger.info(f"Loading fake news from: {fake_path}")
    fake_df = pd.read_csv(fake_path)
    fake_df["label"] = 1
    fake_df["label_name"] = "FAKE"
    logger.info(f"Fake news rows: {len(fake_df)}")

    logger.info(f"Loading real news from: {true_path}")
    true_df = pd.read_csv(true_path)
    true_df["label"] = 0
    true_df["label_name"] = "REAL"
    logger.info(f"Real news rows: {len(true_df)}")

    df = pd.concat([fake_df, true_df], ignore_index=True)
    logger.info(f"Combined dataset size: {len(df)}")
    return df


def generate_synthetic_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate a synthetic dataset when real CSVs are not present.
    Useful for CI/CD and unit tests.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    fake_phrases = [
        "shocking conspiracy", "government hiding", "secret agenda",
        "mainstream media lies", "breaking exclusive", "deep state plot",
        "miracle cure revealed", "doctors don't want you to know",
    ]
    real_phrases = [
        "according to official sources", "analysts report",
        "study confirms", "experts say", "data shows",
        "government announced", "research published",
    ]

    rows = []
    for i in range(n_samples):
        is_fake = random.random() > 0.5
        if is_fake:
            title = f"SHOCKING: {random.choice(fake_phrases).upper()} story {i}"
            text = " ".join(random.choices(fake_phrases, k=50)) + f" article {i}"
            label = 1
        else:
            title = f"Report on {random.choice(real_phrases)} number {i}"
            text = " ".join(random.choices(real_phrases, k=50)) + f" analysis {i}"
            label = 0
        rows.append({"title": title, "text": text, "label": label,
                     "label_name": "FAKE" if is_fake else "REAL",
                     "subject": "politics", "date": "2023-01-01"})

    return pd.DataFrame(rows)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_ingestion(
    fake_csv: str = "data/raw/Fake.csv",
    true_csv: str = "data/raw/True.csv",
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> dict:
    """
    Full ingestion pipeline:
        1. Load raw CSVs (or synthetic data if missing)
        2. Clean text
        3. Create combined text column
        4. Remove short/null samples
        5. Split into train/val/test
        6. Save processed artifacts
        7. Return stats dict

    Returns:
        dict with dataset statistics
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    fake_path = Path(fake_csv)
    true_path = Path(true_csv)

    if fake_path.exists() and true_path.exists():
        df = load_raw_data(fake_path, true_path)
    else:
        logger.warning(
            "Raw CSVs not found. Using synthetic data for demonstration. "
            "Download Fake.csv & True.csv from Kaggle for production use."
        )
        df = generate_synthetic_data(n_samples=4000)

    # ── Clean ─────────────────────────────────────────────────────────────────
    logger.info("Cleaning text ...")
    df["combined_text"] = df.apply(build_combined_text, axis=1)

    # Drop rows with insufficient text
    before = len(df)
    df = df[df["combined_text"].str.split().str.len() >= MIN_TEXT_LENGTH]
    logger.info(f"Dropped {before - len(df)} rows with < {MIN_TEXT_LENGTH} words")

    # Shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # ── Split ─────────────────────────────────────────────────────────────────
    test_size = 1.0 - train_size - val_size
    assert test_size > 0, "train_size + val_size must be < 1.0"

    train_df, temp_df = train_test_split(
        df, train_size=train_size, stratify=df["label"], random_state=random_seed
    )
    relative_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, train_size=relative_val, stratify=temp_df["label"], random_state=random_seed
    )

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    logger.info(f"Saved processed splits to {PROCESSED_DIR}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "fake_pct": round(df["label"].mean() * 100, 2),
        "real_pct": round((1 - df["label"].mean()) * 100, 2),
        "avg_text_length_words": int(df["combined_text"].str.split().str.len().mean()),
    }
    with open(PROCESSED_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Dataset stats: {stats}")
    return stats


if __name__ == "__main__":
    stats = run_ingestion()
    print("\n✅ Ingestion complete:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
