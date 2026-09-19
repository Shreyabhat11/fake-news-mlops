"""
tests/test_data_pipeline.py
============================
Unit tests for the data ingestion pipeline.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.ingest import (
    build_combined_text,
    clean_text,
    generate_synthetic_data,
    run_ingestion,
)


# ── clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_removes_urls(self):
        result = clean_text("Check https://example.com for details")
        assert "http" not in result
        assert "example" not in result

    def test_removes_html(self):
        result = clean_text("<b>Bold</b> and <i>italic</i>")
        assert "<" not in result
        assert "bold" in result

    def test_removes_punctuation(self):
        result = clean_text("Hello, world! Is this real?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_collapses_whitespace(self):
        result = clean_text("hello    world\n\ntest")
        assert "  " not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(123) == ""

    def test_reuters_dateline_removed(self):
        result = clean_text("WASHINGTON (Reuters) - The president announced ...")
        assert "reuters" not in result


# ── build_combined_text ────────────────────────────────────────────────────────

class TestBuildCombinedText:
    def test_combines_title_and_text(self):
        row = pd.Series({"title": "Big News", "text": "Something happened today"})
        result = build_combined_text(row)
        assert "big news" in result
        assert "something happened" in result

    def test_title_repeated(self):
        row = pd.Series({"title": "Hello World", "text": "Article body"})
        result = build_combined_text(row)
        # Title should appear at least twice
        assert result.count("hello world") >= 2

    def test_truncates_long_body(self):
        long_text = " ".join(["word"] * 1000)
        row = pd.Series({"title": "Title", "text": long_text})
        result = build_combined_text(row)
        word_count = len(result.split())
        assert word_count <= 410  # 400 body words + title words

    def test_missing_text_key(self):
        row = pd.Series({"title": "Title Only"})
        result = build_combined_text(row)
        assert "title only" in result


# ── generate_synthetic_data ────────────────────────────────────────────────────

class TestGenerateSyntheticData:
    def test_returns_dataframe(self):
        df = generate_synthetic_data(n_samples=100)
        assert isinstance(df, pd.DataFrame)

    def test_correct_size(self):
        df = generate_synthetic_data(n_samples=200)
        assert len(df) == 200

    def test_has_required_columns(self):
        df = generate_synthetic_data(n_samples=50)
        for col in ["title", "text", "label", "label_name"]:
            assert col in df.columns

    def test_binary_labels(self):
        df = generate_synthetic_data(n_samples=500)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_label_name_consistency(self):
        df = generate_synthetic_data(n_samples=200)
        assert all(df[df["label"] == 1]["label_name"] == "FAKE")
        assert all(df[df["label"] == 0]["label_name"] == "REAL")

    def test_reproducible(self):
        df1 = generate_synthetic_data(n_samples=100)
        df2 = generate_synthetic_data(n_samples=100)
        assert df1["title"].iloc[0] == df2["title"].iloc[0]


# ── run_ingestion ─────────────────────────────────────────────────────────────

class TestRunIngestion:
    """Integration test — runs full ingestion pipeline with synthetic data."""

    def setup_method(self):
        """Use a temporary directory for outputs."""
        self.tmp_dir = tempfile.mkdtemp()
        self.original_processed = Path("data/processed")
        # Monkey-patch the PROCESSED_DIR in the module
        import data_pipeline.ingest as ingest_module
        ingest_module.PROCESSED_DIR = Path(self.tmp_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        import data_pipeline.ingest as ingest_module
        ingest_module.PROCESSED_DIR = self.original_processed

    def test_produces_train_val_test_splits(self):
        stats = run_ingestion(
            fake_csv="nonexistent.csv",  # will use synthetic
            true_csv="nonexistent.csv",
        )
        for split in ["train", "val", "test"]:
            assert (Path(self.tmp_dir) / f"{split}.csv").exists()

    def test_returns_stats_dict(self):
        stats = run_ingestion(fake_csv="n/a", true_csv="n/a")
        for key in ["total_samples", "train_samples", "val_samples", "test_samples"]:
            assert key in stats

    def test_split_sizes_sum_to_total(self):
        stats = run_ingestion(fake_csv="n/a", true_csv="n/a")
        assert (stats["train_samples"] + stats["val_samples"] + stats["test_samples"]
                == stats["total_samples"])

    def test_dataset_stats_json_saved(self):
        run_ingestion(fake_csv="n/a", true_csv="n/a")
        stats_file = Path(self.tmp_dir) / "dataset_stats.json"
        assert stats_file.exists()
        with open(stats_file) as f:
            data = json.load(f)
        assert "fake_pct" in data

    def test_no_short_texts_in_output(self):
        from data_pipeline.ingest import MIN_TEXT_LENGTH
        run_ingestion(fake_csv="n/a", true_csv="n/a")
        train_df = pd.read_csv(Path(self.tmp_dir) / "train.csv")
        word_counts = train_df["combined_text"].str.split().str.len()
        assert (word_counts >= MIN_TEXT_LENGTH).all()
