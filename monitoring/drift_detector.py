"""
monitoring/drift_detector.py
=============================
Detects data drift and concept drift using Evidently AI.

Drift types detected:
    1. Data drift       — feature distribution shift (embedding statistics)
    2. Concept drift    — prediction distribution shift
    3. Confidence drift — model confidence degradation

Reports:
    - HTML report   → reports/drift_report_<timestamp>.html
    - JSON summary  → reports/latest_drift_status.json

This module is called periodically by the orchestration layer.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Drift thresholds (configurable via env)
DATA_DRIFT_THRESHOLD = float(os.getenv("DATA_DRIFT_THRESHOLD", 0.15))
CONFIDENCE_DRIFT_THRESHOLD = float(os.getenv("CONFIDENCE_DRIFT_THRESHOLD", 0.10))
PREDICTION_DRIFT_THRESHOLD = float(os.getenv("PREDICTION_DRIFT_THRESHOLD", 0.10))


# ── Evidently Report Generation ───────────────────────────────────────────────

def build_evidently_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> tuple:
    """
    Build Evidently data drift and target drift reports.

    Args:
        reference_df: training data sample (ground truth distribution)
        current_df:   recent inference data

    Returns:
        (html_path, json_summary)
    """
    try:
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.report import Report
        from evidently.metrics import DatasetDriftMetric, DataDriftTable
    except ImportError:
        logger.error("evidently not installed. pip install evidently")
        return None, {}

    # Evidently requires common column names
    common_cols = [c for c in reference_df.columns if c in current_df.columns]
    ref = reference_df[common_cols].copy()
    cur = current_df[common_cols].copy()

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])

    report.run(reference_data=ref, current_data=cur)

    # Save HTML
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    html_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
    report.save_html(str(html_path))
    logger.info(f"Drift HTML report saved: {html_path}")

    # Extract JSON summary
    report_dict = report.as_dict()
    return html_path, report_dict


# ── Embedding Drift (Statistical) ─────────────────────────────────────────────

def compute_embedding_drift(
    reference_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
) -> dict:
    """
    Detect drift in embedding distributions using:
        - Mean cosine similarity between distributions
        - Wasserstein distance on principal components

    Returns dict with drift_score and is_drifted flag.
    """
    from scipy.spatial.distance import cosine
    from scipy.stats import ks_2samp, wasserstein_distance

    # Reduce to mean-per-dim for statistical comparison
    ref_mean = reference_embeddings.mean(axis=0)
    cur_mean = current_embeddings.mean(axis=0)

    # Cosine distance between mean embeddings
    mean_cosine_dist = cosine(ref_mean, cur_mean)

    # KS test on first 10 principal components
    from sklearn.decomposition import PCA
    n_components = min(10, reference_embeddings.shape[1], reference_embeddings.shape[0], current_embeddings.shape[0])
    pca = PCA(n_components=n_components)
    ref_pca = pca.fit_transform(reference_embeddings)
    cur_pca = pca.transform(current_embeddings)

    ks_pvalues = []
    wasserstein_dists = []
    for dim in range(n_components):
        ks_stat, ks_pval = ks_2samp(ref_pca[:, dim], cur_pca[:, dim])
        ks_pvalues.append(ks_pval)
        wasserstein_dists.append(wasserstein_distance(ref_pca[:, dim], cur_pca[:, dim]))

    # Fraction of dimensions with significant drift (p < threshold)
    drift_fraction = sum(p < DATA_DRIFT_THRESHOLD for p in ks_pvalues) / n_components
    avg_wasserstein = float(np.mean(wasserstein_dists))

    # Overall drift score: weighted combination
    drift_score = 0.5 * drift_fraction + 0.5 * min(mean_cosine_dist * 10, 1.0)

    return {
        "mean_cosine_distance": round(float(mean_cosine_dist), 4),
        "drift_fraction_pca": round(float(drift_fraction), 4),
        "avg_wasserstein_distance": round(avg_wasserstein, 4),
        "drift_score": round(float(drift_score), 4),
        "is_drifted": drift_score > DATA_DRIFT_THRESHOLD,
    }


def compute_prediction_drift(
    reference_preds: pd.Series,
    current_preds: pd.Series,
) -> dict:
    """
    Compare prediction class distributions between reference and current windows.
    Uses chi-squared test on label frequency distributions.
    """
    from scipy.stats import chi2_contingency

    # Class frequencies
    ref_fake_pct = reference_preds.mean()
    cur_fake_pct = current_preds.mean()

    # Contingency table
    n_ref = len(reference_preds)
    n_cur = len(current_preds)
    table = [
        [int(ref_fake_pct * n_ref), int((1 - ref_fake_pct) * n_ref)],
        [int(cur_fake_pct * n_cur), int((1 - cur_fake_pct) * n_cur)],
    ]

    try:
        chi2, p_value, _, _ = chi2_contingency(table)
    except Exception:
        chi2, p_value = 0.0, 1.0

    prediction_shift = abs(ref_fake_pct - cur_fake_pct)

    return {
        "reference_fake_pct": round(float(ref_fake_pct), 4),
        "current_fake_pct": round(float(cur_fake_pct), 4),
        "prediction_shift": round(float(prediction_shift), 4),
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_value), 4),
        "is_drifted": prediction_shift > PREDICTION_DRIFT_THRESHOLD,
    }


def compute_confidence_drift(
    reference_confidences: pd.Series,
    current_confidences: pd.Series,
) -> dict:
    """
    Detect degradation in model confidence scores.
    A significant drop in mean confidence suggests the model is uncertain
    about the current data distribution — a sign of concept drift.
    """
    ref_mean = reference_confidences.mean()
    cur_mean = current_confidences.mean()
    confidence_drop = ref_mean - cur_mean

    return {
        "reference_mean_confidence": round(float(ref_mean), 4),
        "current_mean_confidence": round(float(cur_mean), 4),
        "confidence_drop": round(float(confidence_drop), 4),
        "is_drifted": confidence_drop > CONFIDENCE_DRIFT_THRESHOLD,
    }


# ── Main Drift Detection Run ──────────────────────────────────────────────────

def run_drift_detection(
    reference_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict:
    """
    Run all drift detectors and save the consolidated status report.

    Args:
        reference_embeddings: embeddings from training set
        current_embeddings:   embeddings from recent inference window
        reference_df:         DataFrame with 'prediction', 'confidence' for reference
        current_df:           DataFrame with 'prediction', 'confidence' for current window

    Returns:
        Consolidated drift status dict
    """
    logger.info("Running drift detection ...")

    # ── Embedding drift ───────────────────────────────────────────────────────
    embedding_drift = compute_embedding_drift(reference_embeddings, current_embeddings)
    logger.info(f"Embedding drift: {embedding_drift}")

    # ── Prediction drift ──────────────────────────────────────────────────────
    pred_drift = compute_prediction_drift(
        reference_df["label"] if "label" in reference_df.columns else pd.Series([0] * len(reference_df)),
        current_df["prediction"].map({"FAKE": 1, "REAL": 0}).fillna(0),
    )
    logger.info(f"Prediction drift: {pred_drift}")

    # ── Confidence drift ──────────────────────────────────────────────────────
    ref_conf = reference_df.get("confidence", pd.Series([0.9] * len(reference_df)))
    cur_conf = current_df.get("confidence", pd.Series([0.8] * len(current_df)))
    conf_drift = compute_confidence_drift(ref_conf, cur_conf)
    logger.info(f"Confidence drift: {conf_drift}")

    # ── Evidently report ──────────────────────────────────────────────────────
    # Build feature DataFrame from first 10 embedding dims for Evidently
    n_features = min(10, reference_embeddings.shape[1])
    ref_feat_df = pd.DataFrame(
        reference_embeddings[:, :n_features],
        columns=[f"emb_{i}" for i in range(n_features)]
    )
    cur_feat_df = pd.DataFrame(
        current_embeddings[:, :n_features],
        columns=[f"emb_{i}" for i in range(n_features)]
    )
    html_path, _ = build_evidently_report(ref_feat_df, cur_feat_df)

    # ── Aggregate status ──────────────────────────────────────────────────────
    any_drifted = (
        embedding_drift["is_drifted"]
        or pred_drift["is_drifted"]
        or conf_drift["is_drifted"]
    )

    # Compute overall drift score
    overall_score = max(
        embedding_drift["drift_score"],
        pred_drift["prediction_shift"],
        conf_drift["confidence_drop"],
    )

    if overall_score > 0.35:
        status = "critical"
    elif overall_score > 0.15:
        status = "warning"
    else:
        status = "normal"

    status_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": status,
        "overall_drift_score": round(float(overall_score), 4),
        "should_retrain": overall_score > 0.20,
        "embedding_drift": embedding_drift,
        "prediction_drift": pred_drift,
        "confidence_drift": conf_drift,
        "html_report": str(html_path) if html_path else None,
    }

    # Save for API /drift/status endpoint
    with open(REPORTS_DIR / "latest_drift_status.json", "w") as f:
        json.dump(status_report, f, indent=2, default=str)

    logger.info(f"Drift status: {status} (score={overall_score:.4f})")
    return status_report


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run drift detection")
    parser.add_argument("--reference-window", type=int, default=1000)
    parser.add_argument("--current-window", type=int, default=200)
    args = parser.parse_args()

    # Load reference data
    ref_emb = np.load("data/processed/train_embeddings.npy")
    ref_labels = np.load("data/processed/train_labels.npy")

    # Sample
    idx = np.random.choice(len(ref_emb), min(args.reference_window, len(ref_emb)), replace=False)
    ref_emb_sample = ref_emb[idx]
    ref_df = pd.DataFrame({
        "label": ref_labels[idx],
        "confidence": np.random.uniform(0.85, 0.99, len(idx)),
    })

    # Simulate current window (with some drift for demo)
    cur_idx = np.random.choice(len(ref_emb), min(args.current_window, len(ref_emb)), replace=False)
    noise = np.random.normal(0, 0.05, ref_emb_sample[:args.current_window].shape)
    cur_emb_sample = ref_emb[cur_idx] + noise
    cur_df = pd.DataFrame({
        "prediction": np.random.choice(["FAKE", "REAL"], len(cur_idx), p=[0.55, 0.45]),
        "confidence": np.random.uniform(0.65, 0.85, len(cur_idx)),  # lower confidence = drift
    })

    result = run_drift_detection(ref_emb_sample, cur_emb_sample, ref_df, cur_df)
    print("\n✅ Drift detection complete:")
    print(json.dumps(result, indent=2, default=str))
