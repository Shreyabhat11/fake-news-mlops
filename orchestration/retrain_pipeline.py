"""
orchestration/retrain_pipeline.py
===================================
Prefect flow that runs on a schedule to:
    1. Pull recent predictions from PostgreSQL
    2. Run drift detection
    3. Trigger retraining if drift exceeds threshold
    4. Register the new model version in MLflow
    5. Hot-reload the API model

Run locally:
    python orchestration/retrain_pipeline.py

Deploy to Prefect Cloud:
    prefect deploy orchestration/retrain_pipeline.py:drift_monitoring_flow
"""

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
RETRAIN_DRIFT_SCORE = float(os.getenv("RETRAIN_DRIFT_SCORE", 0.20))
REFERENCE_WINDOW = int(os.getenv("REFERENCE_WINDOW", 1000))
CURRENT_WINDOW = int(os.getenv("CURRENT_WINDOW", 200))


try:
    from prefect import flow, task
    from prefect.logging import get_run_logger
    PREFECT_AVAILABLE = True
except ImportError:
    # Graceful fallback: define no-op decorators so the module still imports
    PREFECT_AVAILABLE = False
    def flow(**kwargs):
        def decorator(fn):
            return fn
        return decorator
    def task(**kwargs):
        def decorator(fn):
            return fn
        return decorator


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="fetch_reference_embeddings", retries=2, retry_delay_seconds=10)
def fetch_reference_embeddings(window: int = REFERENCE_WINDOW) -> tuple:
    """Load reference embeddings from training data."""
    emb_path = Path("data/processed/train_embeddings.npy")
    lbl_path = Path("data/processed/train_labels.npy")

    if not emb_path.exists():
        raise FileNotFoundError("Training embeddings not found. Run feature pipeline first.")

    embeddings = np.load(emb_path)
    labels = np.load(lbl_path)

    # Sample reference window
    idx = np.random.choice(len(embeddings), min(window, len(embeddings)), replace=False)
    ref_emb = embeddings[idx]
    ref_df = pd.DataFrame({
        "label": labels[idx],
        "confidence": np.random.uniform(0.85, 0.99, len(idx)),  # simulated
    })
    logger.info(f"Reference window: {len(idx)} samples")
    return ref_emb, ref_df


@task(name="fetch_current_predictions", retries=2, retry_delay_seconds=10)
def fetch_current_predictions(window: int = CURRENT_WINDOW) -> tuple:
    """
    Fetch recent inference requests from the prediction_log table.
    Falls back to simulating drift if DB is not available.
    """
    try:
        import os
        from sqlalchemy import create_engine, text

        db_url = (
            f"postgresql://{os.getenv('POSTGRES_USER', 'mlops_user')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'mlops_pass')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'fakenews_db')}"
        )
        engine = create_engine(db_url)
        sql = text("""
            SELECT prediction, confidence
            FROM prediction_log
            ORDER BY created_at DESC
            LIMIT :limit
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql, {"limit": window}).fetchall()

        if rows:
            cur_df = pd.DataFrame(rows, columns=["prediction", "confidence"])
            logger.info(f"Fetched {len(cur_df)} recent predictions from DB.")
        else:
            raise ValueError("No predictions in DB yet.")

    except Exception as e:
        logger.warning(f"Could not fetch predictions from DB: {e}. Using simulated data.")
        # Simulate a drifted distribution for demonstration
        cur_df = pd.DataFrame({
            "prediction": np.random.choice(
                ["FAKE", "REAL"], window, p=[0.6, 0.4]  # shifted from balanced
            ),
            "confidence": np.random.uniform(0.55, 0.80, window),  # lower confidence
        })

    # Load corresponding embeddings (approximate: use test embeddings with noise)
    test_emb_path = Path("data/processed/test_embeddings.npy")
    if test_emb_path.exists():
        test_emb = np.load(test_emb_path)
        idx = np.random.choice(len(test_emb), min(len(cur_df), len(test_emb)), replace=True)
        noise = np.random.normal(0, 0.05, test_emb[idx].shape)
        cur_emb = test_emb[idx] + noise
    else:
        # Pure noise as fallback
        cur_emb = np.random.randn(len(cur_df), 384).astype(np.float32)

    return cur_emb, cur_df


@task(name="run_drift_analysis", retries=1)
def run_drift_analysis(
    ref_emb: np.ndarray,
    ref_df: pd.DataFrame,
    cur_emb: np.ndarray,
    cur_df: pd.DataFrame,
) -> dict:
    """Run all drift detectors and return the status report."""
    from monitoring.drift_detector import run_drift_detection
    return run_drift_detection(ref_emb, cur_emb, ref_df, cur_df)


@task(name="trigger_retraining", retries=2, retry_delay_seconds=30)
def trigger_retraining(drift_report: dict, classifier: str = "logistic_regression") -> dict:
    """
    Execute the training pipeline if drift exceeds threshold.
    Returns retraining result dict.
    """
    drift_score = drift_report.get("overall_drift_score", 0.0)

    if not drift_report.get("should_retrain", False):
        logger.info(f"Drift score {drift_score:.4f} below threshold. No retraining needed.")
        return {"retrained": False, "drift_score": drift_score}

    logger.info(f"🔄 Drift score {drift_score:.4f} exceeds threshold. Triggering retraining ...")

    # Step 1: Re-run data ingestion (refresh processed data)
    logger.info("Step 1: Refreshing data ...")
    result = subprocess.run(
        ["python", "data_pipeline/ingest.py"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        logger.error(f"Data ingestion failed: {result.stderr}")
        raise RuntimeError(f"Data ingestion failed: {result.stderr}")

    # Step 2: Re-run feature pipeline
    logger.info("Step 2: Regenerating embeddings ...")
    result = subprocess.run(
        ["python", "feature_pipeline/embeddings.py"],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        logger.warning(f"Feature pipeline warning: {result.stderr}")

    # Step 3: Retrain model
    logger.info(f"Step 3: Training new {classifier} model ...")
    result = subprocess.run(
        ["python", "training_pipeline/train.py", "--classifier", classifier],
        capture_output=True, text=True, timeout=1800
    )
    if result.returncode != 0:
        logger.error(f"Training failed: {result.stderr}")
        raise RuntimeError(f"Training failed: {result.stderr}")

    logger.info("Training output:\n" + result.stdout[-1000:])
    return {"retrained": True, "drift_score": drift_score, "classifier": classifier}


@task(name="reload_api_model", retries=3, retry_delay_seconds=5)
def reload_api_model() -> bool:
    """
    Send a POST /model/reload to the inference API to hot-reload the new model.
    No service restart required.
    """
    try:
        response = requests.post(f"{API_BASE_URL}/model/reload", timeout=30)
        response.raise_for_status()
        logger.info(f"API model reloaded: {response.json()}")
        return True
    except Exception as e:
        logger.error(f"API reload failed (API may not be running): {e}")
        return False


@task(name="notify_drift_alert")
def notify_drift_alert(drift_report: dict) -> None:
    """
    Send drift alert. In production, replace with:
    - Slack webhook
    - PagerDuty
    - Email via SendGrid
    """
    status = drift_report.get("status", "unknown")
    score = drift_report.get("overall_drift_score", 0.0)
    logger.warning(
        f"🚨 DRIFT ALERT: status={status}, score={score:.4f}, "
        f"retrain={'YES' if drift_report.get('should_retrain') else 'NO'}"
    )
    # TODO: Add real notification here
    # e.g.: requests.post(SLACK_WEBHOOK, json={"text": f"Drift alert: {status}"})


# ── Main Flow ─────────────────────────────────────────────────────────────────

@flow(name="drift_monitoring_and_retraining", log_prints=True)
def drift_monitoring_flow(
    classifier: str = "logistic_regression",
    reference_window: int = REFERENCE_WINDOW,
    current_window: int = CURRENT_WINDOW,
) -> dict:
    """
    Full drift monitoring + retraining Prefect flow.

    Schedule this to run every 1-6 hours in production.

    Steps:
        1. Fetch reference embeddings
        2. Fetch current prediction window
        3. Run drift analysis (Evidently + statistical tests)
        4. Alert if drift detected
        5. Retrain model if score exceeds threshold
        6. Hot-reload API model
    """
    logger.info(f"Starting drift monitoring flow at {datetime.utcnow().isoformat()}")

    # Fetch data
    ref_emb, ref_df = fetch_reference_embeddings(window=reference_window)
    cur_emb, cur_df = fetch_current_predictions(window=current_window)

    # Detect drift
    drift_report = run_drift_analysis(ref_emb, ref_df, cur_emb, cur_df)

    # Alert on any drift
    if drift_report.get("status") != "normal":
        notify_drift_alert(drift_report)

    # Retrain if needed
    retrain_result = trigger_retraining(drift_report, classifier=classifier)

    if retrain_result.get("retrained"):
        # Hot-reload the API
        reloaded = reload_api_model()
        logger.info(f"Model reloaded: {reloaded}")

    summary = {
        "flow_completed_at": datetime.utcnow().isoformat(),
        "drift_status": drift_report.get("status"),
        "drift_score": drift_report.get("overall_drift_score"),
        "retrained": retrain_result.get("retrained", False),
    }
    logger.info(f"Flow complete: {summary}")
    return summary


# ── Scheduled Deployment Config ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run drift monitoring flow")
    parser.add_argument("--classifier", default="logistic_regression",
                        choices=["logistic_regression", "xgboost"])
    parser.add_argument("--reference-window", type=int, default=REFERENCE_WINDOW)
    parser.add_argument("--current-window", type=int, default=CURRENT_WINDOW)
    parser.add_argument("--schedule", action="store_true",
                        help="Deploy as a Prefect scheduled flow (every 2 hours)")
    args = parser.parse_args()

    if args.schedule and PREFECT_AVAILABLE:
        # Deploy with a 2-hour interval schedule
        from prefect.client.schemas.schedules import IntervalSchedule

        drift_monitoring_flow.serve(
            name="drift-monitoring-scheduled",
            interval=timedelta(hours=2),
            parameters={
                "classifier": args.classifier,
                "reference_window": args.reference_window,
                "current_window": args.current_window,
            },
        )
    else:
        # Run once immediately
        result = drift_monitoring_flow(
            classifier=args.classifier,
            reference_window=args.reference_window,
            current_window=args.current_window,
        )
        print("\n✅ Flow complete:")
        print(json.dumps(result, indent=2))
