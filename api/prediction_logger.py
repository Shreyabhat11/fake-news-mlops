"""
api/prediction_logger.py
=========================
Logs every inference request to PostgreSQL for:
    - Drift detection (monitoring recent distribution shifts)
    - Audit trail
    - Performance dashboards
"""

import logging
import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'mlops_user')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'mlops_pass')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'fakenews_db')}"
)


class Base(DeclarativeBase):
    pass


class PredictionLogger:
    """
    Async-compatible prediction logger.
    Creates the `prediction_log` table on first use.
    """

    def __init__(self):
        self.engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        self._create_table()

    def _create_table(self):
        ddl = """
        CREATE TABLE IF NOT EXISTS prediction_log (
            id              SERIAL PRIMARY KEY,
            request_id      VARCHAR(64),
            text_snippet    TEXT,
            prediction      VARCHAR(10),
            confidence      FLOAT,
            model_version   VARCHAR(32),
            created_at      TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_pred_log_created
            ON prediction_log (created_at DESC);
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(ddl))
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not create prediction_log table: {e}")

    async def log(
        self,
        request_id: str,
        text_snippet: str,
        prediction: str,
        confidence: float,
        model_version: str,
    ) -> None:
        """Non-blocking log (runs in thread pool via asyncio)."""
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_log,
            request_id, text_snippet, prediction, confidence, model_version
        )

    def _sync_log(self, request_id, text_snippet, prediction, confidence, model_version):
        sql = text("""
            INSERT INTO prediction_log
                (request_id, text_snippet, prediction, confidence, model_version)
            VALUES
                (:request_id, :text_snippet, :prediction, :confidence, :model_version)
        """)
        try:
            with self.engine.connect() as conn:
                conn.execute(sql, {
                    "request_id": request_id,
                    "text_snippet": text_snippet[:200],
                    "prediction": prediction,
                    "confidence": confidence,
                    "model_version": model_version,
                })
                conn.commit()
        except Exception as e:
            logger.warning(f"Prediction log write failed: {e}")

    def get_recent_predictions(self, limit: int = 500) -> list:
        """Fetch recent predictions for drift analysis."""
        sql = text("""
            SELECT prediction, confidence, model_version, created_at
            FROM prediction_log
            ORDER BY created_at DESC
            LIMIT :limit
        """)
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sql, {"limit": limit})
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.warning(f"Could not fetch predictions: {e}")
            return []
