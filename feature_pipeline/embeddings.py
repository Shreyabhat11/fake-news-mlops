"""
feature_pipeline/embeddings.py
================================
Generates sentence embeddings using Sentence Transformers and stores them
in PostgreSQL via the pgvector extension.

Key responsibilities:
    - Load processed CSV splits
    - Generate 384-dim embeddings using all-MiniLM-L6-v2
    - Upsert embeddings into the `embeddings` table (pgvector)
    - Cache embeddings in Redis for inference-time speed
    - Expose a lookup function used by the API
"""

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import redis
from sentence_transformers import SentenceTransformer
from sqlalchemy import Column, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── DB Setup ─────────────────────────────────────────────────────────────────

DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'mlops_user')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'mlops_pass')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'fakenews_db')}"
)

EMBEDDING_DIM = 384
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/0"
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))


class Base(DeclarativeBase):
    pass


def get_engine():
    """Create a SQLAlchemy engine; enables pgvector extension."""
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    return engine


def create_embeddings_table(engine) -> None:
    """
    Create the `article_embeddings` table with a pgvector column.
    Uses raw SQL for pgvector compatibility (SQLAlchemy type isn't standard).
    """
    ddl = f"""
    CREATE TABLE IF NOT EXISTS article_embeddings (
        id          SERIAL PRIMARY KEY,
        article_id  VARCHAR(64) UNIQUE NOT NULL,  -- SHA256 hash of text
        text_snippet TEXT,
        label       INTEGER,
        label_name  VARCHAR(10),
        split       VARCHAR(10),                   -- train / val / test
        embedding   vector({EMBEDDING_DIM}),
        created_at  TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_embedding_cosine
        ON article_embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    logger.info("Embeddings table ready.")


def get_redis_client() -> Optional[redis.Redis]:
    """Return a Redis client, or None if unavailable."""
    try:
        client = redis.from_url(REDIS_URL, decode_responses=False)
        client.ping()
        logger.info("Redis connected.")
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable, caching disabled: {e}")
        return None


# ── Embedding Model ───────────────────────────────────────────────────────────

class EmbeddingService:
    """
    Wraps SentenceTransformer with Redis caching.
    Embeddings are cached by SHA256(text) to avoid recomputing on repeat requests.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.redis = get_redis_client()
        logger.info("EmbeddingService ready.")

    def _cache_key(self, text: str) -> str:
        return "emb:" + hashlib.sha256(text.encode()).hexdigest()

    def embed_single(self, text: str) -> np.ndarray:
        """
        Return a 384-dim embedding for a single text string.
        Checks Redis cache first; falls back to model inference.
        """
        key = self._cache_key(text)

        # Cache hit
        if self.redis:
            cached = self.redis.get(key)
            if cached:
                return np.frombuffer(cached, dtype=np.float32)

        # Model inference
        embedding = self.model.encode(text, normalize_embeddings=True).astype(np.float32)

        # Store in cache
        if self.redis:
            self.redis.setex(key, CACHE_TTL, embedding.tobytes())

        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts, returning shape (N, EMBEDDING_DIM).
        Uncached texts are batched through the model; cached ones are retrieved individually.
        """
        results = {}
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if self.redis:
                cached = self.redis.get(key)
                if cached:
                    results[i] = np.frombuffer(cached, dtype=np.float32)
                    continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        # Batch encode uncached
        if uncached_texts:
            embeddings = self.model.encode(
                uncached_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(uncached_texts) > 100,
            ).astype(np.float32)

            for idx, emb, text in zip(uncached_indices, embeddings, uncached_texts):
                results[idx] = emb
                if self.redis:
                    key = self._cache_key(text)
                    self.redis.setex(key, CACHE_TTL, emb.tobytes())

        return np.array([results[i] for i in range(len(texts))])


# ── Storage ───────────────────────────────────────────────────────────────────

def upsert_embeddings(
    engine,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    split_name: str,
) -> None:
    """
    Insert or update embeddings in PostgreSQL.
    Uses ON CONFLICT DO UPDATE to allow re-runs without duplicates.
    """
    insert_sql = text("""
        INSERT INTO article_embeddings
            (article_id, text_snippet, label, label_name, split, embedding)
        VALUES
            (:article_id, :text_snippet, :label, :label_name, :split, :embedding)
        ON CONFLICT (article_id) DO UPDATE SET
            embedding   = EXCLUDED.embedding,
            text_snippet = EXCLUDED.text_snippet,
            split       = EXCLUDED.split
    """)

    rows = []
    for i, row in df.iterrows():
        emb_list = embeddings[i].tolist()
        rows.append({
            "article_id": hashlib.sha256(row["combined_text"].encode()).hexdigest(),
            "text_snippet": row["combined_text"][:200],
            "label": int(row["label"]),
            "label_name": str(row.get("label_name", "")),
            "split": split_name,
            "embedding": f"[{','.join(map(str, emb_list))}]",
        })

    with engine.connect() as conn:
        for chunk_start in range(0, len(rows), 500):
            chunk = rows[chunk_start: chunk_start + 500]
            conn.execute(insert_sql, chunk)
        conn.commit()

    logger.info(f"Upserted {len(rows)} embeddings for split='{split_name}'.")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_feature_pipeline(
    processed_dir: str = "data/processed",
    model_name: str = EMBEDDING_MODEL_NAME,
) -> dict:
    """
    Full feature pipeline:
        1. Load processed CSV splits
        2. Generate embeddings for each split
        3. Upsert into PostgreSQL (pgvector)
        4. Save numpy arrays locally for training
    """
    processed_path = Path(processed_dir)

    # PostgreSQL/pgvector storage is optional.
    # Local .npy embeddings are sufficient for model training.
    store_in_db = os.getenv("STORE_EMBEDDINGS_IN_DB", "false").lower() == "true"

    engine = None
    if store_in_db:
        engine = get_engine()
        create_embeddings_table(engine)

    service = EmbeddingService(model_name=model_name)
    stats = {}

    for split in ["train", "val", "test"]:
        csv_path = processed_path / f"{split}.csv"
        if not csv_path.exists():
            logger.warning(f"{csv_path} not found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        logger.info(f"Generating embeddings for {split} ({len(df)} samples)...")
        embeddings = service.embed_batch(df["combined_text"].tolist())

        # Save locally for quick training access
        np.save(processed_path / f"{split}_embeddings.npy", embeddings)
        np.save(processed_path / f"{split}_labels.npy", df["label"].values)
        logger.info(f"Saved {split} embeddings to disk: {embeddings.shape}")

        # Optionally store embeddings in pgvector
        if store_in_db:
            try:
                upsert_embeddings(engine, df, embeddings, split_name=split)
            except Exception as e:
                logger.error(f"pgvector upsert failed: {e}")
                logger.info("Embeddings saved to disk only.")

        stats[split] = {"samples": len(df), "embedding_dim": embeddings.shape[1]}

    logger.info("Feature pipeline complete.")
    return stats


# ── Lookup for Inference ──────────────────────────────────────────────────────

def get_embedding_for_inference(text: str, service: EmbeddingService) -> np.ndarray:
    """
    Called by the FastAPI /predict endpoint.
    Returns a 384-dim float32 numpy array.
    """
    return service.embed_single(text)


if __name__ == "__main__":
    stats = run_feature_pipeline()
    print("\n✅ Feature pipeline complete:")
    for split, s in stats.items():
        print(f"   {split}: {s}")
