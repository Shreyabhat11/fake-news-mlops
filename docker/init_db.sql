-- docker/init_db.sql
-- Run automatically by the pgvector Docker image on first start

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for text search

-- Article embeddings table (pgvector)
CREATE TABLE IF NOT EXISTS article_embeddings (
    id          SERIAL PRIMARY KEY,
    article_id  VARCHAR(64) UNIQUE NOT NULL,
    text_snippet TEXT,
    label       INTEGER,
    label_name  VARCHAR(10),
    split       VARCHAR(10),
    embedding   vector(384),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- IVFFlat index for approximate nearest-neighbour search
CREATE INDEX IF NOT EXISTS idx_embedding_cosine
    ON article_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Prediction log for drift monitoring
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

-- Model registry shadow table (supplemental to MLflow)
CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(64),
    model_name      VARCHAR(128),
    version         VARCHAR(32),
    stage           VARCHAR(32) DEFAULT 'Staging',
    val_f1          FLOAT,
    test_f1         FLOAT,
    registered_at   TIMESTAMP DEFAULT NOW(),
    notes           TEXT
);
