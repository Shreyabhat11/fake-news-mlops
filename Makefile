# Makefile
# =========
# Developer convenience commands for the Fake News MLOps project.
# Usage: make <target>

.PHONY: help setup install lint test coverage docker-up docker-down \
        ingest features train drift orchestrate streamlit clean

PYTHON := python
PIP := pip
DOCKER_COMPOSE := docker compose

# ── Default ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Fake News Trend Drift Detector — Developer Commands"
	@echo "  ========================================================"
	@echo "  make setup          — create .env and install dependencies"
	@echo "  make install        — install Python requirements"
	@echo "  make lint           — run ruff + black"
	@echo "  make test           — run pytest suite"
	@echo "  make coverage       — run tests with HTML coverage report"
	@echo ""
	@echo "  make ingest         — run data ingestion pipeline"
	@echo "  make features       — run feature/embedding pipeline"
	@echo "  make train          — train logistic regression model"
	@echo "  make train-xgb      — train XGBoost model"
	@echo "  make drift          — run drift detection"
	@echo "  make orchestrate    — run full Prefect flow once"
	@echo ""
	@echo "  make docker-up      — start all Docker services"
	@echo "  make docker-down    — stop all Docker services"
	@echo "  make docker-logs    — tail all service logs"
	@echo "  make streamlit      — launch Streamlit dashboard locally"
	@echo "  make api            — run FastAPI dev server locally"
	@echo "  make mlflow         — start local MLflow server"
	@echo "  make clean          — remove generated artifacts"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────────
setup:
	@if [ ! -f .env ]; then cp .env.example .env; echo "✅ Created .env from .env.example"; fi
	$(MAKE) install

install:
	$(PIP) install --upgrade pip
	$(PIP) install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt
	$(PYTHON) -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
	@echo "✅ Dependencies installed"

# ── Code Quality ───────────────────────────────────────────────────────────────
lint:
	ruff check . --select E,W,F --ignore E501
	black --check .
	@echo "✅ Linting passed"

format:
	black .
	ruff check . --fix

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

coverage:
	pytest tests/ \
		--cov=api \
		--cov=data_pipeline \
		--cov=feature_pipeline \
		--cov=training_pipeline \
		--cov=monitoring \
		--cov-report=html:coverage_html \
		--cov-report=term-missing \
		-v
	@echo "✅ Coverage report: coverage_html/index.html"

# ── ML Pipeline ────────────────────────────────────────────────────────────────
ingest:
	$(PYTHON) data_pipeline/ingest.py

features:
	$(PYTHON) feature_pipeline/embeddings.py

train:
	$(PYTHON) training_pipeline/train.py --classifier logistic_regression

train-xgb:
	$(PYTHON) training_pipeline/train.py --classifier xgboost

drift:
	$(PYTHON) monitoring/drift_detector.py

orchestrate:
	$(PYTHON) orchestration/retrain_pipeline.py

# Full pipeline (end-to-end)
pipeline: ingest features train
	@echo "✅ Full training pipeline complete"

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	$(DOCKER_COMPOSE) up --build -d
	@echo "✅ Services starting..."
	@echo "   API:       http://localhost:8000"
	@echo "   MLflow:    http://localhost:5000"
	@echo "   Grafana:   http://localhost:3000 (admin/admin)"
	@echo "   Prometheus:http://localhost:9090"
	@echo "   Streamlit: http://localhost:8501"
	@echo "   Prefect:   http://localhost:4200"

docker-down:
	$(DOCKER_COMPOSE) down

docker-clean:
	$(DOCKER_COMPOSE) down -v --remove-orphans

docker-logs:
	$(DOCKER_COMPOSE) logs -f

docker-ps:
	$(DOCKER_COMPOSE) ps

# ── Local Dev Servers ──────────────────────────────────────────────────────────
api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns

streamlit:
	streamlit run streamlit_app/app.py --server.port 8501

prefect:
	prefect server start --host 0.0.0.0

# ── DVC ───────────────────────────────────────────────────────────────────────
dvc-init:
	dvc init
	dvc add data/raw/Fake.csv data/raw/True.csv
	dvc add data/processed/
	@echo "✅ DVC initialized. Run: git add .dvc && git commit"

dvc-push:
	dvc push

dvc-pull:
	dvc pull

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf coverage_html/ .coverage coverage.xml
	@echo "✅ Cleaned build artifacts"

clean-all: clean
	rm -rf models/*.pkl models/*.json
	rm -rf data/processed/
	rm -rf reports/
	rm -rf mlruns/
	@echo "✅ Cleaned all generated files"
