# 🔍 TruthLens — Fake News Detection MLOps

> A production-style machine learning system for fake news classification, with a FastAPI inference service and a modern React frontend.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-TypeScript-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.x-646CFF.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌐 Live Demo

**TruthLens frontend:**  
https://fake-news-mlops.vercel.app/

**FastAPI backend:**  
https://fake-news-mlops.onrender.com

**Interactive API docs:**  
https://fake-news-mlops.onrender.com/docs

TruthLens lets users submit a news headline and article body and receive a machine-learning prediction of **REAL** or **FAKE**, together with confidence and probability information.

> **Important:** TruthLens is a machine-learning classifier, not a fact-checking engine. It does not independently verify claims against external sources.

---

## ✨ What This Project Does

The project takes a news article through the following workflow:

```text
Article
   │
   ▼
Text preprocessing
   │
   ▼
Sentence Transformer
all-MiniLM-L6-v2
   │
   ▼
384-dimensional embedding
   │
   ▼
Logistic Regression classifier
   │
   ▼
REAL / FAKE prediction
   │
   ├── confidence
   ├── real probability
   └── fake probability
```

The trained model is served through FastAPI and consumed by the React frontend.

---

## 🏗️ Current Production Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                         USER                                 │
│                                                              │
│                  TruthLens React UI                          │
│             TypeScript + Vite + Tailwind                    │
│                                                              │
│                    Vercel                                    │
│      https://fake-news-mlops.vercel.app/                    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           │ HTTPS
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Inference API                     │
│                                                              │
│                 Render deployment                            │
│        https://fake-news-mlops.onrender.com                  │
│                                                              │
│  /health       /predict       /batch_predict                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    Inference Pipeline                        │
│                                                              │
│  all-MiniLM-L6-v2  →  384-d embedding  →  Logistic Regression│
│                                                              │
│                         model.pkl                            │
└──────────────────────────────────────────────────────────────┘
```

### Deployment scope

The current public deployment intentionally keeps the runtime lightweight:

- **Frontend:** Vercel
- **API:** Render
- **Model:** tracked deployment artifact
- **Embedding model:** `all-MiniLM-L6-v2`
- **Redis:** optional; not enabled in the current Render deployment
- **PostgreSQL/pgvector:** not required for the current inference deployment
- **MLflow / Prefect / Prometheus / Grafana:** retained as project/MLOps components but not required for the public inference path

This keeps the deployed application practical while preserving the broader MLOps architecture in the repository.

---

## 🧠 Model

### Dataset

The project uses the Fake and Real News Dataset containing:

- **44,856 total articles**
- **35,884 training samples**
- **4,486 validation samples**
- **4,486 test samples**
- 52.26% fake
- 47.74% real

### Feature extraction

Each article is converted into a semantic embedding using:

**Sentence Transformers — `all-MiniLM-L6-v2`**

- Embedding dimension: **384**

### Classifier

The deployed model is:

**Logistic Regression**

The model was selected and trained using the generated sentence embeddings.

### Test performance

| Metric | Test Result |
|---|---:|
| F1 Score | **0.9342** |
| ROC-AUC | **0.9838** |

Test confusion matrix:

```text
[[2001, 141],
 [ 154, 2190]]
```

These metrics describe the model's offline test-set performance. They are not guarantees of factual accuracy on new articles.

---

## 🖥️ TruthLens Frontend

The original Streamlit interface was replaced with a dedicated web frontend.

### Frontend stack

- React
- TypeScript
- Vite
- Tailwind CSS
- Lucide React
- REST API integration

### Main features

- Single article analysis
- Headline + article body input
- REAL / FAKE result visualization
- Confidence display
- Real/fake probability visualization
- API/system health indicator
- Render cold-start/loading handling
- User-friendly API error handling
- Example articles
- Batch analysis
- Model information
- How-it-works explanation
- Responsive desktop/mobile layout
- Production API integration

The frontend does **not** implement the ML logic itself. It sends requests to the deployed FastAPI service and renders the actual model response.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (optional)
- Git

---

### 1. Clone the repository

```bash
git clone https://github.com/Shreyabhat11/fake-news-mlops.git
cd fake-news-mlops
```

---

### 2. Backend setup

Create a Python environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The API-specific deployment dependencies are maintained separately in:

```text
requirements-api.txt
```

---

### 3. Run the FastAPI service locally

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## 🎨 Run the Frontend Locally

The production frontend is located in:

```text
truthlens-frontend/
```

Install dependencies:

```bash
cd truthlens-frontend
npm install
```

Create a `.env` file:

```env
VITE_API_URL=http://localhost:8000
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

### Production build

```bash
npm run build
```

The production build is generated in:

```text
truthlens-frontend/dist/
```

---

## 🔌 API Reference

### `GET /health`

Returns API and model readiness information.

Example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "embedding_service_ready": true,
  "redis_connected": false,
  "timestamp": "2026-09-20T11:29:28.111002"
}
```

`redis_connected: false` is expected in the current lightweight public deployment because Redis is optional.

---

### `POST /predict`

Classifies one article.

Request:

```json
{
  "title": "Government announces new economic policy",
  "text": "The government announced a new economic policy today."
}
```

Example response:

```json
{
  "request_id": "d2e285f2-a4c5-4fc6-9dc4-537c2470ede6",
  "prediction": "REAL",
  "label": 0,
  "confidence": 0.8778,
  "fake_probability": 0.1222,
  "real_probability": 0.8778,
  "drift_status": "normal",
  "model_version": "v1.0",
  "processing_time_ms": 2562.13,
  "cached": false
}
```

The example above is from a successful request to the deployed Render API.

---

### `POST /batch_predict`

Supports batch article classification.

The frontend exposes this as a secondary batch-analysis workflow.

---

### Additional API endpoints

The FastAPI application also contains endpoints for functionality such as:

- Prometheus metrics
- drift status
- model reload

Refer to the interactive Swagger documentation for the current endpoint contract:

https://fake-news-mlops.onrender.com/docs

---

## 📊 Drift Detection

The project includes drift-monitoring components designed to detect changes in model inputs and prediction behavior.

The monitoring code includes checks for:

- data drift
- prediction/class-distribution changes
- confidence degradation

The API exposes a drift status used by the inference response.

The current public inference deployment reports the drift status as part of each prediction response.

---

## 🔄 MLOps Components

The repository was designed as more than a standalone classifier and contains components for an end-to-end MLOps workflow:

```text
Data ingestion
      │
      ▼
Preprocessing
      │
      ▼
Embedding generation
      │
      ▼
Model training
      │
      ▼
Model artifact
      │
      ▼
FastAPI inference
      │
      ▼
Frontend
```

Additional repository components support:

- DVC-based data versioning
- MLflow experiment/model tracking
- PostgreSQL + pgvector integration
- Redis caching
- Prefect orchestration
- drift detection
- Prometheus metrics
- Grafana dashboards
- Docker
- GitHub Actions

These components are part of the broader project architecture; the public deployment currently uses the lightweight FastAPI inference path described above.

---

## 📁 Project Structure

```text
fake-news-mlops/
│
├── api/
│   ├── main.py
│   └── prediction_logger.py
│
├── data_pipeline/
│   └── ingest.py
│
├── feature_pipeline/
│   └── embeddings.py
│
├── training_pipeline/
│   └── train.py
│
├── monitoring/
│   └── drift_detector.py
│
├── orchestration/
│   └── retrain_pipeline.py
│
├── observability/
│   ├── prometheus/
│   └── grafana/
│
├── models/
│   ├── model.pkl
│   └── latest_model_metadata.json
│
├── tests/
│
├── docker/
│   └── Dockerfile.api
│
├── truthlens-frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── types/
│   │   └── lib/
│   ├── package.json
│   └── vite.config.ts
│
├── configs/
├── requirements.txt
├── requirements-api.txt
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## 🧪 Testing

Backend tests can be run with:

```bash
pytest
```

For a specific API test module:

```bash
pytest tests/test_api.py -v
```

Frontend production build:

```bash
cd truthlens-frontend
npm run build
```

The deployed API was also manually verified using:

```powershell
Invoke-RestMethod https://fake-news-mlops.onrender.com/health
```

and:

```powershell
Invoke-RestMethod -Method Post `
  -Uri https://fake-news-mlops.onrender.com/predict `
  -ContentType "application/json" `
  -Body '{"title":"Government announces new economic policy","text":"The government announced a new economic policy today."}'
```

---

## 🐳 Docker

The API has a dedicated deployment image:

```text
docker/Dockerfile.api
```

The API container uses a CPU-only PyTorch setup and a single Uvicorn worker to keep the inference service lightweight enough for the current Render deployment.

Build locally:

```bash
docker build -f docker/Dockerfile.api -t fake-news-api .
```

Run:

```bash
docker run -p 8000:8000 fake-news-api
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + TypeScript + Vite |
| Styling | Tailwind CSS |
| Frontend hosting | Vercel |
| API | FastAPI + Uvicorn |
| API hosting | Render |
| NLP embeddings | Sentence Transformers |
| Embedding model | all-MiniLM-L6-v2 |
| Classifier | Scikit-learn Logistic Regression |
| Data | Fake and Real News Dataset |
| MLOps | MLflow, DVC, drift monitoring |
| Optional storage | PostgreSQL + pgvector |
| Optional caching | Redis |
| Orchestration | Prefect |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Containers | Docker |

---

## 🔮 Future Improvements

- [ ] Add user feedback collection for incorrect predictions
- [ ] Improve model evaluation on newer/out-of-distribution news
- [ ] Add explainability for model predictions
- [ ] Expand multilingual support
- [ ] Add richer drift visualizations
- [ ] Add automated model evaluation gates before deployment
- [ ] Add production monitoring dashboards to the deployed environment
- [ ] Explore transformer-based classifiers such as BERT/RoBERTa

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 👤 Project

Built as an AI/ML and MLOps portfolio project demonstrating:

- NLP feature engineering
- semantic embeddings
- supervised classification
- FastAPI model serving
- Dockerized inference
- production API deployment
- React frontend development
- frontend-to-ML API integration
- drift monitoring concepts
- reproducible ML workflows

**Live application:** https://fake-news-mlops.vercel.app/
