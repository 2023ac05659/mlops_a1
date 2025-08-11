# Iris Classification Project – Summary

## 1. Overview
This project implements a full MLOps workflow for an Iris flower classification model, covering:
- Data preparation and versioning
- Model training and experiment tracking
- API deployment with FastAPI
- Containerization with Docker
- CI/CD automation with GitHub Actions
- Logging (file + SQLite)
- Monitoring with Prometheus and Grafana
- Automated model retraining trigger on data changes

The system trains multiple models, selects the best, and deploys it as a REST API with real-time monitoring.

---

## 2. Technologies Used
- **Language**: Python 3.11
- **ML & Data Processing**: scikit-learn, pandas, numpy, joblib
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI + Pydantic
- **Logging**: Python logging, SQLite
- **Monitoring**: Prometheus client, Grafana
- **Containerization**: Docker, docker-compose
- **CI/CD**: GitHub Actions
- **Testing**: pytest, FastAPI TestClient
- **Automation**: Retraining workflow
- **Load Testing**: Custom script with `requests`

---

## 3. Workflow

### 3.1 Data Preparation
- **Script**: `src/data.py`
- Loads the Iris dataset from scikit-learn, adds target labels, and saves to `data/iris.csv`.
- Ensures reproducibility for training and testing.

### 3.2 Model Training & Tracking
- **Script**: `src/train.py`
- Trains two models: Logistic Regression and Random Forest.
- Logs metrics and parameters to MLflow (`mlruns/` local store).
- Selects best model based on accuracy and saves it as `models/best_model.pkl`.

### 3.3 API Service
- **File**: `app.py`
- Endpoints:
  - `/predict`: Accepts validated Iris measurements, returns predicted class.
  - `/health`: Simple service status check.
  - `/metrics`: Prometheus metrics export.
- Metrics collected:
  - `predictions_total` (labeled by predicted class)
  - `prediction_latency_seconds` (histogram of inference time)
- Logs predictions to `logs/predictions.log` and `logs/predictions.db`.

### 3.4 Containerization
- **Dockerfile**:
  - Based on `python:3.11-slim`
  - Installs dependencies, copies code & models, sets environment variables
  - Exposes port 8000 and starts API via `uvicorn`
- **docker-compose.yml**:
  - Runs API, Prometheus, and Grafana in a shared network
  - Prometheus scrapes API metrics every 5s
  - Grafana pre-provisioned with Prometheus datasource and Iris dashboard

### 3.5 CI/CD Pipeline
- **Workflow**: `.github/workflows/ci_cd.yml`
  - Runs on push/PR
  - Steps:
    1. Linting with flake8
    2. Data preparation & training
    3. Unit tests (`tests/test_api.py`)
    4. Docker build
    5. Push to Docker Hub (on `dev` branch)
- **Retraining Workflow**: `.github/workflows/retrain.yml`
  - Triggers retraining when dataset or training code changes

### 3.6 Monitoring
- **Prometheus**: Collects request count and latency metrics from API
- **Grafana**: Visualizes total predictions, per-class rate, and latency percentiles
- **Load Generator**: `scripts/load_generator.py` sends random prediction requests to populate metrics

---

## 4. How to Run

### Local (dev mode)
```bash
python src/data.py
python src/train.py
uvicorn app:app --reload --port 8000
