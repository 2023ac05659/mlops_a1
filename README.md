
---

## **README.md**
```markdown
Iris Classification Project

Overview
A complete MLOps pipeline for training, deploying, and monitoring an Iris classification model using FastAPI, Docker, Prometheus, and Grafana — with automated CI/CD via GitHub Actions.

---

Features
- Data preparation & versioning
- MLflow experiment tracking
- Best-model selection & saving
- REST API for predictions
- Logging to file & SQLite
- Prometheus metrics & Grafana dashboard
- Dockerized deployment
- GitHub Actions CI/CD
- Automated retraining on data change
- Load generator for demo

---

Tech Stack
- Python 3.11, FastAPI, Pydantic
- scikit-learn, pandas, numpy, joblib
- MLflow
- Prometheus client
- Docker, docker-compose
- Grafana
- pytest, flake8

---

Setup & Run
- Local Development
- python src/data.py
- python src/train.py
- uvicorn app:app --reload --port 8000

---

Test Prediction
- curl -X POST http://localhost:8000/predict \
      -H "content-type: application/json" \
      -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

---

Full Stack with Docker Compose
- Run API + Prometheus + Grafana together:
- python src/data.py && python src/train.py
- docker compose up -d --build

---

Services:
- API → http://localhost:8000
- Prometheus → http://localhost:9090
- Grafana → http://localhost:3000 (admin/admin by default)

---

CI/CD
- ci_cd.yml:
- - Runs on push/PR.
- -  Steps: Lint → Train → Test → Build → Push Docker image.
- retrain.yml:
- - Detects changes in data or training scripts.
- - Retrains model, updates Docker image, redeploys.

---

Monitoring
- /metrics endpoint exposes:
- - predictions_total{class="X"}
- - prediction_latency_seconds
- Prometheus scrapes metrics every 5s.
- Grafana dashboard displays:
- - Total predictions
- - Predictions/sec by class
- - Latency percentiles (p50, p95)
Run load generator for live dashboard updates:
- python scripts/load_generator.py

---


