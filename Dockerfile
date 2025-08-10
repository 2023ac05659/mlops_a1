FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py
COPY src ./src
COPY README.md ./README.md
COPY SUMMARY.md ./SUMMARY.md

RUN mkdir -p models logs
COPY models/RandomForestClassifier.pkl ./models/RandomForestClassifier.pkl

ENV MODEL_SOURCE=local
ENV BEST_MODEL_PATH=models/RandomForestClassifier.pkl
ENV PRED_DB_PATH=logs/predictions.db

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
