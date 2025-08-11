# app.py
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
import os, time, uuid, logging
import joblib
import numpy as np
from src.db import get_conn, init_db, log_prediction
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/predictions.log", level=logging.INFO)

PREDICTIONS_TOTAL = Counter("predictions_total", "Total number of predictions", ["class"])
PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction requests",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

app = FastAPI(title="Iris Classifier API", version="1.0")

MODEL_PATH = os.getenv("BEST_MODEL_PATH", "models/best_model.pkl")
model = joblib.load(MODEL_PATH)

DB_CONN = get_conn()
init_db(DB_CONN)

class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0)
    sepal_width:  float = Field(..., ge=0)
    petal_length: float = Field(..., ge=0)
    petal_width:  float = Field(..., ge=0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: IrisInput):
    t0 = time.perf_counter()
    x = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    pred = model.predict(x)
    latency_s = time.perf_counter() - t0

    class_id = int(pred[0])
    # metrics
    PREDICTIONS_TOTAL.labels(str(class_id)).inc()
    PREDICTION_LATENCY_SECONDS.observe(latency_s)

    logging.info(
        f"Input: {data.model_dump()}, Prediction: {class_id}, latency_ms={latency_s*1000:.2f}"
    )
    log_prediction(DB_CONN, str(uuid.uuid4()), data.model_dump(), class_id, latency_s * 1000.0)

    return {"prediction": class_id}

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
