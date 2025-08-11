from fastapi.testclient import TestClient
import os, joblib
from app import app

def test_predict_endpoint():
    assert os.path.exists("models/best_model.pkl"), "Train first to create best_model.pkl"
    client = TestClient(app)
    r = client.post("/predict", json={"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2})
    assert r.status_code == 200
    j = r.json()
    assert "prediction" in j
