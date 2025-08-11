from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging
logging.basicConfig(filename='logs/predictions.log', level=logging.INFO)

app = FastAPI()
model = joblib.load("models/RandomForestClassifier.pkl")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisInput):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    pred = model.predict(features)
    logging.info(f"Input: {data.dict()}, Prediction: {pred[0]}")
    return {"prediction": int(pred[0])}
