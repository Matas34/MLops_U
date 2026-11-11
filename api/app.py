import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel

MODEL_NAME = "poly_model"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

model = mlflow.pyfunc.load_model(MODEL_URI)
version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version

app = FastAPI(title="Polynomial Regression API")

class PredictRequest(BaseModel):
    x: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-version")
def model_version():
    return {
        "version": version
    }

@app.post("/predict")
def predict(request: PredictRequest):
    df = pd.DataFrame([{"x": request.x}])
    prediction = model.predict(df)
    
    return {"prediction": float(prediction[0])}