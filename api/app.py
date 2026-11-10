import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Load the model at startup
# Make sure MLFLOW_TRACKING_URI is set in the environment

MODEL_URI = "models:/poly_model@champion"
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(title="Polynomial Regression API")

class PredictRequest(BaseModel):
    x: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        df = pd.DataFrame([{"x": request.x}])
        prediction = model.predict(df)
        
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))