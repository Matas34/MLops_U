import json
import sys

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from check_quality_gate import check_quality_gate

model = joblib.load("models/model.pkl")
df = pd.read_csv("data/raw/data.csv")

with open("best_params.yaml") as f:
    params = yaml.safe_load(f)

_, x_test, _, y_test = train_test_split(df[['x']], df['y'], test_size=0.2, random_state=params['seed'])

preds = model.predict(x_test)

r2 = r2_score(y_test, preds)

metrics = {
    "MSE": mean_squared_error(y_test, preds),
    "R2": r2
}

threshold = 0.9
gate_passed = check_quality_gate(r2, threshold)

with open("models/run_id.txt", "r") as f:
    run_id = f.read().strip()

with open("models/model_uri.txt", "r") as f:
    model_uri = f.read().strip()
    
client = mlflow.MlflowClient()

if gate_passed:
    mv = mlflow.register_model(model_uri, "poly_model")
    client.set_registered_model_alias("poly_model", "champion", mv.version)

for key, value in metrics.items():
    client.log_metric(run_id, key, value)

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

if not gate_passed:
    sys.exit(f"Model quality gate failed to pass: {r2:.2f} < {threshold}")