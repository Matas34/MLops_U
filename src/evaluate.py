import joblib, json, pandas as pd, yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from sklearn.preprocessing import PolynomialFeatures

model = joblib.load("models/model.pkl")
df = pd.read_csv("data/raw/data.csv")

with open("params.yaml") as f:
    params = yaml.safe_load(f)

_, x_test, _, y_test = train_test_split(df[['x']], df['y'], test_size=0.2, random_state=params['seed'])

pf = PolynomialFeatures(degree=params['degree'])

x_test = pf.fit_transform(x_test)

preds = model.predict(x_test)

metrics = {
    "MSE": mean_squared_error(y_test, preds),
    "R2": r2_score(y_test, preds)
}

with open("models/run_id.txt", "r") as f:
    run_id = f.read().strip()
    client = mlflow.MlflowClient()
    for key, value in metrics.items():
        client.log_metric(run_id, key, value)

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)