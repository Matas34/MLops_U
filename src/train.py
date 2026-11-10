import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

with open("best_params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/raw/data.csv")

x, _, y, _ = train_test_split(df[['x']], df['y'], test_size=0.2, random_state=params['seed'])

x_train_raw, x_val_raw, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=params['seed'])

scaler = StandardScaler().fit(x_train_raw)

x_train = scaler.transform(x_train_raw)
x_val = scaler.transform(x_val_raw)

model = MLPRegressor(
    hidden_layer_sizes=(params['hidden'], params['hidden']),
    learning_rate_init=params['lr'], 
    alpha=params['alpha'],
    batch_size=32, 
    random_state=params['seed']
)

mlflow.set_experiment("polynomial-regression")
os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"
commit = os.environ.get("GIT_COMMIT")

with mlflow.start_run(log_system_metrics=True) as run:
    if commit:
        mlflow.set_tag("mlflow.source.git.commit", commit)

    for epoch in range(params['epochs']):
        model.partial_fit(x_train, y_train)
        mlflow.log_metric("train_mse_loss", mean_squared_error(y_train, model.predict(x_train)), step=epoch)
        mlflow.log_metric("val_mse_loss", mean_squared_error(y_val, model.predict(x_val)), step=epoch)

    mlflow.log_param("degree", params['degree'])
    mlflow.log_param("data_size", params['data_size'])
    mlflow.log_param("seed", params['seed'])
    mlflow.log_param("hidden", params['hidden'])
    mlflow.log_param("lr", params['lr'])
    mlflow.log_param("alpha", params['alpha'])
    mlflow.log_param("epochs", params['epochs'])

    train_ds = mlflow.data.from_pandas(
        x_train_raw.assign(y=y_train),
        name="train", 
        targets="y"
    )

    val_ds = mlflow.data.from_pandas(
        x_val_raw.assign(y=y_val),
        name="val", 
        targets="y"
    )

    mlflow.log_input(train_ds, context="training")
    mlflow.log_input(val_ds, context="validation")

    ppl = Pipeline([("scaler", scaler), ("model", model)])

    model_info = mlflow.sklearn.log_model(ppl, name="model",
        signature=infer_signature(x.iloc[:5], ppl.predict(x.iloc[:5])),
        input_example=x.iloc[:5],
        skops_trusted_types=["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"])

    os.makedirs("models", exist_ok=True)
    joblib.dump(ppl, "models/model.pkl")
    
    with open("models/run_id.txt", "w") as f:
        f.write(run.info.run_id)

    with open("models/model_uri.txt", "w") as f:
        f.write(model_info.model_uri)