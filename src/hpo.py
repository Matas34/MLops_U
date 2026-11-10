import os

import mlflow
import optuna
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from load_params import load_params

os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"
commit = os.environ.get("GIT_COMMIT")

params = load_params()

df = pd.read_csv("data/raw/data.csv")

x, _, y, _ = train_test_split(df[['x']], df['y'], test_size=0.2, random_state=params['seed'])

x_train_raw, x_val_raw, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=params['seed'])

scaler = StandardScaler().fit(x_train_raw)

x_train = scaler.transform(x_train_raw)
x_val = scaler.transform(x_val_raw)

mlflow.set_experiment("polynomial-regression")

with mlflow.start_run(run_name="hpo", log_system_metrics=True) as run:
    if commit:
        mlflow.set_tag("mlflow.source.git.commit", commit)

    mlflow.log_param("degree", params['degree'])
    mlflow.log_param("data_size", params['data_size'])
    mlflow.log_param("seed", params['seed'])

    def objective(trial):
        p = {**params,
             "hidden": trial.suggest_int("hidden", 4, 128),
             "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
             "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
             "epochs": trial.suggest_int("epochs", 5, 100)}
        
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            model = MLPRegressor(
                        hidden_layer_sizes=(p['hidden'], p['hidden']),
                        learning_rate_init=p['lr'], 
                        alpha=p['alpha'],
                        batch_size=32, 
                        random_state=p['seed']
                    )
            
            for epoch in range(p['epochs']):
                model.partial_fit(x_train, y_train)
                mlflow.log_metric("train_mse_loss", mean_squared_error(y_train, model.predict(x_train)), step=epoch)
                mlflow.log_metric("val_mse_loss", mean_squared_error(y_val, model.predict(x_val)), step=epoch)
            
            mlflow.log_params({k: p[k] for k in ("hidden", "lr", "alpha", "epochs")})

            val_mse = mean_squared_error(y_val, model.predict(x_val))
            mlflow.log_metric("val_mse", val_mse)

            val_r2 = r2_score(y_val, model.predict(x_val))
            mlflow.log_metric("val_r2", val_r2)

            return val_mse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=params["seed"])
    )

    study.optimize(objective, n_trials=5)

    mlflow.log_metric("best_mse", study.best_value)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

    params.update(study.best_params)
    with open("best_params.yaml", "w") as f:
        yaml.dump(params, f, sort_keys=False)