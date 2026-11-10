import subprocess

import mlflow
import yaml

with open("models/run_id.txt") as f:
    run_id = f.read().strip()

run = mlflow.MlflowClient().get_run(run_id)

def parse_param(v):
    try:
        return float(v) if '.' in v else int(v)
    except ValueError:
        return v

params = {k: parse_param(v) for k, v in run.data.params.items()}

with open("best_params.yaml", "w") as f:
    yaml.dump(params, f, sort_keys=False)

subprocess.run(["python", "src/train.py"], check=True)
subprocess.run(["python", "src/evaluate.py"], check=True)