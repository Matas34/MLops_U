import subprocess

import mlflow
import yaml

with open("models/run_id.txt") as f:
    run_id = f.read().strip()

run = mlflow.MlflowClient().get_run(run_id)

params = {k: int(v) for k, v in run.data.params.items()}

with open("params.yaml", "w") as f:
    yaml.dump(params, f, sort_keys=False)

subprocess.run(["dvc", "repro", "--force"], check=True)