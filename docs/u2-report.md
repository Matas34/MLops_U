# U2 Report

## 1.1 Architecture

```text
params.yaml
    │
    ▼
┌─────────────┐   data/raw/data.csv   ┌──────────┐   best_params.yaml   ┌───────────┐
│  prepare    │ ────────────────────► │   hpo    │ ───────────────────► │   train   │
│ (synthetic) │                       │ (Optuna) │                      │ (sklearn) │
└─────────────┘                       └──────────┘                      └─────┬─────┘
                                                                              │ model.pkl
                                                                              │ run_id.txt
                                                                              ▼
                                                                        ┌───────────┐
                                                                        │ evaluate  │
                                                                        │ R² gate   │
                                                                        └─────┬─────┘
                                                                              │ register + alias
                                                                              ▼
                                                                    ┌──────────────────┐
                                                                    │ MLflow Registry  │
                                                                    │ poly_model       │
                                                                    │  @champion       │
                                                                    └────────┬─────────┘
                                                                             │ load at startup
                                                                             ▼
                                                                    ┌──────────────────┐
                                                                    │ FastAPI model-api│
                                                                    │ /predict         │
                                                                    │ /model-version   │
                                                                    └──────────────────┘
```


## 2.2 Measurement

```text
(u1) matas@DESKTOP-8HUKPGA:~/u1$ python scripts/load_test.py
--- Results ---
Total Successful Requests: 100
Average Latency: 27.17 ms
p95 Latency: 61.13 ms
Max Latency: 77.10 ms
Throughput: 347.33 requests/sec
(u1) matas@DESKTOP-8HUKPGA:~/u1$ python scripts/load_test.py
--- Results ---
Total Successful Requests: 100
Average Latency: 23.33 ms
p95 Latency: 28.32 ms
Max Latency: 29.87 ms
Throughput: 401.01 requests/sec
(u1) matas@DESKTOP-8HUKPGA:~/u1$ python scripts/load_test.py
--- Results ---
Total Successful Requests: 100
Average Latency: 26.20 ms
p95 Latency: 34.33 ms
Max Latency: 37.05 ms
Throughput: 353.81 requests/sec
(u1) matas@DESKTOP-8HUKPGA:~/u1$ 
```

## 3.2 Rollback proof

```text
(u1) matas@DESKTOP-8HUKPGA:~/u1$ export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
(u1) matas@DESKTOP-8HUKPGA:~/u1$ curl -s -w "\n" localhost:8080/model-version
{"version":"11"}
(u1) matas@DESKTOP-8HUKPGA:~/u1$ python -c "from mlflow import MlflowClient; MlflowClient().set_registered_model_alias('poly_model', 'champion', 10)"
(u1) matas@DESKTOP-8HUKPGA:~/u1$ curl -s -w "\n" localhost:8080/model-version
{"version":"11"}
(u1) matas@DESKTOP-8HUKPGA:~/u1$ kubectl rollout restart deployment/model-api -n mlops
deployment.apps/model-api restarted
(u1) matas@DESKTOP-8HUKPGA:~/u1$ curl -s -w "\n" localhost:8080/model-version
{"version":"10"}
(u1) matas@DESKTOP-8HUKPGA:~/u1$
```

Step by step:

1. Baseline: pod serves version 11.
2. Alias moved from 11 to 10 in the registry.
3. Without a restart the same pod still serves 11 — a running pod keeps the model it loaded.
4. After `kubectl rollout restart`, the new pod resolves `champion` and serves version 10.
