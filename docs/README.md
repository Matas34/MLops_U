## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Docker | ≥ 24 | https://docs.docker.com/get-docker/ |
| minikube | ≥ 1.30 | https://minikube.sigs.k8s.io/docs/start/ |
| kubectl | ≥ 1.28 | https://kubernetes.io/docs/tasks/tools/ |
| uv | ≥ 0.4 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Trivy | ≥ 0.50 | https://aquasecurity.github.io/trivy/ |

**Resources:** 4 CPUs, 8 GB RAM free for minikube.

---

## 1. Install python dependencies and enable venv

```bash
uv sync && source .venv/bin/activate

```

## 2. Configure DVC remote (one-time)

```bash
mkdir -p ~/dvcstore
dvc remote add -d localremote ~/dvcstore
```

## 3. Start cluster and deploy MLflow

```bash
minikube start --cpus=4 --memory=8192

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/mlflow-pvc.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/mlflow-service.yaml

kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/mlflow-pvc -n mlops --timeout=120s
kubectl wait --for=condition=Ready pod -n mlops -l app=mlflow --timeout=180s

kubectl get pods,pvc,svc -n mlops
```

Expected:
```
pod/mlflow-xxxxx   1/1  Running
pvc/mlflow-pvc     Bound
svc/mlflow         NodePort 5000:30500/TCP
```

## 4. Expose MLflow to host

**Terminal A** (leave running):

```bash
kubectl port-forward -n mlops svc/mlflow 5000:5000
```

Verify:

```bash
curl -s http://127.0.0.1:5000/health
# → OK
```

Open **http://127.0.0.1:5000** — empty MLflow UI.

## 5. Run pipeline

**Terminal B:**

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

echo $MLFLOW_TRACKING_URI   # → http://127.0.0.1:5000

dvc repro --force
dvc push
```

## 6. View results

Open **http://127.0.0.1:5000** → **quadratic-regression** experiment. Each run shows params (`degree`, `data_size`, `seed`), metrics (`MSE`, `R2`), and the model artifact.

## 7. Reproduce a historical run

```bash
# Takes the run_id from models/run_id.txt
python src/reproduce.py
```

## 8. Tear down

```bash
# Ctrl+C in the port-forward terminal
minikube delete
```