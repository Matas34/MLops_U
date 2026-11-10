# Polynomial Regression MLOps Pipeline

An end-to-end MLOps workflow: reproducible training with **DVC**, experiment tracking and model registry with **MLflow**, deployment on **Kubernetes (minikube)**, a **FastAPI** inference service, and CI/CD through **GitHub Actions**.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| **Docker** | ≥ 24 | [docs.docker.com/get-docker](https://docs.docker.com/get-docker/) |
| **minikube** | ≥ 1.30 | [minikube.sigs.k8s.io](https://minikube.sigs.k8s.io/docs/start/) |
| **kubectl** | ≥ 1.28 | [kubernetes.io/docs/tasks/tools](https://kubernetes.io/docs/tasks/tools/) |
| **uv** | ≥ 0.4 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Trivy** | ≥ 0.50 | [aquasecurity.github.io/trivy](https://aquasecurity.github.io/trivy/) |


---

## Setup

### 1. Install Python dependencies

```bash
uv sync && source .venv/bin/activate
```

### 2. Configure the DVC remote *(one-time)*

```bash
mkdir -p ~/dvcstore
dvc remote add -d localremote ~/dvcstore
```

### 3. Start the cluster and deploy MLflow

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

**Expected output:**

```text
pod/mlflow-xxxxx   1/1  Running
pvc/mlflow-pvc     Bound
svc/mlflow         NodePort 5000:30500/TCP
```

### 4. Expose MLflow to your host

**Terminal A** *(leave running)*:

```bash
kubectl port-forward -n mlops svc/mlflow 5000:5000
```

Verify:

```bash
curl -s http://127.0.0.1:5000/health
# → OK
```

Then open <http://127.0.0.1:5000> — you should see an empty MLflow UI.

---

## Run the Pipeline Locally

In **Terminal B**:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

echo $MLFLOW_TRACKING_URI   # → http://127.0.0.1:5000

dvc repro --force
dvc push
```

---

## Train in Kubernetes

Run the full training pipeline inside the cluster. Parameters are loaded via a **ConfigMap**, and the job connects to the internal MLflow service.

```bash
# Apply the ConfigMap (parameters) and the Job
kubectl apply -f k8s/train-config.yaml
kubectl apply -f k8s/train-job.yaml

# Watch the logs
kubectl logs -f -l job-name=train-job -n mlops
```

Once the Job status is **Complete**, a new run with a `@champion` model alias will be available in the MLflow UI.

---

## Deploy the Inference API

The FastAPI service dynamically loads the model from the MLflow Registry using the `@champion` alias.

```bash
# Deploy the API and its Service
kubectl apply -f k8s/inference.yaml
kubectl apply -f k8s/inference-service.yaml

# Verify the pod is running
kubectl get pods -l app=model-api -n mlops
```

Expose the API to your host in **Terminal C** *(leave running)*:

```bash
kubectl port-forward -n mlops svc/model-api-service 8080:80
```

### Test the API

**Health check**

```bash
curl http://localhost:8080/health
# → {"status":"ok"}
```

**Predict**

```bash
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"x": 5.0}'
# → {"prediction": <value>}
```

> Interactive Swagger UI is available at <http://localhost:8080/docs>.

---

## Load Test

Verify the non-functional requirements (latency and throughput) documented in [`docs/requirements.md`](docs/requirements.md):

```bash
python scripts/load_test.py
```

**Expected output:**

```text
--- Results ---
Total Successful Requests: 100
Average Latency: 27.46 ms
p95 Latency: 36.77 ms
Max Latency: 38.85 ms
Throughput: 344.68 requests/sec
```

---

## CI/CD & Rollback

The repository uses **GitHub Actions** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) to automate the pipeline:

| Stage | What it does |
|-------|--------------|
| **Lint & Test** | Runs `ruff` and `pytest` |
| **Train & Gate** | Runs `dvc repro` and checks the R² metric against the **0.9** threshold in `evaluate.py` |
| **Build & Push** | Builds the Training and API Docker images and pushes them to GHCR |
| **Update Manifest** | Updates the image tag in `k8s/inference.yaml` and commits it back to the repo |

---

## Reproduce a Historical Run

Uses the `run_id` stored in `models/run_id.txt`:

```bash
python src/reproduce.py
```

---

## View Results

Open <http://127.0.0.1:5000> and select the **`polynomial-regression`** experiment. Each run shows its parameters, metrics (**MSE**, **R²**), and the model artifact.

---

## Tear Down

```bash
# Press Ctrl+C in the port-forward terminals (Terminal A and C), then:
minikube delete
```