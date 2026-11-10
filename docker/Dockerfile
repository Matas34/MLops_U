FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "joblib==1.6.0" \
    "mlflow==3.17.1" \
    "numpy==2.5.4" \
    "pandas==3.0.6" \
    "pyyaml==6.0.3" \
    "scikit-learn==1.9.1" \
    "skops==0.16.0" \
    "psutil==7.2.2"\
    "optuna==5.0.0"

COPY src/ /app/src/

# dont store compiled bytecode (pycache)
ENV PYTHONDONTWRITEBYTECODE=1

ARG GIT_COMMIT=""
ENV GIT_COMMIT=$GIT_COMMIT

CMD ["sh", "-c", "python src/prepare.py && python src/hpo.py && python src/train.py && python src/evaluate.py"]