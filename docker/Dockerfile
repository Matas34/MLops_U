FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "joblib==1.6.0" \
    "mlflow==3.17.1" \
    "numpy==2.5.4" \
    "pandas==3.0.6" \
    "pyyaml==6.0.3"

COPY src/prepare.py src/train.py /app/src/
COPY params.yaml /app/params.yaml

# dont store compiled bytecode (pycache)
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["sh", "-c", "python src/prepare.py && python src/train.py"]