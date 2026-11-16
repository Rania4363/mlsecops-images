ARG MLFLOW_VERSION=3.18.0
FROM ghcr.io/mlflow/mlflow:v${MLFLOW_VERSION}
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
