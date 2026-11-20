import os
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from mlflow.tracking import MlflowClient
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)
from pydantic import BaseModel

FEATURES = ["tenure", "monthly_charges", "contract", "support_tickets"]
MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

PREDICTIONS = Counter("churn_predictions_total", "Prédictions par classe", ["predicted_class"])
LATENCY = Histogram("churn_prediction_latency_seconds", "Latence de /predict")
PROBA = Histogram("churn_probability", "Probabilité de churn",
                  buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
TENURE = Histogram("churn_feature_tenure", "tenure reçu",
                   buckets=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72])
MONTHLY = Histogram("churn_feature_monthly_charges", "monthly_charges reçu",
                    buckets=[30, 45, 60, 75, 90, 105, 120, 135, 150])
TICKETS = Histogram("churn_feature_support_tickets", "support_tickets reçu",
                    buckets=[0, 1, 2, 3, 4, 5, 6, 8, 10])
CONTRACT = Counter("churn_feature_contract_total", "contract reçu", ["value"])
MODEL_VERSION = Gauge("churn_model_version", "Version du modèle servi")

app = FastAPI(title="churn-serving")
state = {"model": None, "version": None}


class Customer(BaseModel):
    tenure: int
    monthly_charges: float
    contract: int
    support_tickets: int


class Batch(BaseModel):
    customers: list[Customer]


def load():
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    state["model"], state["version"] = model, mv.version
    MODEL_VERSION.set(int(mv.version))


@app.on_event("startup")
def startup():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        load()
    except Exception as e:
        print(f"chargement du modèle impossible : {e}")


@app.get("/health")
def health():
    if state["model"] is None:
        raise HTTPException(503, "modèle non chargé")
    return {"status": "ok", "model_version": state["version"]}


@app.post("/reload")
def reload():
    load()
    return {"model_version": state["version"]}


@app.post("/predict")
def predict(batch: Batch):
    if state["model"] is None:
        raise HTTPException(503, "modèle non chargé")
    start = time.time()
    df = pd.DataFrame([c.model_dump() for c in batch.customers])[FEATURES]
    proba = state["model"].predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    for _, row in df.iterrows():
        TENURE.observe(row["tenure"])
        MONTHLY.observe(row["monthly_charges"])
        TICKETS.observe(row["support_tickets"])
        CONTRACT.labels(str(int(row["contract"]))).inc()
    for p, c in zip(proba, pred):
        PROBA.observe(float(p))
        PREDICTIONS.labels(str(int(c))).inc()
    LATENCY.observe(time.time() - start)
    return {"model_version": state["version"],
            "churn_probability": [float(p) for p in proba],
            "prediction": [int(c) for c in pred]}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
