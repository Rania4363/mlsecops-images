import json
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from train import FEATURES, make_data

PSI_ALERT = 0.2


def psi(ref, new, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    n = np.histogram(new, edges)[0] / len(new)
    r, n = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


def main():
    n = int(os.getenv("N_SAMPLES", "2000"))
    drift = float(os.getenv("DRIFT", "0.0"))
    concept = float(os.getenv("CONCEPT", "0.0"))
    seed = int(os.getenv("SEED", "123"))

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    mv = client.get_model_version_by_alias("churn-model", "champion")
    ref_path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="data/reference.csv")
    ref = pd.read_csv(ref_path)
    base = client.get_run(mv.run_id).data.metrics
    model = mlflow.sklearn.load_model("models:/churn-model@champion")

    new = make_data(n, seed, drift, concept)

    report = {"champion_version": mv.version, "drift": drift, "concept": concept, "features": {}}
    for f in FEATURES:
        report["features"][f] = {
            "psi": psi(ref[f].to_numpy(), new[f].to_numpy()),
            "ks_pvalue": float(ks_2samp(ref[f], new[f]).pvalue),
        }
    max_psi = max(v["psi"] for v in report["features"].values())

    proba = model.predict_proba(new[FEATURES])[:, 1]
    pred = (proba >= 0.5).astype(int)
    perf = {
        "accuracy": accuracy_score(new["churn"], pred),
        "f1": f1_score(new["churn"], pred),
        "roc_auc": roc_auc_score(new["churn"], proba),
    }
    perf["roc_auc_drop"] = base.get("roc_auc", perf["roc_auc"]) - perf["roc_auc"]
    detected = max_psi > PSI_ALERT
    report.update(max_psi=max_psi, drift_detected=detected, performance=perf)

    mlflow.set_experiment("churn-monitoring")
    with mlflow.start_run():
        mlflow.log_params({"champion_version": mv.version, "drift": drift,
                           "n_samples": n, "psi_threshold": PSI_ALERT})
        mlflow.set_tag("drift_detected", str(detected))
        mlflow.log_param("concept", concept)
        metrics = {"max_psi": max_psi, **perf}
        for f, v in report["features"].items():
            metrics[f"psi_{f}"] = v["psi"]
            metrics[f"ks_pvalue_{f}"] = v["ks_pvalue"]
        mlflow.log_metrics(metrics)
        mlflow.log_dict(report, "report.json")

    print(json.dumps(report, indent=2))
    print("DRIFT DETECTE" if detected else "pas de drift")


if __name__ == "__main__":
    main()
