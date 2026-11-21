import json
import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from train import FEATURES, make_data

NAME = "churn-model"
MIN_AUC_GAIN = float(os.getenv("MIN_AUC_GAIN", "0.01"))


def score(model, df):
    proba = model.predict_proba(df[FEATURES])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(df["churn"], pred)),
        "f1": float(f1_score(df["churn"], pred)),
        "roc_auc": float(roc_auc_score(df["churn"], proba)),
    }


def main():
    drift = float(os.getenv("DRIFT", "0.6"))
    n = int(os.getenv("N_SAMPLES", "3000"))
    seed = int(os.getenv("SEED", "777"))

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    champion = client.get_model_version_by_alias(NAME, "champion")
    versions = client.search_model_versions(f"name='{NAME}'")
    cand = str(max(int(v.version) for v in versions))
    if cand == str(champion.version):
        raise SystemExit("le candidat est déjà le champion : rien à évaluer")

    champ_model = mlflow.sklearn.load_model(f"models:/{NAME}/{champion.version}")
    cand_model = mlflow.sklearn.load_model(f"models:/{NAME}/{cand}")

    df = make_data(n, seed, drift)
    champ_score, cand_score = score(champ_model, df), score(cand_model, df)
    gain = cand_score["roc_auc"] - champ_score["roc_auc"]
    verdict = "pass" if gain >= MIN_AUC_GAIN else "fail"

    report = {
        "champion_version": str(champion.version),
        "candidate_version": cand,
        "eval_drift": drift,
        "eval_samples": n,
        "champion": champ_score,
        "candidate": cand_score,
        "roc_auc_gain": gain,
        "min_auc_gain": MIN_AUC_GAIN,
        "verdict": verdict,
    }

    mlflow.set_experiment("churn-promotion")
    with mlflow.start_run():
        mlflow.log_params({"champion_version": champion.version, "candidate_version": cand,
                           "eval_drift": drift, "eval_samples": n,
                           "min_auc_gain": MIN_AUC_GAIN})
        mlflow.log_metrics({**{f"champion_{k}": v for k, v in champ_score.items()},
                            **{f"candidate_{k}": v for k, v in cand_score.items()},
                            "roc_auc_gain": gain})
        mlflow.set_tag("evaluation", verdict)
        mlflow.log_dict(report, "evaluation.json")
    client.set_model_version_tag(NAME, cand, "evaluation", verdict)

    print(json.dumps(report, indent=2))
    print("EVALUATION:", verdict.upper())


if __name__ == "__main__":
    main()
