import json
import os

import mlflow
from mlflow.tracking import MlflowClient

from llm import advise

MODEL = os.getenv("MODEL_NAME", "churn-model")
ALIAS = os.getenv("MODEL_ALIAS", "champion")
PSI_ALERT = float(os.getenv("PSI_ALERT", "0.2"))
AUC_DROP_ALERT = float(os.getenv("AUC_DROP_ALERT", "0.05"))


def observe(client):
    mv = client.get_model_version_by_alias(MODEL, ALIAS)
    obs = {"champion_version": mv.version, "monitoring": None}
    exp = client.get_experiment_by_name("churn-monitoring")
    if exp is None:
        return obs
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"params.champion_version = '{mv.version}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs:
        r = runs[0]
        obs["monitoring"] = {
            "run_id": r.info.run_id,
            "metrics": dict(r.data.metrics),
            "tags": dict(r.data.tags),
        }
    return obs


def decide(obs):
    m = obs["monitoring"]
    if m is None:
        return {"action": "no_data", "severity": "info",
                "reason": "Aucun rapport de dérive pour le champion actuel : lancer d'abord le contrôle de dérive."}
    metrics = m["metrics"]
    psi_by_feature = {k[4:]: v for k, v in metrics.items() if k.startswith("psi_")}
    drifted = sorted((f for f, v in psi_by_feature.items() if v > PSI_ALERT),
                     key=lambda f: -psi_by_feature[f])
    auc_drop = metrics.get("roc_auc_drop", 0.0)
    detected = m["tags"].get("drift_detected") == "True"
    detail = ", ".join(f"{f} (PSI {psi_by_feature[f]:.2f})" for f in drifted) or "aucune variable"
    if not detected and auc_drop >= AUC_DROP_ALERT:
        return {"action": "propose_retrain", "severity": "high", "drifted_features": drifted,
                "reason": f"Aucune dérive des variables, mais le ROC-AUC du champion a baissé de {auc_drop:.3f} "
                          f"(seuil {AUC_DROP_ALERT}) : dérive de concept probable (la relation entre les "
                          f"variables et le churn a changé). Réentraînement recommandé."}
    if not detected:
        return {"action": "no_action", "severity": "info", "drifted_features": drifted,
                "reason": f"Pas de dérive détectée (PSI max {metrics.get('max_psi', 0):.2f})."}
    if auc_drop >= AUC_DROP_ALERT:
        return {"action": "propose_retrain", "severity": "high", "drifted_features": drifted,
                "reason": f"Dérive sur : {detail}. Le ROC-AUC du champion a baissé de {auc_drop:.3f} "
                          f"(seuil {AUC_DROP_ALERT}). Réentraînement recommandé."}
    return {"action": "monitor", "severity": "medium", "drifted_features": drifted,
            "reason": f"Dérive sur : {detail}, mais la performance est stable "
                      f"(baisse de ROC-AUC {auc_drop:.3f}). Surveillance renforcée."}


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    obs = observe(client)
    decision = decide(obs)
    advice = advise(obs, decision)
    record = {"observation": obs, "decision": decision, "llm": advice}

    mlflow.set_experiment("churn-agent")
    with mlflow.start_run():
        mlflow.log_params({"champion_version": obs["champion_version"],
                           "action": decision["action"]})
        mlflow.set_tag("severity", decision["severity"])
        mlflow.log_params({"llm_ok": advice["ok"],
                           "llm_action": advice.get("advice", {}).get("action", "none"),
                           "llm_agrees": advice.get("agrees", "n/a")})
        mlflow.log_metric("llm_latency_s", advice.get("latency_s", 0))
        if obs["monitoring"]:
            mlflow.log_metrics({k: v for k, v in obs["monitoring"]["metrics"].items()
                                if k in ("max_psi", "roc_auc_drop", "roc_auc")})
        mlflow.log_dict(record, "decision.json")

    print("LLM:", json.dumps(advice, ensure_ascii=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print("DECISION:", decision["action"])


if __name__ == "__main__":
    main()
