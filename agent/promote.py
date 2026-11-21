import json
import os
import urllib.request

import mlflow
from mlflow.tracking import MlflowClient

NAME = "churn-model"
SERVING = os.getenv("SERVING_URL", "http://churn-serving.ml-serving")


def reload_serving():
    req = urllib.request.Request(f"{SERVING}/reload", method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    old = str(client.get_model_version_by_alias(NAME, "champion").version)
    versions = [v for v in client.search_model_versions(f"name='{NAME}'")
                if str(v.version) != old]
    if not versions:
        raise SystemExit("REFUS : aucun candidat disponible")
    latest = max(versions, key=lambda v: int(v.version))
    if latest.tags.get("evaluation") != "pass":
        raise SystemExit(f"REFUS : la version {latest.version} n'a pas l'étiquette evaluation=pass")
    cand = str(latest.version)

    client.set_registered_model_alias(NAME, "previous", old)
    client.set_registered_model_alias(NAME, "champion", cand)
    outcome = "promoted"
    try:
        served = str(reload_serving()["model_version"])
        if served != cand:
            raise RuntimeError(f"le serving sert la version {served}, attendu {cand}")
    except Exception as e:
        client.set_registered_model_alias(NAME, "champion", old)
        try:
            reload_serving()
        except Exception:
            pass
        outcome = "rolled_back"
        print(f"ECHEC, retour arrière vers la version {old} : {e}")

    mlflow.set_experiment("churn-agent")
    with mlflow.start_run():
        mlflow.log_params({"action": "promote", "from_version": old,
                           "to_version": cand, "outcome": outcome})
        mlflow.set_tag("severity", "high")
    print("PROMOTION:", outcome, f"(champion {old} -> {cand})")
    if outcome != "promoted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
