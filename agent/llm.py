import json
import os
import time
import urllib.request

BACKEND = os.getenv("LLM_BACKEND", "ollama")  # ollama | none
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama.ml-llm:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))
ALLOWED = {"propose_retrain", "monitor", "no_action"}


def _prompt(obs, decision):
    m = (obs.get("monitoring") or {}).get("metrics", {})
    psi = {k[4:]: round(v, 3) for k, v in m.items() if k.startswith("psi_")}
    faits = {
        "version_champion": obs["champion_version"],
        "psi_par_variable": psi,
        "seuil_psi": float(os.getenv("PSI_ALERT", "0.2")),
        "baisse_roc_auc": round(m.get("roc_auc_drop", 0.0), 3),
        "seuil_baisse_roc_auc": float(os.getenv("AUC_DROP_ALERT", "0.05")),
    }
    return (
        "Tu es un agent MLOps. Voici les mesures du contrôle de dérive d'un modèle de churn :\n"
        f"{json.dumps(faits, ensure_ascii=False)}\n"
        "Choisis UNE action parmi : propose_retrain (réentraîner), monitor (surveiller), "
        "no_action (rien à faire). Réponds uniquement en JSON : "
        '{"action": "...", "explication": "2 phrases maximum", "confiance": 0.0}'
    )


def _ollama(prompt):
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 150},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)["message"]["content"]


BACKENDS = {"ollama": _ollama}


def advise(obs, decision):
    out = {"backend": BACKEND, "model": OLLAMA_MODEL, "ok": False}
    if BACKEND == "none" or BACKEND not in BACKENDS or obs.get("monitoring") is None:
        out["error"] = "désactivé ou pas de données"
        return out
    t0 = time.time()
    try:
        advice = json.loads(BACKENDS[BACKEND](_prompt(obs, decision)))
        if advice.get("action") not in ALLOWED:
            raise ValueError(f"action hors périmètre : {advice.get('action')}")
        out.update(ok=True, advice=advice,
                   agrees=advice["action"] == decision["action"])
    except Exception as e:
        out["error"] = str(e)[:200]
    out["latency_s"] = round(time.time() - t0, 1)
    return out
