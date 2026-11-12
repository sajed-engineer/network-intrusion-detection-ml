"""Calcule les performances du MODÈLE FINAL sur le jeu de test et les écrit dans
results/final_model_metrics.json (utilisé par l'interface).

Usage :  python -m src.export_metrics
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.evaluate import FULL_PATH, ROOT
from src.train_final import MODEL_PATH


def main():
    d = joblib.load(FULL_PATH)
    bundle = joblib.load(MODEL_PATH)
    X_test, y_test, ym_test = d["X_test"], d["y_test"], d["ym_test"]

    proba = bundle["model"].predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    per_class = {}
    for label, grp in ym_test.groupby(ym_test):
        p = y_pred[grp.index.to_numpy()]
        flagged = float((p == 1).mean())
        per_class[label] = {"n": int(len(grp)), "flagged_pct": round(100 * flagged, 2)}

    out = {
        "n_test": int(len(y_test)),
        "n_attacks": int(y_test.sum()),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "false_alarm_pct": round(100 * fp / (fp + tn), 3),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_class": per_class,
    }
    path = ROOT / "results" / "final_model_metrics.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
