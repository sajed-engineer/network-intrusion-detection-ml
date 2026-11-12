"""Entraîne le modèle final (XGBoost, jeu d'entraînement COMPLET) et le sauvegarde.

Usage :  python -m src.train_final
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from xgboost import XGBClassifier

from src.evaluate import FULL_PATH, ROOT

MODEL_PATH = ROOT / "models" / "ids_xgboost.joblib"


def main():
    d = joblib.load(FULL_PATH)
    X_train, y_train, X_test, y_test = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    print(f"train: {X_train.shape} | test: {X_test.shape}", flush=True)

    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
        scale_pos_weight=spw, n_jobs=-1, random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("précision:", round(precision_score(y_test, y_pred), 4),
          "| rappel:", round(recall_score(y_test, y_pred), 4),
          "| F1:", round(f1_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attaque"], digits=4))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "scaler": d["scaler"],
        "dropped_constant": d["dropped_constant"],
        "columns": list(X_train.columns),
    }, MODEL_PATH)
    print("modèle sauvegardé :", MODEL_PATH)


if __name__ == "__main__":
    main()
