"""Entraînement et évaluation des modèles IDS."""
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_PATH = ROOT / "data" / "processed" / "dataset.joblib"
FULL_PATH = ROOT / "data" / "processed" / "dataset_full.joblib"


def load_data(train_sample: int | None = None, seed: int = 42, path: Path = DATA_PATH):
    """Charge le jeu prétraité. Sous-échantillonne le TRAIN (stratifié) si demandé ;
    le test reste complet pour une évaluation honnête."""
    d = joblib.load(path)
    X_train, y_train = d["X_train"], d["y_train"]
    if train_sample and train_sample < len(X_train):
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=train_sample, stratify=y_train, random_state=seed
        )
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True), d["X_test"], d["y_test"]


def load_test_labels(path: Path = DATA_PATH) -> pd.Series:
    """Étiquettes multi-classes du jeu de test (pour l'analyse par attaque)."""
    return joblib.load(path)["ym_test"]


def per_class_report(y_multi_test: pd.Series, preds: dict) -> pd.DataFrame:
    """Pour chaque type de trafic : nombre de flux et part correctement classée.

    Pour BENIGN, c'est la part correctement classée comme normale (spécificité) ;
    pour une attaque, c'est le rappel (part détectée comme attaque).
    """
    rows = []
    for label, grp in y_multi_test.groupby(y_multi_test):
        row = {"classe": label, "n_test": len(grp)}
        for name, y_pred in preds.items():
            p = pd.Series(y_pred, index=y_multi_test.index).loc[grp.index]
            row[name] = float((p == 0).mean() if label == "BENIGN" else (p == 1).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("classe").round(4).sort_values("n_test", ascending=False)


def make_models(scale_pos_weight: float, seed: int = 42):
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, class_weight="balanced", random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1, tree_method="hist",
            scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=seed,
            eval_metric="logloss"),
    }


def metrics(y_true, y_pred, y_score=None) -> dict:
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    return out


def run_all(X_train, y_train, X_test, y_test, seed: int = 42):
    """Entraîne RF, XGBoost (supervisés) et Isolation Forest (non supervisé)."""
    rows, preds, fitted = [], {}, {}
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    for name, model in make_models(spw, seed).items():
        t = time.time()
        model.fit(X_train, y_train)
        fit_s = time.time() - t
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        rows.append({"modèle": name, **metrics(y_test, y_pred, y_score), "temps_entraînement_s": round(fit_s, 1)})
        preds[name], fitted[name] = y_pred, model

    # Isolation Forest : entraîné UNIQUEMENT sur du trafic normal (détection d'anomalies)
    t = time.time()
    iso = IsolationForest(n_estimators=200, contamination=0.02, n_jobs=-1, random_state=seed)
    iso.fit(X_train[y_train == 0])
    fit_s = time.time() - t
    y_pred = (iso.predict(X_test) == -1).astype(int)  # -1 = anomalie = attaque
    y_score = -iso.score_samples(X_test)
    rows.append({"modèle": "Isolation Forest", **metrics(y_test, y_pred, y_score), "temps_entraînement_s": round(fit_s, 1)})
    preds["Isolation Forest"], fitted["Isolation Forest"] = y_pred, iso

    table = pd.DataFrame(rows).set_index("modèle").round(4)
    return table, preds, fitted


def save_confusion_matrices(y_test, preds: dict, out: Path = RESULTS_DIR / "confusion_matrices.png"):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4))
    for ax, (name, y_pred) in zip(np.atleast_1d(axes), preds.items()):
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Normal", "Attaque"]).plot(
            ax=ax, colorbar=False, values_format="d")
        ax.set_title(name)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    return fig


def save_feature_importance(model, columns, top: int = 15, out: Path = RESULTS_DIR / "feature_importance.png"):
    imp = pd.Series(model.feature_importances_, index=columns).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, title=f"Top {top} features (Random Forest)")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    return imp
