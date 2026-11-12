"""Prétraitement de CIC-IDS2017 : nettoyage, split, normalisation."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.load_data import find_csv_files, load_day

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Colonnes où une valeur négative est NORMALE (-1 = "pas de fenêtre TCP", etc.)
NEGATIVE_OK = {"Init_Win_bytes_forward", "Init_Win_bytes_backward", "min_seg_size_forward"}


def load_all(nrows_per_file: int | None = None) -> pd.DataFrame:
    """Charge les 8 fichiers, les concatène (float32 pour économiser la RAM)."""
    frames = []
    for f in find_csv_files():
        df = load_day(f, nrows=nrows_per_file)
        num_cols = df.select_dtypes("number").columns
        df[num_cols] = df[num_cols].astype("float32")
        frames.append(df)
        print(f"{f.name:55s} {len(df):>8,d} lignes")
    full = pd.concat(frames, ignore_index=True).drop_duplicates()
    return full.reset_index(drop=True)


def fix_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace par 0 les valeurs négatives impossibles (débits, durées...)."""
    num_cols = df.select_dtypes("number").columns
    neg_cols = [c for c in num_cols if (df[c] < 0).any() and c not in NEGATIVE_OK]
    for c in neg_cols:
        n = int((df[c] < 0).sum())
        print(f"  {c}: {n} valeurs négatives -> 0")
        df[c] = df[c].clip(lower=0)
    return df


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """y_binary : 0 = BENIGN, 1 = attaque. On garde 'Label' pour le multi-classe."""
    df["y_binary"] = (df["Label"] != "BENIGN").astype("int8")
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split stratifié PUIS normalisation (fit sur train seulement : pas de fuite)."""
    X = df.drop(columns=["Label", "y_binary"])
    y = df["y_binary"]
    y_multi = df["Label"]

    X_train, X_test, y_train, y_test, ym_train, ym_test = train_test_split(
        X, y, y_multi, test_size=test_size, stratify=y, random_state=seed
    )

    # Colonnes constantes (variance nulle sur le train) : inutiles
    constant = [c for c in X_train.columns if X_train[c].nunique() <= 1]
    X_train = X_train.drop(columns=constant)
    X_test = X_test.drop(columns=constant)

    del X, y, y_multi  # libère la RAM

    scaler = StandardScaler().fit(X_train)  # fit sur train uniquement
    cols = X_train.columns
    X_train_s = pd.DataFrame(scaler.transform(X_train).astype("float32"), columns=cols)
    del X_train
    X_test_s = pd.DataFrame(scaler.transform(X_test).astype("float32"), columns=cols)
    del X_test

    return {
        "X_train": X_train_s, "X_test": X_test_s,
        "y_train": y_train.reset_index(drop=True), "y_test": y_test.reset_index(drop=True),
        "ym_train": ym_train.reset_index(drop=True), "ym_test": ym_test.reset_index(drop=True),
        "scaler": scaler, "dropped_constant": constant,
    }


def main():
    """Pipeline complet sur les 8 fichiers -> data/processed/dataset_full.joblib"""
    df = load_all(nrows_per_file=None)
    df = fix_negative_values(df)
    df = add_binary_label(df)
    data = split_and_scale(df)
    del df
    print("train:", data["X_train"].shape, "| test:", data["X_test"].shape)
    print("sauvegardé :", save(data, name="dataset_full.joblib"))


def save(data: dict, out_dir: Path = PROCESSED_DIR, name: str = "dataset.joblib") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    joblib.dump(data, path)
    return path


if __name__ == "__main__":
    main()
