"""Chargement et nettoyage de base de CIC-IDS2017."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def find_csv_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Retourne tous les CSV du dataset, même dans des sous-dossiers."""
    return sorted(raw_dir.rglob("*.csv"))


def load_day(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Charge un CSV et fait le nettoyage minimal.

    - supprime les espaces dans les noms de colonnes (le dataset en contient)
    - remplace inf/-inf par NaN, puis supprime les lignes incomplètes
    - supprime les doublons
    """
    df = pd.read_csv(path, nrows=nrows, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    return df.reset_index(drop=True)
