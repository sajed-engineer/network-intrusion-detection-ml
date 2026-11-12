"""Démo de détection « en temps réel » : rejoue des flux réseau un par un.

Le modèle travaille sur les 70 features calculées par CICFlowMeter (durée du flux,
tailles de paquets, délais...). Cette démo REJOUE des flux d'un fichier CSV du dataset
à cadence réglable et affiche une alerte pour chaque flux jugé malveillant.

Pour analyser du vrai trafic : capturer un .pcap (Wireshark / Scapy), le convertir en
flux avec CICFlowMeter, puis passer le CSV obtenu à ce script.

Usage :
    python -m src.live_detect --csv data/raw/archive/Wednesday-workingHours.pcap_ISCX.csv --n 30
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluate import ROOT
from src.preprocess import NEGATIVE_OK
from src.train_final import MODEL_PATH

RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"


def prepare(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Applique EXACTEMENT le même prétraitement qu'à l'entraînement."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    X = df.drop(columns=["Label"], errors="ignore").drop(columns=bundle["dropped_constant"], errors="ignore")
    X = X[bundle["scaler"].feature_names_in_]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float64")
    for c in X.columns:
        if c not in NEGATIVE_OK:
            X[c] = X[c].clip(lower=0)
    return pd.DataFrame(bundle["scaler"].transform(X), columns=X.columns)


def main():
    ap = argparse.ArgumentParser(description="Démo IDS : rejeu de flux réseau")
    ap.add_argument("--csv", required=True, help="CSV de flux (format CICIDS2017)")
    ap.add_argument("--n", type=int, default=30, help="nombre de flux à rejouer")
    ap.add_argument("--delay", type=float, default=0.3, help="pause entre flux (s)")
    ap.add_argument("--threshold", type=float, default=0.5, help="seuil d'alerte (0-1)")
    ap.add_argument("--attack-share", type=float, default=0.3, help="part d'attaques dans la démo")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    bundle = joblib.load(MODEL_PATH)
    raw = pd.read_csv(args.csv, encoding="latin-1", low_memory=False)
    raw.columns = raw.columns.str.strip()

    # Échantillon de démo : mélange de trafic normal et d'attaques
    has_label = "Label" in raw.columns
    rng = np.random.default_rng(args.seed)
    if has_label:
        att = raw[raw["Label"] != "BENIGN"]
        ben = raw[raw["Label"] == "BENIGN"]
        n_att = min(int(args.n * args.attack_share), len(att))
        demo = pd.concat([att.sample(n_att, random_state=args.seed),
                          ben.sample(args.n - n_att, random_state=args.seed)])
        demo = demo.sample(frac=1, random_state=args.seed)
    else:
        demo = raw.sample(min(args.n, len(raw)), random_state=args.seed)

    X = prepare(demo, bundle)
    proba = bundle["model"].predict_proba(X)[:, 1]

    print(f"\nIDS en écoute... ({len(demo)} flux, seuil = {args.threshold})\n")
    correct = 0
    for i, (p, (_, row)) in enumerate(zip(proba, demo.iterrows()), 1):
        alert = p >= args.threshold
        port = int(row.get("Destination Port", -1))
        dur = float(row.get("Flow Duration", 0)) / 1e6
        verdict = f"{RED}ALERTE  {RESET}" if alert else f"{GREEN}normal  {RESET}"
        line = f"[{i:03d}] port {port:>5d} | durée {dur:8.3f}s | score {p:5.3f} | {verdict}"
        if has_label:
            truth = row["Label"]
            ok = alert == (truth != "BENIGN")
            correct += ok
            line += f"| réel : {truth:<18s} {'✓' if ok else '✗ ERREUR'}"
        print(line)
        time.sleep(args.delay)

    if has_label:
        print(f"\nRésultat : {correct}/{len(demo)} flux correctement classés")


if __name__ == "__main__":
    main()
