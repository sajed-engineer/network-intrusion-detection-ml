# Détection d'intrusions réseau par Machine Learning (CIC-IDS2017)

Système de détection d'intrusions (IDS) basé sur le machine learning : il classe des flux réseau en **trafic normal** ou **attaque**. Trois approches sont comparées (Random Forest, XGBoost, Isolation Forest) sur le dataset CIC-IDS2017, avec une évaluation **attaque par attaque**.

## Résultats

Jeu de test : **499 596 flux** (17 % d'attaques), split aléatoire stratifié 80/20 sur 2 497 980 flux (15 classes).

| Modèle | Précision | Rappel | F1 | Entraînement |
|---|---|---|---|---|
| Random Forest (400 k flux) | 0,995 | 0,997 | 0,996 | 28 s |
| XGBoost (400 k flux) | 0,995 | 0,999 | 0,997 | 10 s |
| **XGBoost final (2 M flux)** | **0,995** | **0,9996** | **0,9974** | ~1-2 min |
| Isolation Forest (non supervisé) | 0,799 | 0,388 | 0,522 | 3 s |

Modèle final : **99,96 % des attaques détectées**, **0,10 %** de fausses alertes sur le trafic normal.

### Détection par type de trafic (XGBoost, 400 k flux d'entraînement)

| Classe | Flux de test | Détectés |
|---|---|---|
| DoS Hulk / DDoS / PortScan | 18 000 à 35 000 | > 99,9 % |
| DoS GoldenEye / slowloris / Slowhttptest | 1 000 à 2 000 | > 99,2 % |
| FTP-Patator / SSH-Patator | 642 à 1 148 | > 99,5 % |
| Web Attack Brute Force / XSS | 317 / 122 | 99,1 % / 95,1 % |
| Bot | 375 | 93,1 % |
| **Infiltration** | 6 | **16,7 %** (3 sur 6 avec le modèle final : trop peu pour conclure) |
| SQL Injection / Heartbleed | 4 / 2 | non significatif (trop peu d'exemples) |

## Interface : NetGuard IDS

Tableau de bord Streamlit (thème sombre) construit sur le modèle final :

- **Analyse de flux** : choisis un fichier du dataset (ou importe un CSV au format CICFlowMeter), le modèle score chaque flux ; indicateurs, distribution des scores, détection par type de trafic, ports visés, variables influentes, tableau des flux signalés, export CSV.
- **Rejeu en direct** : les flux défilent un par un avec leur score de risque et des alertes.
- **Performances** : résultats du modèle final sur le jeu de test (précision, rappel, F1, matrice de confusion, détection par attaque) et comparaison des 3 approches.
- **À propos** : méthode et limites.

```bash
streamlit run app.py        # puis ouvrir http://localhost:8501
```

> L'interface **rejoue des flux du dataset** : elle ne capture pas le réseau en direct, et les flux affichés ont pu servir à l'entraînement. Les performances de référence sont celles de l'onglet *Performances*.

## Pipeline

```
CSV bruts (8 fichiers)  →  nettoyage  →  split stratifié 80/20  →  normalisation  →  modèles  →  évaluation
                          (inf/NaN,       (avant la              (fit sur le
                          doublons,        normalisation)         train seulement)
                          valeurs < 0)
```

1. **Exploration** (`01_eda.ipynb`) : classes, statistiques, corrélations. Découverte de valeurs négatives impossibles (durées, débits), un défaut connu du dataset.
2. **Prétraitement** (`02_preprocessing.ipynb`) : suppression des `inf`/`NaN` et des doublons, correction des valeurs négatives, suppression de 10 colonnes constantes (il reste 70 features), split stratifié puis normalisation ajustée **sur le train uniquement** (pas de fuite de données).
3. **Modèles** (`03_models.ipynb`) : comparaison Random Forest / XGBoost / Isolation Forest.
4. **Évaluation complète** (`04_full_evaluation.ipynb`) : 8 fichiers, rappel par classe, matrices de confusion.
5. **Démo** (`src/live_detect.py`) : rejeu de flux avec alertes.

## Limites (à lire)

- **Split aléatoire.** Des flux d'une même attaque se retrouvent dans le train et le test. Un IDS réel doit détecter des attaques de jours futurs : une évaluation par jour serait plus exigeante, et les scores seraient plus bas.
- **Classes très rares.** Infiltration (36 flux au total), SQL Injection (21) et Heartbleed (11) ont trop peu d'exemples pour conclure. Infiltration est la vraie faiblesse du modèle (17 % détecté).
- **Les comparaisons Random Forest / XGBoost / Isolation Forest utilisent 400 000 flux d'entraînement**, pour limiter la mémoire. Seul le modèle final utilise l'ensemble du jeu d'entraînement.
- **La démo rejoue des flux du dataset**, elle ne capture pas le réseau en direct. Le modèle attend les 70 features de CICFlowMeter. Pour du vrai trafic : capturer un `.pcap`, le convertir avec CICFlowMeter, puis passer le CSV à la démo.
- Les flux de la démo peuvent avoir servi à l'entraînement : c'est une illustration, pas une mesure de performance.

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

Placer les CSV de CIC-IDS2017 (`MachineLearningCSV`) dans `data/raw/`
(source : https://www.unb.ca/cic/datasets/ids-2017.html, ou Kaggle).

## Utilisation

```bash
# 1. Prétraitement complet (nécessite ~4 Go de RAM libre)
python -m src.preprocess
# 2. Entraîner et sauvegarder le modèle final
python -m src.train_final

# 3. Calculer les métriques du modèle final (utilisées par l'interface)
python -m src.export_metrics

# 4. Lancer l'interface
streamlit run app.py

# (ou la démo en ligne de commande)
python -m src.live_detect --csv "data/raw/archive/Wednesday-workingHours.pcap_ISCX.csv" --n 30 --delay 0.3
```

## Structure

```
data/raw/          CSV du dataset (non versionnés)
data/processed/    jeux prétraités (non versionnés)
models/            modèle final (ids_xgboost.joblib)
notebooks/         01 exploration, 02 prétraitement, 03 modèles, 04 évaluation complète
src/               load_data, preprocess, evaluate, train_final, export_metrics, live_detect
app.py             interface NetGuard IDS (Streamlit)
results/           graphiques et tableaux de résultats
```

## Technologies
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn, Jupyter.

## Référence
Sharafaldin, Habibi Lashkari, Ghorbani, *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*, ICISSP 2018.
