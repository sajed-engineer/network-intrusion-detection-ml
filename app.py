"""NetGuard IDS : interface de détection d'intrusions réseau (projet 1).

Lancer :  streamlit run app.py
"""
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.live_detect import prepare
from src.train_final import MODEL_PATH

RAW_DIR = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

C_OK, C_BAD, C_WARN, C_ACC, C_UP, C_MUTED = "#34d399", "#f87171", "#fbbf24", "#2dd4bf", "#818cf8", "#94a3b8"

st.set_page_config(page_title="NetGuard IDS", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
      #MainMenu, footer {visibility:hidden;}
      .block-container {padding-top:1.6rem; padding-bottom:2rem; max-width:1500px;}
      [data-testid="stMetric"] {background:#111a2e; border:1px solid #1f2b46; border-radius:12px; padding:14px 18px;}
      [data-testid="stMetricLabel"] p {font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color:#94a3b8;}
      [data-testid="stMetricValue"] {font-size:1.45rem; font-weight:650;}
      [data-testid="stMetricValue"] > div, [data-testid="stMetricDelta"] > div {
          overflow:visible !important; text-overflow:clip !important; white-space:nowrap;}
      .ng-title {font-size:1.7rem; font-weight:700; margin:0; letter-spacing:-.01em;}
      .ng-sub {color:#94a3b8; margin:2px 0 0 0; font-size:.92rem;}
      .pill {display:inline-block; padding:4px 12px; border-radius:999px; font-size:.78rem; font-weight:600;
             background:#2b2457; color:#a5b4fc; border:1px solid #4338ca;}
      .stTabs [data-baseweb="tab"] {font-weight:600;}
      table.flows {width:100%; border-collapse:collapse; font-size:.88rem;}
      table.flows th {text-align:left; color:#94a3b8; font-weight:600; font-size:.72rem; letter-spacing:.06em;
                      text-transform:uppercase; padding:6px 10px; border-bottom:1px solid #1f2b46;}
      table.flows td {padding:7px 10px; border-bottom:1px solid #16213b;}
      .badge {padding:2px 10px; border-radius:6px; font-weight:700; font-size:.78rem;}
      .badge.alert {background:#4a1d1d; color:#f87171;} .badge.ok {background:#0f3d36; color:#34d399;}
      .bar {display:inline-block; height:8px; border-radius:4px; vertical-align:middle;}
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_label(s: str) -> str:
    """Remplace les caractères mal décodés (tiret long du dataset) par un tiret simple."""
    return re.sub(r"[^\x20-\x7e]+", "-", str(s)).strip()


@st.cache_resource(show_spinner=False)
def get_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Lecture et analyse des flux…")
def load_and_score(path_str: str | None, n: int, share: float, seed: int, upload: bytes | None = None) -> pd.DataFrame:
    bundle = get_bundle()
    raw = pd.read_csv(io.BytesIO(upload) if upload is not None else path_str, encoding="latin-1", low_memory=False)
    raw.columns = raw.columns.str.strip()
    needed = list(bundle["scaler"].feature_names_in_)
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes ({len(missing)}) : {', '.join(missing[:6])}…")
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    if "Label" in raw.columns:
        raw["Label"] = raw["Label"].map(clean_label)
        att, ben = raw[raw["Label"] != "BENIGN"], raw[raw["Label"] == "BENIGN"]
        k = min(int(n * share), len(att))
        sample = pd.concat([att.sample(k, random_state=seed),
                            ben.sample(min(n - k, len(ben)), random_state=seed)])
    else:
        sample = raw.sample(min(n, len(raw)), random_state=seed)
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    sample["score"] = bundle["model"].predict_proba(prepare(sample, bundle))[:, 1]
    return sample


def layout(title: str, sub: str = "", height: int = 340, **over) -> dict:
    base = dict(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"<b>{title}</b>  <span style='color:{C_MUTED};font-size:12px'>{sub}</span>",
                   x=0.0, font=dict(size=15)),
        height=height, margin=dict(l=8, r=8, t=56, b=8),
        legend=dict(orientation="h", y=1.14, x=1.0, xanchor="right"),
        xaxis=dict(gridcolor="#1a2540"), yaxis=dict(gridcolor="#1a2540", zeroline=False),
    )
    base.update(over)
    return base


def flows_html(rows: list[dict]) -> str:
    head = "<tr><th>#</th><th>Port dest.</th><th>Durée</th><th>Score de risque</th><th>Verdict</th><th>Réel</th><th></th></tr>"
    body = ""
    for r in rows:
        color = C_BAD if r["alert"] else C_OK
        verdict = '<span class="badge alert">ALERTE</span>' if r["alert"] else '<span class="badge ok">normal</span>'
        bar = f'<span class="bar" style="width:{max(r["score"] * 90, 3):.0f}px;background:{color}"></span> {r["score"]:.3f}'
        real = r.get("real", "—")
        mark = "" if "ok" not in r else ("✓" if r["ok"] else '<span style="color:#f87171">✗ erreur</span>')
        body += (f"<tr><td>{r['i']}</td><td>{r['port']}</td><td>{r['dur']:.3f} s</td><td>{bar}</td>"
                 f"<td>{verdict}</td><td>{real}</td><td>{mark}</td></tr>")
    return f'<table class="flows">{head}{body}</table>'


# ------------------------------------------------------------------ barre latérale
bundle = get_bundle() if MODEL_PATH.exists() else None
csv_files = sorted(RAW_DIR.rglob("*.csv")) if RAW_DIR.exists() else []

with st.sidebar:
    st.markdown("### 🛡️ NetGuard IDS")
    choices = [f.name for f in csv_files] + ["Importer un fichier CSV…"]
    source = st.selectbox("Source des flux", choices)
    upload = st.file_uploader("Fichier de flux (format CICFlowMeter)", type="csv") if source == choices[-1] else None
    n = st.slider("Flux à analyser", 1000, 20000, 5000, 1000)
    share = st.slider("Part d'attaques dans l'échantillon", 0.05, 0.60, 0.25, 0.05, format="%.2f",
                      help="L'échantillon est rééquilibré pour qu'il contienne assez d'attaques à observer.")
    threshold = st.slider("Seuil d'alerte", 0.05, 0.95, 0.50, 0.05,
                          help="Score minimal pour déclencher une alerte. Plus bas = plus sensible, plus de fausses alertes.")
    seed = st.number_input("Graine aléatoire", 0, 9999, 7)
    st.divider()
    st.caption("Modèle : XGBoost entraîné sur 2 M de flux CIC-IDS2017. "
               "Les flux affichés viennent du dataset : ils peuvent avoir servi à l'entraînement. "
               "Les vraies performances sont dans l'onglet *Performances*.")

# ------------------------------------------------------------------ en-tête
left, right = st.columns([5, 2])
left.markdown('<p class="ng-title">Détection d\'intrusions réseau</p>'
              '<p class="ng-sub">Classification de flux par machine learning · CIC-IDS2017</p>', unsafe_allow_html=True)
right.markdown('<div style="text-align:right;padding-top:10px"><span class="pill">● RELECTURE DE FLUX DU DATASET</span></div>',
               unsafe_allow_html=True)
st.write("")

if bundle is None:
    st.error("Modèle introuvable. Lance d'abord :  `python -m src.train_final`")
    st.stop()

tab_an, tab_live, tab_perf, tab_about = st.tabs(["Analyse de flux", "Rejeu en direct", "Performances", "À propos"])


def get_scored() -> pd.DataFrame | None:
    try:
        if source == choices[-1]:
            if upload is None:
                st.info("Importe un fichier CSV dans la barre latérale (colonnes CICFlowMeter, comme CIC-IDS2017).")
                return None
            return load_and_score(None, n, share, int(seed), upload.getvalue())
        path = next(f for f in csv_files if f.name == source)
        return load_and_score(str(path), n, share, int(seed))
    except Exception as e:  # noqa: BLE001
        st.error(f"Impossible d'analyser ce fichier : {e}")
        return None


if not csv_files and source == choices[-1] and upload is None:
    pass

# ------------------------------------------------------------------ 1. Analyse
with tab_an:
    df = get_scored()
    if df is None:
        if not csv_files:
            st.warning("Aucun CSV trouvé dans `data/raw/`. Place-y le dataset CIC-IDS2017 ou importe un fichier.")
    else:
        df = df.assign(alert=df["score"] >= threshold)
        has_label = "Label" in df.columns
        k = st.columns(6)
        k[0].metric("Flux analysés", f"{len(df):,}".replace(",", " "))
        k[1].metric("Alertes", f"{int(df['alert'].sum()):,}".replace(",", " "))
        k[2].metric("Taux d'alerte", f"{100 * df['alert'].mean():.1f} %")
        if has_label:
            y = (df["Label"] != "BENIGN").to_numpy()
            p = df["alert"].to_numpy()
            tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            k[3].metric("Précision*", f"{100 * prec:.1f} %")
            k[4].metric("Rappel*", f"{100 * rec:.1f} %")
            k[5].metric("Attaques ratées*", f"{fn}", delta_color="off")
            st.caption("*Sur cet échantillon rééquilibré (voir « Part d'attaques ») : la précision dépend de cette proportion. "
                       "Les performances de référence sont dans l'onglet Performances.")
        else:
            k[3].metric("Précision", "—"); k[4].metric("Rappel", "—"); k[5].metric("Attaques ratées", "—")
            st.caption("Ce fichier n'a pas de colonne `Label` : pas de comparaison avec la réalité.")
        st.write("")

        c1, c2 = st.columns(2)
        f = go.Figure()
        if has_label:
            for name, mask, col in (("Trafic normal", df["Label"] == "BENIGN", C_ACC),
                                    ("Attaque", df["Label"] != "BENIGN", C_BAD)):
                f.add_trace(go.Histogram(x=df.loc[mask, "score"], name=name, marker_color=col, xbins=dict(start=0, end=1, size=0.02)))
            f.update_layout(barmode="overlay")
            f.update_traces(opacity=0.8)
        else:
            f.add_trace(go.Histogram(x=df["score"], name="Flux", marker_color=C_ACC, xbins=dict(start=0, end=1, size=0.02)))
        f.add_vline(x=threshold, line=dict(color=C_WARN, dash="dash"), annotation_text="seuil", annotation_font_color=C_WARN)
        f.update_layout(**layout("Distribution des scores de risque", "échelle log"), yaxis_type="log")
        c1.plotly_chart(f, width="stretch")

        if has_label:
            g = (df.groupby("Label").agg(n=("alert", "size"), flagged=("alert", "mean")).reset_index()
                 .sort_values("flagged"))
            g["pct"] = 100 * g["flagged"]
            colors = [C_MUTED if l == "BENIGN" else (C_OK if v >= 90 else C_WARN if v >= 50 else C_BAD)
                      for l, v in zip(g["Label"], g["pct"])]
            f = go.Figure(go.Bar(
                x=g["pct"], y=[("Normal (fausses alertes)" if l == "BENIGN" else l) for l in g["Label"]],
                orientation="h", marker_color=colors, customdata=g["n"],
                text=[f"{v:.0f} %  (n={n_})" for v, n_ in zip(g["pct"], g["n"])], textposition="outside",
                hovertemplate="%{y}<br>%{x:.1f} % signalés<br>n=%{customdata}<extra></extra>"))
            f.update_layout(**layout("Part signalée comme attaque", "par type de trafic",
                                     xaxis=dict(range=[0, 165], gridcolor="#1a2540", tickvals=[0, 25, 50, 75, 100])))
            c2.plotly_chart(f, width="stretch")
        else:
            ports = df.loc[df["alert"], "Destination Port"].value_counts().head(10)
            f = go.Figure(go.Bar(x=ports.index.astype(str), y=ports.values, marker_color=C_BAD))
            f.update_layout(**layout("Ports les plus visés", "parmi les alertes"))
            c2.plotly_chart(f, width="stretch")

        c3, c4 = st.columns(2)
        ports = df.loc[df["alert"], "Destination Port"].astype(int).value_counts().head(10)
        f = go.Figure(go.Bar(x=ports.index.astype(str), y=ports.values, marker_color=C_BAD))
        f.update_layout(**layout("Ports les plus visés", "parmi les alertes",
                                 xaxis=dict(type="category", gridcolor="#1a2540")))
        c3.plotly_chart(f, width="stretch")

        imp = pd.Series(bundle["model"].feature_importances_, index=bundle["columns"]).sort_values().tail(12)
        f = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=C_UP))
        f.update_layout(**layout("Variables les plus influentes", "importance dans le modèle"))
        c4.plotly_chart(f, width="stretch")

        st.markdown("#### Flux signalés")
        only_err = st.checkbox("Afficher seulement les erreurs du modèle", value=False, disabled=not has_label)
        view = df.copy()
        if has_label:
            view["verdict_ok"] = view["alert"] == (view["Label"] != "BENIGN")
            if only_err:
                view = view[~view["verdict_ok"]]
        else:
            view = view[view["alert"]]
        view = view.sort_values("score", ascending=False).head(300)
        table = pd.DataFrame({
            "Score": view["score"].round(3),
            "Verdict": np.where(view["alert"], "🔴 Alerte", "🟢 Normal"),
            "Port dest.": view["Destination Port"].astype(int),
            "Durée (s)": (view["Flow Duration"] / 1e6).round(3),
            "Paquets fwd": view["Total Fwd Packets"].astype(int),
            "Octets fwd": view["Total Length of Fwd Packets"].astype(int),
        })
        if has_label:
            table.insert(2, "Réel", view["Label"].values)
            table.insert(3, "Correct", np.where(view["verdict_ok"], "✓", "✗"))
        st.dataframe(table, hide_index=True, width="stretch", height=360)
        st.download_button("Exporter les résultats en CSV",
                           df.drop(columns=[c for c in df.columns if c.startswith("alert")]).assign(alert=df["alert"])
                           .to_csv(index=False).encode("utf-8"), file_name="netguard_resultats.csv", mime="text/csv")

# ------------------------------------------------------------------ 2. Rejeu
with tab_live:
    st.markdown("Les flux sont rejoués **un par un** : le modèle donne un score de risque et déclenche une alerte au-delà du seuil.")
    c = st.columns([1, 1, 1, 2])
    n_live = c[0].number_input("Nombre de flux", 10, 200, 30, 10)
    delay = c[1].number_input("Pause (s)", 0.0, 2.0, 0.3, 0.1)
    go_live = c[2].button("▶ Démarrer le rejeu", type="primary", width="stretch")
    if go_live:
        df = get_scored()
        if df is not None:
            demo = df.head(int(n_live))
            kpi, tbl = st.empty(), st.empty()
            rows, alerts, correct = [], 0, 0
            has_label = "Label" in demo.columns
            for i, (_, r) in enumerate(demo.iterrows(), 1):
                al = bool(r["score"] >= threshold)
                alerts += al
                row = {"i": i, "port": int(r["Destination Port"]), "dur": float(r["Flow Duration"]) / 1e6,
                       "score": float(r["score"]), "alert": al}
                if has_label:
                    ok = al == (r["Label"] != "BENIGN")
                    correct += ok
                    row.update(real=r["Label"], ok=ok)
                rows.append(row)
                with kpi.container():
                    m = st.columns(3)
                    m[0].metric("Flux traités", f"{i}/{len(demo)}")
                    m[1].metric("Alertes", alerts)
                    m[2].metric("Correctement classés", f"{correct}/{i}" if has_label else "—")
                tbl.markdown(flows_html(rows[-12:][::-1]), unsafe_allow_html=True)
                time.sleep(float(delay))
            st.success("Rejeu terminé.")
    else:
        st.caption("Clique sur « Démarrer » pour lancer le rejeu avec les paramètres de la barre latérale.")

# ------------------------------------------------------------------ 3. Performances
with tab_perf:
    mpath = RESULTS / "final_model_metrics.json"
    if not mpath.exists():
        st.warning("Métriques du modèle final absentes. Lance :  `python -m src.export_metrics`")
    else:
        m = json.loads(mpath.read_text(encoding="utf-8"))
        st.markdown(f"**Modèle final (XGBoost)** · jeu de test de **{m['n_test']:,}** flux "
                    f"dont **{m['n_attacks']:,}** attaques · split aléatoire stratifié.".replace(",", " "))
        k = st.columns(5)
        k[0].metric("Précision", f"{100 * m['precision']:.2f} %")
        k[1].metric("Rappel", f"{100 * m['recall']:.2f} %")
        k[2].metric("F1-score", f"{m['f1']:.4f}")
        k[3].metric("Fausses alertes", f"{m['false_alarm_pct']:.2f} %")
        k[4].metric("Attaques ratées", f"{m['confusion']['fn']:,}".replace(",", " "))
        st.write("")

        c1, c2 = st.columns([3, 2])
        pc = pd.DataFrame(m["per_class"]).T.reset_index().rename(columns={"index": "Label"})
        pc["Label"] = pc["Label"].map(clean_label)
        pc = pc[pc["Label"] != "BENIGN"].sort_values("flagged_pct")
        colors = [C_MUTED if n_ < 50 else (C_OK if v >= 90 else C_WARN if v >= 50 else C_BAD)
                  for n_, v in zip(pc["n"], pc["flagged_pct"])]
        f = go.Figure(go.Bar(
            x=pc["flagged_pct"], y=pc["Label"], orientation="h", marker_color=colors,
            text=[f"{v:.1f} %  (n={int(n_)}{' ⚠' if n_ < 50 else ''})" for v, n_ in zip(pc["flagged_pct"], pc["n"])],
            textposition="outside"))
        f.update_layout(**layout("Détection par type d'attaque", "gris ⚠ = trop peu d'exemples pour conclure", 430,
                                 xaxis=dict(range=[0, 170], gridcolor="#1a2540", tickvals=[0, 25, 50, 75, 100])))
        c1.plotly_chart(f, width="stretch")

        cm = m["confusion"]
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        f = go.Figure(go.Heatmap(z=np.log1p(z), x=["Prédit normal", "Prédit attaque"], y=["Réel normal", "Réel attaque"],
                                 text=[[f"{v:,}".replace(",", " ") for v in row] for row in z], texttemplate="%{text}",
                                 colorscale=[[0, "#111a2e"], [1, C_ACC]], showscale=False,
                                 textfont=dict(size=16)))
        f.update_layout(**layout("Matrice de confusion", "jeu de test", 430,
                                 yaxis=dict(autorange="reversed", showgrid=False)))
        c2.plotly_chart(f, width="stretch")

        cmp_path = RESULTS / "model_comparison_full.csv"
        if cmp_path.exists():
            st.markdown("#### Comparaison des 3 approches")
            cmp = pd.read_csv(cmp_path)
            cmp.columns = ["Modèle", "Précision", "Rappel", "F1", "ROC AUC", "Entraînement (s)"]
            st.dataframe(cmp, hide_index=True, width="stretch")
            st.caption("Random Forest, XGBoost et Isolation Forest comparés avec 400 000 flux d'entraînement. "
                       "Le modèle final ci-dessus utilise les 2 millions de flux d'entraînement.")

# ------------------------------------------------------------------ 4. À propos
with tab_about:
    st.markdown(
        """
**Projet.** Système de détection d'intrusions par apprentissage automatique sur **CIC-IDS2017**
(2,5 M de flux réseau, 15 classes de trafic, 70 variables calculées par CICFlowMeter).

**Méthode.** Nettoyage (valeurs infinies, doublons, valeurs négatives impossibles) → split stratifié 80/20 →
normalisation ajustée sur l'entraînement seulement (pas de fuite de données) → comparaison Random Forest /
XGBoost / Isolation Forest → modèle final XGBoost.

**Limites à connaître.**
- Le split est **aléatoire** : des flux d'une même attaque sont dans l'entraînement et le test. Un test par jour serait plus exigeant.
- **Infiltration** (6 flux de test), **SQL Injection** (4) et **Heartbleed** (2) ont trop peu d'exemples pour conclure :
  leurs scores ne sont pas significatifs. Infiltration reste la classe la plus difficile (comportement proche du trafic normal).
- Cette interface **rejoue des flux du dataset** : elle ne capture pas le réseau en direct. Le modèle attend les
  variables de CICFlowMeter ; pour du trafic réel, il faut convertir une capture `.pcap` avec cet outil.
- Les flux montrés peuvent avoir servi à l'entraînement : ils illustrent le fonctionnement, ils ne mesurent pas la performance.
        """
    )
