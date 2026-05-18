"""
Análisis de sensibilidad del umbral del target (percentil + piso).

Re-evalúa XGBoost en una grilla de definiciones del target manteniendo los
hiperparámetros óptimos ya encontrados (no re-tuneo). El objetivo es responder
"¿el desempeño es estable al cambiar el umbral, o depende fuertemente de la
elección piso=2 documentada en D12?".

Grid evaluado:
  - Percentil histórico: 75, 90
  - Piso mínimo:         0, 1, 2, 3

Total: 8 combinaciones × 3 municipios = 24 re-entrenamientos (rápido).

Salidas:
  - data/processed/sensibilidad_target.csv  (tabla long-format con métricas)
  - results_graphs/foco/08_sensibilidad_target.png (grid 3×3: P/R/F1 vs piso)
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# Forzar backend no interactivo antes de cualquier import de matplotlib (transitivo).
os.environ["MPLBACKEND"] = "Agg"

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import MUNICIPIOS_FOCO, PROJECT_ROOT, configurar_estilo  # noqa: E402

configurar_estilo()

PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "panel_municipal_mensual.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "sensibilidad_target.csv"
OUT_FIG = PROJECT_ROOT / "results_graphs" / "foco" / "08_sensibilidad_target.png"

ANO_TEST_DESDE = 2020
PERCENTILES = [75, 90]
PISOS = [0, 1, 2, 3]
TARGET_REF = (75, 2)  # combinación documentada en D12

FEATURES_CLIMA = ["temperatura_c", "precipitacion_mm", "ndvi", "dewpoint_c"]
FEATURES_CLIMA_LAG = [f"{v}_lag{L}" for v in FEATURES_CLIMA for L in [1, 2, 3]]
FEATURES_CLIMA_MM3 = [f"{v}_mm3" for v in FEATURES_CLIMA]
FEATURES_CASOS_LAG = [f"casos_total_lag{L}" for L in [1, 2, 3]]
FEATURES_INC_LAG = [f"incidencia_x100k_lag{L}" for L in [1, 2, 3]]
FEATURES_TIEMPO = ["mes_sin", "mes_cos"]
FEATURES = (FEATURES_CLIMA + FEATURES_CLIMA_LAG + FEATURES_CLIMA_MM3
            + FEATURES_CASOS_LAG + FEATURES_INC_LAG + FEATURES_TIEMPO)


def umbral_historico(g: pd.DataFrame, q: float) -> pd.Series:
    casos = g["casos_total"].values
    anos = g["ano"].values
    out = np.full(len(g), np.nan)
    for i in range(len(g)):
        hist = casos[anos < anos[i]]
        if len(hist) >= 2:
            out[i] = np.percentile(hist, q)
    return pd.Series(out, index=g.index)


def construir_target(panel: pd.DataFrame, percentil: float, piso: int) -> pd.Series:
    panel = panel.sort_values(["cod_mpio", "ano", "mes"]).reset_index(drop=True)
    umbral = (
        panel.groupby(["cod_mpio", "mes"], group_keys=False)
        .apply(lambda g: umbral_historico(g, percentil))
    )
    umbral = umbral.reindex(panel.index)
    if piso > 0:
        umbral = umbral.where(umbral.isna(), umbral.clip(lower=piso))
    exceso = pd.Series(pd.NA, index=panel.index, dtype="Int64")
    valid = umbral.notna()
    exceso.loc[valid] = (panel.loc[valid, "casos_total"] > umbral.loc[valid]).astype(int)
    return exceso, umbral


def main():
    print("Cargando panel y mejores hiperparámetros…")
    panel = pd.read_parquet(PANEL_PATH)
    panel["cod_mpio"] = panel["cod_mpio"].astype(str)
    panel["mes_sin"] = np.sin(2 * np.pi * panel["mes"] / 12)
    panel["mes_cos"] = np.cos(2 * np.pi * panel["mes"] / 12)

    best_params = {}
    for cod in MUNICIPIOS_FOCO:
        bundle = joblib.load(MODELS_DIR / f"{cod}_xgboost.joblib")
        best_params[cod] = bundle["best_params"]
    print(f"  best_params por municipio: {best_params}")

    resultados = []
    combos = [(p, f) for p in PERCENTILES for f in PISOS]
    for percentil, piso in tqdm(combos, desc="Grid sensibilidad"):
        panel_local = panel.copy()
        exceso, _ = construir_target(panel_local, percentil, piso)
        panel_local["exceso"] = exceso

        for cod, nom in MUNICIPIOS_FOCO.items():
            sub = panel_local[(panel_local["cod_mpio"] == cod)
                              & panel_local["exceso"].notna()]
            sub = sub.dropna(subset=FEATURES).sort_values(["ano", "mes"])
            tr = sub[sub["ano"] < ANO_TEST_DESDE]
            te = sub[sub["ano"] >= ANO_TEST_DESDE]

            # Caso degenerado: una sola clase en train o test
            degenerado = (
                len(tr) < 10 or len(te) < 5
                or tr["exceso"].nunique() < 2
                or te["exceso"].nunique() < 2
            )
            if degenerado:
                resultados.append({
                    "percentil": percentil, "piso": piso, "municipio": nom,
                    "n_train": len(tr), "n_test": len(te),
                    "prev_train": float(tr["exceso"].mean()) if len(tr) else np.nan,
                    "prev_test": float(te["exceso"].mean()) if len(te) else np.nan,
                    "precision": np.nan, "recall": np.nan, "f1": np.nan, "accuracy": np.nan,
                    "degenerado": True,
                })
                continue

            X_tr = tr[FEATURES].values
            y_tr = tr["exceso"].astype(int).values
            X_te = te[FEATURES].values
            y_te = te["exceso"].astype(int).values

            scaler = StandardScaler().fit(X_tr)
            Xs_tr = scaler.transform(X_tr)
            Xs_te = scaler.transform(X_te)

            pos = int(y_tr.sum())
            neg = int(len(y_tr) - pos)
            spw = neg / max(pos, 1)

            params = best_params[cod]
            modelo = xgb.XGBClassifier(
                **params,
                objective="binary:logistic",
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )
            modelo.fit(Xs_tr, y_tr)
            y_pred = modelo.predict(Xs_te)

            resultados.append({
                "percentil": percentil, "piso": piso, "municipio": nom,
                "n_train": len(tr), "n_test": len(te),
                "prev_train": float(tr["exceso"].mean()),
                "prev_test": float(te["exceso"].mean()),
                "precision": precision_score(y_te, y_pred, zero_division=0),
                "recall":    recall_score(y_te, y_pred, zero_division=0),
                "f1":        f1_score(y_te, y_pred, zero_division=0),
                "accuracy":  accuracy_score(y_te, y_pred),
                "degenerado": False,
            })

    df = pd.DataFrame(resultados)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nResultados guardados: {OUT_CSV}")
    print(f"\nResumen (Precision/Recall/F1 ordenado por municipio y piso):")
    cols_show = ["percentil", "piso", "municipio", "prev_test", "precision", "recall", "f1"]
    print(df[cols_show].round(3).to_string(index=False))

    # Figura: 3 columnas (municipios) × 3 filas (P, R, F1)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey="row")
    metricas = ["precision", "recall", "f1"]
    colores = {75: "#1f77b4", 90: "#d62728"}

    for row, metrica in enumerate(metricas):
        for col, (_, nom) in enumerate(MUNICIPIOS_FOCO.items()):
            ax = axes[row, col]
            for p in PERCENTILES:
                sub = df[(df["municipio"] == nom) & (df["percentil"] == p)
                          & (~df["degenerado"])].sort_values("piso")
                ax.plot(sub["piso"], sub[metrica], marker="o", lw=2,
                        color=colores[p], label=f"P{p}")
            # marcar la configuración usada en producción (P75, piso 2)
            if (metrica is not None) and (row == 0):
                ax.axvline(TARGET_REF[1], color="gray", lw=0.6, linestyle="--",
                           alpha=0.7)
            ax.set_title(f"{nom} — {metrica}", fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(PISOS)
            ax.grid(True, alpha=0.3)
            if row == 2:
                ax.set_xlabel("Piso del umbral")
            if col == 0:
                ax.set_ylabel(metrica.capitalize())
            if row == 0 and col == 2:
                ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        f"Sensibilidad de XGBoost al umbral del target "
        f"(P75 piso 2 = configuración usada, marcada con línea gris)",
        fontweight="bold", y=1.005,
    )
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=120, bbox_inches="tight")
    print(f"\nFigura guardada: {OUT_FIG}")


if __name__ == "__main__":
    main()
