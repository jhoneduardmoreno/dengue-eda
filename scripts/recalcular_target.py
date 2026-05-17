"""Recalcula la columna `exceso` del panel ya construido, sin re-leer SIVIGILA.

Implementa el target revisado tras el EDA: percentil 75 histórico por mes
calendario con piso mínimo de 2 casos. Documentado en D12 revisada en
docs/decisiones_proyecto.md.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import MUNICIPIOS_FOCO, PROJECT_ROOT  # noqa: E402


PERCENTIL = 75
PISO = 2

PANEL_DIR = PROJECT_ROOT / "data" / "processed"
PANEL_PARQUET = PANEL_DIR / "panel_municipal_mensual.parquet"
PANEL_CSV = PANEL_DIR / "panel_municipal_mensual.csv"


def umbral_historico(g: pd.DataFrame, q: float) -> pd.Series:
    casos = g["casos_total"].values
    anos = g["ano"].values
    out = np.full(len(g), np.nan)
    for i in range(len(g)):
        hist = casos[anos < anos[i]]
        if len(hist) >= 2:
            out[i] = np.percentile(hist, q)
    return pd.Series(out, index=g.index)


def main():
    panel = pd.read_parquet(PANEL_PARQUET)
    print(f"Panel cargado: {panel.shape[0]:,} filas × {panel.shape[1]} columnas")

    panel = panel.sort_values(["cod_mpio", "ano", "mes"]).reset_index(drop=True)

    # Drop columnas viejas si existen (por si re-ejecutamos)
    for c in ("umbral_p75", "umbral_exceso", "exceso"):
        if c in panel.columns:
            panel = panel.drop(columns=c)

    grupos = list(panel.groupby(["cod_mpio", "mes"], group_keys=False))
    pieces = []
    for _, g in tqdm(grupos, desc=f"Calculando umbral p{PERCENTIL}", total=len(grupos)):
        pieces.append(umbral_historico(g, PERCENTIL))
    umbral = pd.concat(pieces).reindex(panel.index)

    umbral_con_piso = umbral.where(umbral.isna(), umbral.clip(lower=PISO))

    panel["umbral_exceso"] = umbral_con_piso
    panel["exceso"] = (panel["casos_total"] > panel["umbral_exceso"]).astype("Int64")
    panel.loc[panel["umbral_exceso"].isna(), "exceso"] = pd.NA

    prev = float(panel["exceso"].mean()) * 100
    n_def = int(panel["exceso"].notna().sum())
    print(f"\nTarget: exceso = casos > max(p{PERCENTIL}_hist(mes), {PISO})")
    print(f"  Prevalencia global: {prev:.1f}%")
    print(f"  Filas con target definido: {n_def}/{len(panel)}")
    for cod, nom in MUNICIPIOS_FOCO.items():
        sub = panel[panel["cod_mpio"] == cod]
        p = float(sub["exceso"].mean()) * 100
        n = int(sub["exceso"].sum())
        print(f"    {nom}: prevalencia = {p:.1f}% ({n} meses de exceso)")

    panel.to_parquet(PANEL_PARQUET, index=False)
    panel.to_csv(PANEL_CSV, index=False)
    print(f"\nGuardado:\n  {PANEL_PARQUET}\n  {PANEL_CSV}")


if __name__ == "__main__":
    main()
