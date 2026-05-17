"""
Construye data/processed/panel_municipal_mensual.{parquet,csv} para los
municipios foco (decisiones D1-D14 en docs/decisiones_proyecto.md).

Pipeline:
  1. Cargar SIVIGILA regular+grave SOLO de columnas clave (memoria).
  2. Filtrar a MUNICIPIOS_FOCO desde el inicio (3 mpios vs 1.040).
  3. Asignar cada caso al mes de INI_SIN (fallback FEC_NOT).
  4. Agregar por (cod_mpio, año, mes): conteos de casos.
  5. Cross-join con (foco × 2007-2024 × 12 meses) y fillna(0) en conteos.
  6. Merge clima (depto-mes) — D14.
  7. Merge población (DANE wide) — lookup directo pob_<año>.
  8. Target binario "exceso" = percentil 75 histórico por mes calendario
     usando solo años previos (D12).
  9. Lags 1-3 meses y MM3 globales por municipio (no within-year).
 10. Export .parquet y .csv.
"""
from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import (  # noqa: E402
    ANOS_ESTUDIO,
    MUNICIPIOS_FOCO,
    PROJECT_ROOT,
    cargar_clima,
    cargar_dane,
    cargar_dengue,
)


COLS_NECESARIAS = [
    "CONSECUTIVE", "COD_EVE", "ANO", "SEMANA",
    "COD_DPTO_O", "COD_MUN_O", "COD_DPTO_N", "COD_MUN_N",
    "INI_SIN", "FEC_NOT", "PAC_HOS", "CON_FIN",
    "Departamento_ocurrencia", "Municipio_ocurrencia",
]

CODS_FOCO = list(MUNICIPIOS_FOCO.keys())  # ['23855', '47288', '95025']


def normalizar(s):
    if not isinstance(s, str):
        return s
    s = s.upper().strip()
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# Mapeo de departamento del panel (SIVIGILA) → nombre en clima (FAO/GAUL).
# Los 3 foco están en: Córdoba, Magdalena, Guaviare. Todos tienen match directo
# tras normalizar acentos. Dejo el mapeo abierto por si extendemos foco.
MAPEO_DPTO_CLIMA = {
    "NORTE SANTANDER": "NORTE DE SANTANDER",
    "VALLE": "VALLE DEL CAUCA",
    "LA GUAJIRA": "GUAJIRA",
    "BOGOTA D.C.": "CUNDINAMARCA",
    "SAN ANDRES": "SAN ANDRES Y PROVIDENCIA",
}


# ---------------------------------------------------------------------------
# 1-2. Cargar SIVIGILA filtrado a foco
# ---------------------------------------------------------------------------
def cargar_casos_foco() -> pd.DataFrame:
    """Carga regular+grave, columnas clave, filtrado a MUNICIPIOS_FOCO."""
    print("=" * 70)
    print("Cargando SIVIGILA (filtrado a municipios foco)")
    print("=" * 70)

    dfs = []
    for tipo in ("regular", "grave"):
        print(f"\n--- Dengue {tipo} ---")
        df = cargar_dengue(tipo, usecols=COLS_NECESARIAS)
        df["tipo_dengue"] = tipo
        # Pad código de municipio (ocurrencia) y filtrar a foco
        df["cod_mpio"] = df["COD_MUN_N"].astype("Int64").astype(str).str.zfill(5)
        antes = len(df)
        df = df[df["cod_mpio"].isin(CODS_FOCO)].copy()
        print(f"  {tipo}: {len(df):,} registros tras filtrar a foco (de {antes:,})")
        dfs.append(df)

    casos = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal foco (regular + grave): {len(casos):,} casos")
    return casos


# ---------------------------------------------------------------------------
# 3-4. Asignar mes (INI_SIN con fallback FEC_NOT) y agregar
# ---------------------------------------------------------------------------
def construir_agregado_mensual(casos: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("Agregando casos por (municipio, año, mes)")
    print("=" * 70)

    casos["INI_SIN"] = pd.to_datetime(casos["INI_SIN"], errors="coerce")
    casos["FEC_NOT"] = pd.to_datetime(casos["FEC_NOT"], errors="coerce")
    fecha = casos["INI_SIN"].fillna(casos["FEC_NOT"])

    n_nulos = fecha.isna().sum()
    if n_nulos:
        print(f"  Descartados {n_nulos:,} casos sin INI_SIN ni FEC_NOT")
    casos = casos[fecha.notna()].copy()
    fecha = fecha[fecha.notna()]
    casos["ano"] = fecha.dt.year.astype(int)
    casos["mes"] = fecha.dt.month.astype(int)

    # Quedarse solo en el rango de estudio
    casos = casos[casos["ano"].isin(ANOS_ESTUDIO)]
    print(f"  Casos en rango 2007-2024: {len(casos):,}")

    # Dedupe por CONSECUTIVE (defensa anti-duplicados entre regular/grave)
    antes = len(casos)
    casos = casos.drop_duplicates(subset=["CONSECUTIVE"], keep="first")
    if antes != len(casos):
        print(f"  Duplicados por CONSECUTIVE eliminados: {antes - len(casos):,}")

    # Indicadores
    casos["es_regular"] = (casos["tipo_dengue"] == "regular").astype(int)
    casos["es_grave"] = (casos["tipo_dengue"] == "grave").astype(int)
    casos["es_hosp"] = (casos["PAC_HOS"] == 1).astype(int)
    casos["es_fallecido"] = (casos["CON_FIN"] == 2).astype(int)

    agg = (
        casos.groupby(["cod_mpio", "ano", "mes"])
        .agg(
            casos_total=("CONSECUTIVE", "count"),
            casos_regular=("es_regular", "sum"),
            casos_grave=("es_grave", "sum"),
            hospitalizaciones=("es_hosp", "sum"),
            fallecidos=("es_fallecido", "sum"),
        )
        .reset_index()
    )
    print(f"\n  Agregado: {len(agg):,} filas (mpio × mes con al menos 1 caso)")
    return agg


# ---------------------------------------------------------------------------
# 5. Cross-join completo (foco × año × mes) y fillna en conteos
# ---------------------------------------------------------------------------
def construir_panel_completo(agg: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("Cross-join panel completo (foco × año × mes)")
    print("=" * 70)

    grid = pd.DataFrame(
        [(c, a, m) for c in CODS_FOCO for a in ANOS_ESTUDIO for m in range(1, 13)],
        columns=["cod_mpio", "ano", "mes"],
    )
    grid["municipio"] = grid["cod_mpio"].map(MUNICIPIOS_FOCO)

    panel = grid.merge(agg, on=["cod_mpio", "ano", "mes"], how="left")
    cols_conteo = ["casos_total", "casos_regular", "casos_grave",
                   "hospitalizaciones", "fallecidos"]
    panel[cols_conteo] = panel[cols_conteo].fillna(0).astype(int)

    print(f"  Panel: {len(panel):,} filas ({len(CODS_FOCO)} mpios × "
          f"{len(ANOS_ESTUDIO)} años × 12 meses)")
    return panel


# ---------------------------------------------------------------------------
# 6. Merge clima (depto-mes)
# ---------------------------------------------------------------------------
def integrar_clima(panel: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("Integrando clima (departamento-mes)")
    print("=" * 70)

    # Mapeo cod_mpio → depto normalizado del clima.
    # Los 3 foco viven en Córdoba (23), Magdalena (47), Guaviare (95).
    dpto_por_mpio = {
        "23855": "CORDOBA",       # Valencia
        "47288": "MAGDALENA",     # Fundación
        "95025": "GUAVIARE",      # El Retorno
    }
    panel["dpto_clima"] = panel["cod_mpio"].map(dpto_por_mpio)

    clima = cargar_clima()
    if clima is None:
        raise RuntimeError("No se pudo cargar clima_consolidado.csv")
    clima["dpto_norm"] = clima["departamento"].apply(normalizar)
    clima["dpto_norm"] = clima["dpto_norm"].replace(MAPEO_DPTO_CLIMA)

    cols_clima = ["temperatura_c", "precipitacion_mm", "ndvi", "dewpoint_c"]
    panel = panel.merge(
        clima[["dpto_norm", "ano", "mes"] + cols_clima],
        left_on=["dpto_clima", "ano", "mes"],
        right_on=["dpto_norm", "ano", "mes"],
        how="left",
    ).drop(columns=["dpto_norm"])

    cobertura = panel[cols_clima[0]].notna().mean() * 100
    print(f"  Cobertura clima: {cobertura:.1f}%")
    if cobertura < 100:
        print(f"    Filas sin clima: {panel[cols_clima[0]].isna().sum()}")
        print(f"    Pares (dpto, ano, mes) sin match: "
              f"{panel.loc[panel[cols_clima[0]].isna(), ['dpto_clima','ano','mes']].drop_duplicates()}")
    return panel


# ---------------------------------------------------------------------------
# 7. Población
# ---------------------------------------------------------------------------
def integrar_poblacion(panel: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("Integrando población DANE")
    print("=" * 70)

    dane = cargar_dane()  # cod_mpio, ..., pob_2007..pob_2024
    pob_long = dane.melt(
        id_vars=["cod_mpio"],
        value_vars=[f"pob_{a}" for a in ANOS_ESTUDIO],
        var_name="ano_str", value_name="poblacion",
    )
    pob_long["ano"] = pob_long["ano_str"].str.replace("pob_", "").astype(int)
    pob_long = pob_long[["cod_mpio", "ano", "poblacion"]]
    pob_long = pob_long[pob_long["cod_mpio"].isin(CODS_FOCO)]

    panel = panel.merge(pob_long, on=["cod_mpio", "ano"], how="left")
    panel["incidencia_x100k"] = panel["casos_total"] / panel["poblacion"] * 100_000

    cobertura = panel["poblacion"].notna().mean() * 100
    print(f"  Cobertura población: {cobertura:.1f}%")
    return panel


# ---------------------------------------------------------------------------
# 8. Target: exceso por percentil 75 histórico por mes calendario (solo años previos)
# ---------------------------------------------------------------------------
def construir_target(panel: pd.DataFrame, percentil: float = 75) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"Target: exceso = casos_total > percentil-{percentil:.0f} histórico mes")
    print("=" * 70)

    panel = panel.sort_values(["cod_mpio", "ano", "mes"]).reset_index(drop=True)

    # Para cada fila (mpio, año, mes): percentil de casos_total en ese mismo mes
    # calendario, considerando solo los años estrictamente anteriores.
    def umbral_historico(g: pd.DataFrame) -> pd.Series:
        # g está ordenado por año (un solo mpio, un solo mes)
        casos = g["casos_total"].values
        anos = g["ano"].values
        out = np.full(len(g), np.nan)
        for i in range(len(g)):
            hist = casos[anos < anos[i]]
            if len(hist) >= 2:  # necesita al menos 2 años para un cuartil estable
                out[i] = np.percentile(hist, percentil)
        return pd.Series(out, index=g.index)

    panel["umbral_p75"] = (
        panel.groupby(["cod_mpio", "mes"], group_keys=False)
        .apply(umbral_historico)
    )
    panel["exceso"] = (panel["casos_total"] > panel["umbral_p75"]).astype("Int64")
    # Para filas sin umbral (primeros años, < 2 años de historia): NaN
    panel.loc[panel["umbral_p75"].isna(), "exceso"] = pd.NA

    prevalencia = panel["exceso"].mean()
    print(f"  Prevalencia global de exceso: {float(prevalencia)*100:.1f}%")
    print(f"  Filas con target definido: {int(panel['exceso'].notna().sum())}/{len(panel)}")
    for cod, nom in MUNICIPIOS_FOCO.items():
        sub = panel[panel["cod_mpio"] == cod]
        prev = sub["exceso"].mean()
        print(f"    {nom}: prevalencia = {float(prev)*100:.1f}%")
    return panel


# ---------------------------------------------------------------------------
# 9. Lags globales por municipio (continuos en el tiempo, no within-year)
# ---------------------------------------------------------------------------
def crear_lags(panel: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("Creando lags y medias móviles (globales por municipio)")
    print("=" * 70)

    vars_lag = ["temperatura_c", "precipitacion_mm", "ndvi", "dewpoint_c",
                "casos_total", "incidencia_x100k"]
    lags = [1, 2, 3]

    panel = panel.sort_values(["cod_mpio", "ano", "mes"]).reset_index(drop=True)
    for var in vars_lag:
        for L in lags:
            panel[f"{var}_lag{L}"] = panel.groupby("cod_mpio")[var].shift(L)
        panel[f"{var}_mm3"] = (
            panel.groupby("cod_mpio")[var]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )
    n_nuevos = len(vars_lag) * (len(lags) + 1)
    print(f"  Features temporales creados: {n_nuevos}")
    return panel


# ---------------------------------------------------------------------------
# 10. Export
# ---------------------------------------------------------------------------
def exportar(panel: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Exportando panel")
    print("=" * 70)

    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_parquet = out_dir / "panel_municipal_mensual.parquet"
    p_csv = out_dir / "panel_municipal_mensual.csv"
    panel.to_parquet(p_parquet, index=False)
    panel.to_csv(p_csv, index=False)
    print(f"  → {p_parquet}")
    print(f"  → {p_csv}")
    print(f"\nPanel final: {panel.shape[0]:,} filas × {panel.shape[1]} columnas")
    print(f"Columnas: {list(panel.columns)}")


def main():
    casos = cargar_casos_foco()
    agg = construir_agregado_mensual(casos)
    panel = construir_panel_completo(agg)
    panel = integrar_clima(panel)
    panel = integrar_poblacion(panel)
    panel = construir_target(panel)
    panel = crear_lags(panel)
    exportar(panel)


if __name__ == "__main__":
    main()
