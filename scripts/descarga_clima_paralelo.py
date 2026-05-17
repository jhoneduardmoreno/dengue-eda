"""
Descarga clima desde Google Earth Engine usando reduceRegions (batched)
y ThreadPoolExecutor (paralelo por año).

Speedup esperado vs notebook original: 15-30x.
- reduceRegions: 33 deptos en una sola llamada por (dataset, mes) → 48 calls/año
  en vez de 1.584.
- ThreadPoolExecutor: corre 4 años en paralelo.

Solo procesa años de ANOS_ESTUDIO cuyo CSV todavía no exista.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import CLIMA_DIR, ANOS_ESTUDIO, GEE_PROJECT_ID  # noqa: E402


# (dataset_id, banda, columna_salida, reducer_temporal, escala_m, factor, offset)
# El offset se suma DESPUÉS de multiplicar por factor (para Kelvin → Celsius: offset=-273.15)
DATASETS = [
    ("MODIS/061/MOD11A2",            "LST_Day_1km",            "temperatura_c",    "mean", 1000,  0.02,   -273.15),
    ("UCSB-CHG/CHIRPS/DAILY",        "precipitation",          "precipitacion_mm", "sum",  5000,  1.0,     0.0),
    ("MODIS/061/MOD13A2",            "NDVI",                   "ndvi",             "mean", 1000,  0.0001,  0.0),
    ("ECMWF/ERA5_LAND/MONTHLY_AGGR", "dewpoint_temperature_2m","dewpoint_c",       "mean", 11000, 1.0,    -273.15),
]


def extraer_ano(ano: int, fc_deptos, nombres_deptos: list[str]) -> list[dict]:
    """Extrae las 4 variables para todos los deptos × 12 meses de un año, usando reduceRegions."""
    # estructura intermedia: {nombre_depto: {mes: {col: val}}}
    acc = {n: {m: {} for m in range(1, 13)} for n in nombres_deptos}

    for ds_id, banda, col, reducer_temp, escala, factor, offset in DATASETS:
        for mes in range(1, 13):
            inicio = ee.Date.fromYMD(ano, mes, 1)
            fin = inicio.advance(1, "month")
            ic = ee.ImageCollection(ds_id).filterDate(inicio, fin).select(banda)
            img = ic.sum() if reducer_temp == "sum" else ic.mean()

            try:
                fc_result = img.reduceRegions(
                    collection=fc_deptos,
                    reducer=ee.Reducer.mean(),
                    scale=escala,
                ).getInfo()
            except Exception as e:
                print(f"  [{ano}-{mes:02d}] error {col}: {e}", flush=True)
                continue

            for feat in fc_result["features"]:
                p = feat["properties"]
                nombre = p.get("ADM1_NAME")
                val = p.get("mean")
                if val is None or nombre is None:
                    continue
                val = val * factor + offset
                acc[nombre][mes][col] = round(val, 4 if col == "ndvi" else 2)

    registros = []
    for nombre in nombres_deptos:
        for mes in range(1, 13):
            r = {"departamento": nombre, "ano": ano, "mes": mes}
            for _, _, col, *_ in DATASETS:
                r[col] = acc[nombre][mes].get(col)
            registros.append(r)
    return registros


def main():
    print(f"Inicializando GEE (project={GEE_PROJECT_ID})...", flush=True)
    ee.Initialize(project=GEE_PROJECT_ID)

    colombia = (
        ee.FeatureCollection("FAO/GAUL/2015/level1")
        .filter(ee.Filter.eq("ADM0_NAME", "Colombia"))
    )
    nombres = sorted(colombia.aggregate_array("ADM1_NAME").getInfo())
    print(f"Deptos encontrados: {len(nombres)}", flush=True)

    os.makedirs(CLIMA_DIR, exist_ok=True)
    pendientes = [a for a in ANOS_ESTUDIO if not (CLIMA_DIR / f"clima_{a}.csv").exists()]
    print(f"Años pendientes ({len(pendientes)}): {pendientes}", flush=True)
    if not pendientes:
        print("Nada que descargar.", flush=True)
        return

    t0 = time.time()
    completados = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futuros = {pool.submit(extraer_ano, a, colombia, nombres): a for a in pendientes}
        for fut in as_completed(futuros):
            ano = futuros[fut]
            try:
                regs = fut.result()
            except Exception as e:
                print(f"[FAIL] año {ano}: {e}", flush=True)
                continue
            df = pd.DataFrame(regs)
            ruta = CLIMA_DIR / f"clima_{ano}.csv"
            df.to_csv(ruta, index=False)
            completados += 1
            elapsed = time.time() - t0
            print(
                f"[OK] año {ano} → {ruta.name} ({len(df)} filas, "
                f"{completados}/{len(pendientes)} en {elapsed:.0f}s)",
                flush=True,
            )

    # Consolidar TODOS los años (no solo los nuevos)
    print("Consolidando clima_consolidado.csv...", flush=True)
    dfs = []
    for a in sorted(ANOS_ESTUDIO):
        ruta = CLIMA_DIR / f"clima_{a}.csv"
        if ruta.exists():
            dfs.append(pd.read_csv(ruta))
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(CLIMA_DIR / "clima_consolidado.csv", index=False)
    print(
        f"clima_consolidado.csv: {df_all.shape}, "
        f"años: {sorted(df_all['ano'].unique())}",
        flush=True,
    )
    print(f"TOTAL: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
