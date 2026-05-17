"""
Construye data/dane/poblacion_municipios_2007_2024.xlsx combinando las dos
series oficiales del DANE con base CNPV 2018:

  - Retroproyección municipal 2005-2017 (archivo DCD).
  - Proyección municipal 2018-2042 post-COVID (archivo PPED).

Ambas usan la misma base censal, así que la serie resultante 2007-2024 es
continua (sin el escalón que tenía el archivo anterior, que mezclaba bases
Censo 2005 + Censo 2018).

Salida: misma estructura que el archivo anterior (1 fila por municipio,
columnas pob_2007 .. pob_2024, área = Total), para no romper cargar_dane().
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/dane/raw")
OUT = Path("data/dane/poblacion_municipios_2007_2024.xlsx")

A1 = RAW_DIR / "DCD-area-proypoblacion-Mun-2005-2017_VP.xlsx"   # 2005-2017
A2 = RAW_DIR / "PPED-AreaMun-2018-2042_VP.xlsx"                  # 2018-2042

ANOS_OBJETIVO = list(range(2007, 2025))


def cargar_serie_2005_2017() -> pd.DataFrame:
    """Hoja NuevaMpal, header en fila 11.
    Columnas: DP, DPNOM, DPMP=nombre, MPIO=codigo_DIVIPOLA, AÑO, ÁREA GEOGRÁFICA, Población.
    """
    df = pd.read_excel(A1, sheet_name="NuevaMpal", header=11)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "DP": "cod_dpto",
        "DPNOM": "departamento",
        "DPMP": "municipio",
        "MPIO": "cod_mpio",
        "AÑO": "ano",
        "ÁREA GEOGRÁFICA": "area",
        "Población": "poblacion",
    })
    df = df[df["area"].astype(str).str.strip() == "Total"]
    return df[["cod_dpto", "departamento", "cod_mpio", "municipio", "ano", "poblacion"]]


def cargar_serie_2018_2042() -> pd.DataFrame:
    """Hoja PobMunicipalxÁrea, header en fila 7.
    Columnas: DP, DPNOM, MPIO=codigo, DPMP=nombre, AÑO, ÁREA GEOGRÁFICA, TOTAL.
    """
    df = pd.read_excel(A2, sheet_name="PobMunicipalxÁrea", header=7)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "DP": "cod_dpto",
        "DPNOM": "departamento",
        "MPIO": "cod_mpio",
        "DPMP": "municipio",
        "AÑO": "ano",
        "ÁREA GEOGRÁFICA": "area",
        "TOTAL": "poblacion",
    })
    df = df[df["area"].astype(str).str.strip() == "Total"]
    return df[["cod_dpto", "departamento", "cod_mpio", "municipio", "ano", "poblacion"]]


def main():
    s1 = cargar_serie_2005_2017()
    s2 = cargar_serie_2018_2042()
    print(f"Serie 2005-2017: {s1.shape}, años {sorted(s1['ano'].unique())}")
    print(f"Serie 2018-2042: {s2.shape}, años {sorted(s2['ano'].unique())[:5]}...")

    long = pd.concat([s1, s2], ignore_index=True)
    long = long.dropna(subset=["cod_mpio", "ano"])
    long["ano"] = long["ano"].astype(int)
    long["cod_dpto"] = long["cod_dpto"].astype(int).astype(str).str.zfill(2)
    long["cod_mpio"] = long["cod_mpio"].astype(int).astype(str).str.zfill(5)
    long["poblacion"] = long["poblacion"].astype(int)

    long = long[long["ano"].isin(ANOS_OBJETIVO)]

    # pivot a wide
    wide = long.pivot_table(
        index=["cod_dpto", "departamento", "cod_mpio", "municipio"],
        columns="ano",
        values="poblacion",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    # renombrar años a DP/DPNOM/COD_MPIO/NOM_MPIO + año-como-string (esquema del archivo previo)
    wide = wide.rename(columns={
        "cod_dpto": "DP",
        "departamento": "DPNOM",
        "cod_mpio": "COD_MPIO",
        "municipio": "NOM_MPIO",
    })
    cols_finales = ["DP", "DPNOM", "COD_MPIO", "NOM_MPIO"] + [a for a in ANOS_OBJETIVO]
    wide = wide[cols_finales]
    wide.columns = ["DP", "DPNOM", "COD_MPIO", "NOM_MPIO"] + [str(a) for a in ANOS_OBJETIVO]

    print(f"\nResultado: {wide.shape}")
    print(f"Municipios: {wide['COD_MPIO'].nunique()}, Deptos: {wide['DPNOM'].nunique()}")
    nulos = wide[[str(a) for a in ANOS_OBJETIVO]].isna().sum()
    print(f"Nulos totales por año: {nulos.sum()}")

    # verificación de continuidad en los foco
    foco = ["23855", "47288", "95025", "20710", "23570"]
    print("\nContinuidad en municipios foco (Valencia, Fundación, El Retorno, San Alberto, Pueblo Nuevo):")
    chk_cols = ["NOM_MPIO", "2017", "2018", "2020", "2021"]
    chk = wide[wide["COD_MPIO"].isin(foco)][chk_cols].copy()
    chk["Δ_2017→2018_%"] = ((chk["2018"] - chk["2017"]) / chk["2017"] * 100).round(2)
    chk["Δ_2020→2021_%"] = ((chk["2021"] - chk["2020"]) / chk["2020"] * 100).round(2)
    print(chk.to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        wide.to_excel(w, sheet_name="Población Total", index=False)
        notas = pd.DataFrame({
            "NOTAS METODOLÓGICAS": [
                "",
                "FUENTE 2007–2017",
                "",
                "",
                "FUENTE 2018–2024",
                "",
                "",
                "Base censal unificada: CNPV 2018 (sin escalón 2020→2021).",
                "Área: TOTAL (cabecera + rural).",
            ],
            "Detalle": [
                "",
                "DANE — Retroproyecciones municipales 2005-2017, base CNPV 2018.",
                "Archivo: DCD-area-proypoblacion-Mun-2005-2017_VP.xlsx",
                "",
                "DANE — Proyecciones municipales 2018-2042 post COVID-19, base CNPV 2018.",
                "Archivo: PPED-AreaMun-2018-2042_VP.xlsx",
                "",
                "",
                "",
            ],
        })
        notas.to_excel(w, sheet_name="Notas", index=False)

    print(f"\n→ Escrito: {OUT}")


if __name__ == "__main__":
    main()
