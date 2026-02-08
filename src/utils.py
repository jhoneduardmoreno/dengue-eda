"""
Funciones auxiliares para el EDA de Dengue en Colombia.
Proyecto: Maestria en Inteligencia Artificial - Desarrollo de Soluciones
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# Rutas del proyecto
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DENGUE_DIR = DATA_DIR / "dengue"
DENGUE_GRAVE_DIR = DATA_DIR / "dengue fuerte"
DANE_DIR = DATA_DIR / "dane"
CLIMA_DIR = DATA_DIR / "clima"

# Anos de interes
ANOS_ESTUDIO = [2010, 2016, 2019, 2022, 2024]
ANOS_DENGUE_REGULAR = [2010, 2016, 2022, 2024]  # No hay 2019 para cod 210
ANOS_DENGUE_GRAVE = [2010, 2016, 2019, 2022, 2024]

# Columnas comunes clave para el analisis
COLS_CLAVE = [
    'COD_EVE', 'ANO', 'SEMANA', 'SEXO', 'EDAD', 'UNI_MED',
    'COD_DPTO_O', 'COD_MUN_O', 'AREA', 'TIP_SS', 'PER_ETN',
    'COD_DPTO_R', 'COD_MUN_R', 'COD_DPTO_N', 'COD_MUN_N',
    'INI_SIN', 'TIP_CAS', 'PAC_HOS', 'FEC_HOS', 'CON_FIN',
    'FEC_DEF', 'FECHA_NTO', 'FEC_NOT',
    'Departamento_ocurrencia', 'Municipio_ocurrencia',
    'Departamento_residencia', 'Municipio_residencia',
    'Estado_final_de_caso', 'nom_est_f_caso'
]

# Columnas extra que varian entre archivos (pueden no existir en todos)
COLS_EXTRA = ['Particion', 'Partición', 'COD_EVE.1', 'consecutive_origen']


# ============================================================================
# Configuracion de graficos
# ============================================================================
def configurar_estilo():
    """Configura estilo global para graficos consistentes."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'figure.dpi': 100,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
    })


COLORES_ANOS = {
    2010: '#1f77b4',
    2016: '#ff7f0e',
    2019: '#2ca02c',
    2022: '#d62728',
    2024: '#9467bd',
}

PALETA_DENGUE = {
    'regular': '#1f77b4',
    'grave': '#d62728',
}


# ============================================================================
# Carga de datos
# ============================================================================
def cargar_dengue(tipo='regular'):
    """
    Carga y concatena los archivos de dengue por tipo.

    Parameters
    ----------
    tipo : str
        'regular' para dengue (cod 210) o 'grave' para dengue grave (cod 220).

    Returns
    -------
    pd.DataFrame
        DataFrame concatenado con todos los anos disponibles.
    """
    if tipo == 'regular':
        directorio = DENGUE_DIR
        archivos = {
            2010: 'Datos_2010_210.xlsx',
            2016: 'Datos_2016_210.xlsx',
            2022: 'Datos_2022_210.xlsx',
            2024: 'Datos_2024_210.xlsx',
        }
    elif tipo == 'grave':
        directorio = DENGUE_GRAVE_DIR
        archivos = {
            2010: 'Datos_2010_220.xls',
            2016: 'Datos_2016_220.xls',
            2019: 'Datos_2019_220.xls',
            2022: 'Datos_2022_220.xls',
            2024: 'Datos_2024_220.xlsx',
        }
    else:
        raise ValueError("tipo debe ser 'regular' o 'grave'")

    dfs = []
    for ano, archivo in archivos.items():
        ruta = directorio / archivo
        print(f"  Cargando {archivo}...", end=" ")
        df = pd.read_excel(ruta)
        # Eliminar columnas extra que varian entre archivos
        for col in COLS_EXTRA:
            if col in df.columns:
                df = df.drop(columns=[col])
        print(f"{len(df):,} registros, {len(df.columns)} columnas")
        dfs.append(df)

    df_total = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total: {len(df_total):,} registros")
    return df_total


def cargar_dane():
    """
    Carga las proyecciones de poblacion del DANE.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: cod_dpto, departamento, cod_mpio, municipio,
        y una columna por cada ano (2005-2020) con la poblacion total.
    """
    ruta = DANE_DIR / "ProyeccionMunicipios2005_2020.xls"
    df = pd.read_excel(ruta, header=8, sheet_name='Mpios')

    # Las primeras 16 columnas de anos (2005-2020) son TOTAL
    # Las siguientes 16 son HOMBRES y las ultimas 16 son MUJERES
    cols_base = ['DP', 'DPNOM', 'DPMP', 'MPIO']
    anos = list(range(2005, 2021))

    # Renombrar columnas de poblacion total
    nuevas_cols = cols_base + [f'pob_{a}' for a in anos]
    # Solo tomar las primeras 20 columnas (4 base + 16 anos total)
    df_total = df.iloc[:, :20].copy()
    df_total.columns = nuevas_cols

    # Limpiar
    df_total = df_total.dropna(subset=['DP'])
    df_total['DP'] = df_total['DP'].astype(str).str.zfill(2)
    df_total['DPMP'] = df_total['DPMP'].astype(str).str.zfill(5)

    df_total = df_total.rename(columns={
        'DP': 'cod_dpto',
        'DPNOM': 'departamento',
        'DPMP': 'cod_mpio',
        'MPIO': 'municipio',
    })

    return df_total


def cargar_clima():
    """
    Carga los datos climaticos desde data/clima/.

    Returns
    -------
    pd.DataFrame o None
        DataFrame con datos climaticos, o None si no existen.
    """
    ruta = CLIMA_DIR / "clima_consolidado.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    else:
        print("  Advertencia: No se encontraron datos climaticos.")
        print("  Ejecute el notebook 00_descarga_clima_gee.ipynb primero.")
        return None


# ============================================================================
# Estandarizacion de columnas
# ============================================================================
def estandarizar_columnas(df):
    """
    Estandariza nombres de columnas: minusculas y sin espacios.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas estandarizadas.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('.', '_', regex=False)
    )
    return df


def convertir_edad_anos(df, col_edad='EDAD', col_unidad='UNI_MED'):
    """
    Convierte la edad a anos segun la unidad de medida SIVIGILA.
    UNI_MED: 1=Anos, 2=Meses, 3=Dias, 4=Horas, 5=Minutos

    Parameters
    ----------
    df : pd.DataFrame
    col_edad : str
    col_unidad : str

    Returns
    -------
    pd.Series
        Edad en anos.
    """
    edad = df[col_edad].copy().astype(float)
    unidad = df[col_unidad].copy().astype(float)

    edad_anos = pd.Series(np.nan, index=df.index)
    edad_anos[unidad == 1] = edad[unidad == 1]
    edad_anos[unidad == 2] = edad[unidad == 2] / 12
    edad_anos[unidad == 3] = edad[unidad == 3] / 365
    edad_anos[unidad == 4] = edad[unidad == 4] / (365 * 24)
    edad_anos[unidad == 5] = edad[unidad == 5] / (365 * 24 * 60)

    return edad_anos


def clasificar_grupo_etario(edad_anos):
    """
    Clasifica la edad en grupos etarios.

    Parameters
    ----------
    edad_anos : pd.Series

    Returns
    -------
    pd.Series
        Grupo etario como categoria.
    """
    bins = [0, 5, 15, 30, 45, 60, 120]
    labels = ['0-4', '5-14', '15-29', '30-44', '45-59', '60+']
    return pd.cut(edad_anos, bins=bins, labels=labels, right=False)


# ============================================================================
# Funciones de graficos
# ============================================================================
def grafico_barras_por_ano(df, col_ano='ANO', titulo='', ylabel='Casos',
                           ax=None, color_map=None):
    """Grafico de barras de conteo por ano."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    if color_map is None:
        color_map = COLORES_ANOS

    conteo = df[col_ano].value_counts().sort_index()
    colores = [color_map.get(a, '#333333') for a in conteo.index]
    conteo.plot(kind='bar', ax=ax, color=colores, edgecolor='white')

    ax.set_title(titulo, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Ano')
    ax.tick_params(axis='x', rotation=0)

    for i, v in enumerate(conteo.values):
        ax.text(i, v + v * 0.01, f'{v:,.0f}', ha='center', va='bottom',
                fontsize=9)

    return ax


def grafico_semana_epi(df, col_semana='SEMANA', col_ano='ANO', titulo='',
                       ax=None):
    """Curva epidemica por semana epidemiologica y ano."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    for ano in sorted(df[col_ano].unique()):
        datos_ano = df[df[col_ano] == ano]
        conteo = datos_ano[col_semana].value_counts().sort_index()
        color = COLORES_ANOS.get(ano, '#333333')
        ax.plot(conteo.index, conteo.values, label=str(ano), color=color,
                linewidth=2, marker='o', markersize=3)

    ax.set_title(titulo, fontweight='bold')
    ax.set_xlabel('Semana epidemiologica')
    ax.set_ylabel('Casos')
    ax.legend(title='Ano')
    ax.set_xlim(1, 53)

    return ax


def grafico_top_departamentos(df, col_dpto='Departamento_ocurrencia',
                              top_n=15, titulo='', ax=None):
    """Grafico de barras horizontales de los top departamentos."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    conteo = df[col_dpto].value_counts().head(top_n)
    conteo.sort_values().plot(kind='barh', ax=ax, color='steelblue',
                              edgecolor='white')

    ax.set_title(titulo, fontweight='bold')
    ax.set_xlabel('Casos')
    ax.set_ylabel('')

    return ax


def grafico_piramide_edad_sexo(df, col_edad_anos='edad_anos', col_sexo='SEXO',
                               titulo='', ax=None):
    """Piramide poblacional por edad y sexo."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Mapear sexo
    sexo_map = {1: 'Masculino', 'M': 'Masculino', 2: 'Femenino', 'F': 'Femenino'}
    df = df.copy()
    df['sexo_label'] = df[col_sexo].map(sexo_map)

    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 120]
    labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34',
              '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',
              '70-74', '75-79', '80+']
    df['grupo_edad'] = pd.cut(df[col_edad_anos], bins=bins, labels=labels,
                              right=False)

    hombres = df[df['sexo_label'] == 'Masculino']['grupo_edad'].value_counts().sort_index()
    mujeres = df[df['sexo_label'] == 'Femenino']['grupo_edad'].value_counts().sort_index()

    y = range(len(labels))
    ax.barh(y, -hombres.reindex(labels, fill_value=0).values, color='steelblue',
            label='Masculino', edgecolor='white')
    ax.barh(y, mujeres.reindex(labels, fill_value=0).values, color='salmon',
            label='Femenino', edgecolor='white')

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(titulo, fontweight='bold')
    ax.set_xlabel('Casos')
    ax.legend()

    max_val = max(hombres.max(), mujeres.max()) if len(hombres) > 0 and len(mujeres) > 0 else 100
    ax.set_xlim(-max_val * 1.1, max_val * 1.1)

    return ax


def grafico_heatmap_dpto_ano(df, col_dpto='Departamento_ocurrencia',
                             col_ano='ANO', top_n=20, titulo='', ax=None):
    """Heatmap de casos por departamento y ano."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    # Top departamentos por total de casos
    top_dptos = df[col_dpto].value_counts().head(top_n).index
    df_filtrado = df[df[col_dpto].isin(top_dptos)]

    tabla = pd.crosstab(df_filtrado[col_dpto], df_filtrado[col_ano])
    tabla = tabla.loc[top_dptos]  # Mantener orden por total

    sns.heatmap(tabla, annot=True, fmt=',d', cmap='YlOrRd', ax=ax,
                linewidths=0.5)
    ax.set_title(titulo, fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('Ano')

    return ax


def resumen_dataframe(df, nombre=''):
    """Imprime resumen rapido de un DataFrame."""
    print(f"\n{'='*60}")
    print(f"  Resumen: {nombre}")
    print(f"{'='*60}")
    print(f"  Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"  Memoria: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    nulos = df.isnull().sum()
    cols_nulos = nulos[nulos > 0]
    if len(cols_nulos) > 0:
        print(f"  Columnas con nulos: {len(cols_nulos)}")
        for col, n in cols_nulos.items():
            pct = n / len(df) * 100
            if pct > 5:
                print(f"    - {col}: {n:,} ({pct:.1f}%)")
    else:
        print("  Sin valores nulos")
    print()
