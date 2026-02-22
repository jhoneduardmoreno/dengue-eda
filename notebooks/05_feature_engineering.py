#!/usr/bin/env python
# coding: utf-8

# # 05 - Feature Engineering: Panel Municipal-Mensual
# 
# **Proyecto:** EDA de Dengue en Colombia  
# **Maestria en Inteligencia Artificial** - Desarrollo de Soluciones  
# 
# Este notebook construye un dataset panel a nivel **municipio-mes** integrando:
# - Conteo y metricas de casos de dengue (regular + grave)
# - Variables climaticas departamentales
# - Poblacion DANE
# - Variable objetivo (exceso epidemico)
# - Features con rezagos temporales (lags)
# 
# **Exporta:** `data/processed/panel_municipal_mensual.parquet` y `.csv`

# In[1]:


import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join('..', 'src'))
from utils import (
    cargar_dengue, cargar_dane, cargar_clima,
    convertir_edad_anos, resumen_dataframe,
    ANOS_ESTUDIO, PROJECT_ROOT
)

pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 100)

print(f'Anos de estudio: {ANOS_ESTUDIO}')


# ## 1. Carga y preprocesamiento de casos de dengue

# In[2]:


# Cargar dengue regular y grave
print('--- Dengue Regular (210) ---')
df_regular = cargar_dengue(tipo='regular')
df_regular['tipo_dengue'] = 'regular'

print('\n--- Dengue Grave (220) ---')
df_grave = cargar_dengue(tipo='grave')
df_grave['tipo_dengue'] = 'grave'

# Concatenar
df = pd.concat([df_regular, df_grave], ignore_index=True)
print(f'\nTotal combinado: {len(df):,} registros')
print(f'  Regular: {len(df_regular):,}')
print(f'  Grave: {len(df_grave):,}')


# In[ ]:


# Convertir fechas
cols_fecha = ['FEC_NOT', 'INI_SIN', 'FEC_HOS', 'FEC_DEF', 'FECHA_NTO']
for col in cols_fecha:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Calcular edad en anos
df['edad_anos'] = convertir_edad_anos(df)

# Eliminar duplicados por CONSECUTIVE
antes = len(df)
df = df.drop_duplicates(subset=['CONSECUTIVE'], keep='first')
print(f'Duplicados eliminados: {antes - len(df):,}')

# Extraer ano y mes desde INI_SIN (inicio de sintomas)
nulos_ini_sin = df['INI_SIN'].isnull().sum()
print(f'Registros con INI_SIN nulo: {nulos_ini_sin}')
df = df.dropna(subset=['INI_SIN'])

df['anio_sin'] = df['INI_SIN'].dt.year
df['mes_sin'] = df['INI_SIN'].dt.month

# Estandarizar codigos municipales a string de 5 digitos
df['cod_mun_n_str'] = df['COD_MUN_N'].astype(int).astype(str).str.zfill(5)
df['cod_dpto_n_str'] = df['COD_DPTO_N'].astype(int).astype(str).str.zfill(2)

# Corregir variantes de nombres de municipios
df['Municipio_notificacion'] = df['Municipio_notificacion'].replace({
    'SANTA MARTHA': 'SANTA MARTA',
})

# Filtrar solo anos de estudio
df = df[df['anio_sin'].isin(ANOS_ESTUDIO)].copy()
print(f'\nRegistros tras filtrar anos {ANOS_ESTUDIO}: {len(df):,}')
print(f'Registros por ano:')
print(df['anio_sin'].value_counts().sort_index().to_string())


# ## 2. Agregacion municipal-mensual

# In[ ]:


# Crear indicadores para la agregacion
df['es_grave'] = (df['tipo_dengue'] == 'grave').astype(int)
df['es_regular'] = (df['tipo_dengue'] == 'regular').astype(int)
df['es_hospitalizado'] = (df['PAC_HOS'] == 1).astype(int)
df['es_fallecido'] = (df['CON_FIN'] == 2).astype(int)
df['es_femenino'] = (df['SEXO'] == 'F').astype(int)
df['es_masculino'] = (df['SEXO'] == 'M').astype(int)

# Resolver municipios con multiples nombres de notificacion:
# Quedarse con el nombre mas frecuente por cod_mun_n_str
nombres_mun = (
    df.groupby(['cod_dpto_n_str', 'cod_mun_n_str', 'Departamento_Notificacion', 'Municipio_notificacion'])
    .size().reset_index(name='n')
    .sort_values('n', ascending=False)
    .drop_duplicates(subset=['cod_mun_n_str'], keep='first')
    .drop(columns='n')
)
print(f'Municipios unicos por cod_mun_n_str: {len(nombres_mun)}')

# Agregar por municipio-mes (solo por codigos, no por nombres)
panel = df.groupby(['cod_mun_n_str', 'cod_dpto_n_str', 'anio_sin', 'mes_sin']).agg(
    casos_total=('CONSECUTIVE', 'count'),
    casos_regular=('es_regular', 'sum'),
    casos_grave=('es_grave', 'sum'),
    hospitalizaciones=('es_hospitalizado', 'sum'),
    fallecidos=('es_fallecido', 'sum'),
    edad_media=('edad_anos', 'mean'),
    n_femenino=('es_femenino', 'sum'),
    n_masculino=('es_masculino', 'sum'),
).reset_index()

# Renombrar columnas de periodo
panel = panel.rename(columns={'anio_sin': 'ano', 'mes_sin': 'mes'})

# Agregar nombres de departamento y municipio
panel = panel.merge(nombres_mun, on=['cod_dpto_n_str', 'cod_mun_n_str'], how='left')

# Proporciones derivadas
panel['prop_grave'] = panel['casos_grave'] / panel['casos_total']
panel['prop_hospitalizado'] = panel['hospitalizaciones'] / panel['casos_total']
panel['prop_femenino'] = panel['n_femenino'] / panel['casos_total']

print(f'Panel agregado: {panel.shape[0]:,} filas x {panel.shape[1]} columnas')
print(f'Municipios unicos: {panel["cod_mun_n_str"].nunique()}')
print(f'Periodos: {panel[["ano","mes"]].drop_duplicates().shape[0]}')
panel.head()


# In[5]:


# Verificar que el total de casos coincide con el original
total_panel = panel['casos_total'].sum()
total_original = len(df)
print(f'Total casos en panel: {total_panel:,}')
print(f'Total casos original: {total_original:,}')
print(f'Coinciden: {total_panel == total_original}')


# ## 3. Panel completo con ceros (cross join municipios x periodos)

# In[ ]:


# Municipios unicos con sus nombres y departamentos
municipios = nombres_mun.copy()
print(f'Municipios unicos con al menos 1 caso: {len(municipios)}')

# Todos los periodos
periodos = pd.DataFrame([
    {'ano': a, 'mes': m}
    for a in ANOS_ESTUDIO
    for m in range(1, 13)
])
print(f'Periodos totales: {len(periodos)} (5 anos x 12 meses)')

# Cross join
municipios['_key'] = 1
periodos['_key'] = 1
panel_completo = municipios.merge(periodos, on='_key').drop(columns='_key')
print(f'Panel completo (cross join): {len(panel_completo):,} filas')

# Left join con datos agregados
merge_cols = ['cod_dpto_n_str', 'cod_mun_n_str',
              'Departamento_Notificacion', 'Municipio_notificacion',
              'ano', 'mes']
panel_completo = panel_completo.merge(panel, on=merge_cols, how='left')

# Rellenar NaN con 0 en columnas de conteo
cols_conteo = ['casos_total', 'casos_regular', 'casos_grave',
               'hospitalizaciones', 'fallecidos',
               'n_femenino', 'n_masculino']
panel_completo[cols_conteo] = panel_completo[cols_conteo].fillna(0).astype(int)

# Proporciones: rellenar con 0 donde no hay casos
cols_prop = ['prop_grave', 'prop_hospitalizado', 'prop_femenino']
panel_completo[cols_prop] = panel_completo[cols_prop].fillna(0)

# edad_media: dejar NaN donde no hubo casos (no tiene sentido poner 0)

print(f'\nPanel completo final: {len(panel_completo):,} filas x {panel_completo.shape[1]} columnas')
print(f'Verificacion: casos_total sum = {panel_completo["casos_total"].sum():,}')

# Verificar no hay duplicados en (municipio, ano, mes)
dup_check = panel_completo.duplicated(subset=['cod_mun_n_str', 'ano', 'mes'], keep=False).sum()
print(f'Duplicados en (municipio, ano, mes): {dup_check}')

panel = panel_completo
del panel_completo


# ## 4. Integrar datos climaticos

# In[7]:


# Cargar clima (nivel departamento-mes)
df_clima = cargar_clima()
if df_clima is not None:
    print(f'Datos climaticos: {df_clima.shape[0]:,} filas x {df_clima.shape[1]} columnas')
    print(f'Columnas: {list(df_clima.columns)}')
    print(f'\nDepartamentos unicos en clima: {df_clima["departamento"].nunique()}')
    print(df_clima['departamento'].unique())
    print(f'\nDepartamentos unicos en panel: {panel["Departamento_Notificacion"].nunique()}')
    print(sorted(panel['Departamento_Notificacion'].unique()))


# In[ ]:


# Normalizar nombres de departamento para el merge
import unicodedata

def normalizar_texto(s):
    """Quita acentos y pasa a UPPER."""
    if not isinstance(s, str):
        return s
    s = s.upper().strip()
    # Quitar acentos
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

# Normalizar en ambos datasets
panel['dpto_norm'] = panel['Departamento_Notificacion'].apply(normalizar_texto)
df_clima['dpto_norm'] = df_clima['departamento'].apply(normalizar_texto)

# Mapeo de discrepancias conocidas (panel → clima)
mapeo_dpto = {
    'BOGOTA D.C.': 'CUNDINAMARCA',
    'BOGOTA': 'CUNDINAMARCA',
    'VALLE': 'VALLE DEL CAUCA',
    'LA GUAJIRA': 'GUAJIRA',
    'STA MARTA D.E.': 'MAGDALENA',
    'CARTAGENA': 'BOLIVAR',
    'BARRANQUILLA': 'ATLANTICO',
    'BUENAVENTURA': 'VALLE DEL CAUCA',
    'NORTE SANTANDER': 'NORTE DE SANTANDER',
    'SAN ANDRES': 'SAN ANDRES Y PROVIDENCIA',
}

panel['dpto_clima_key'] = panel['dpto_norm'].replace(mapeo_dpto)

# Verificar cobertura antes del merge
dptos_clima = set(df_clima['dpto_norm'].unique())
dptos_panel = set(panel['dpto_clima_key'].unique())

sin_clima = dptos_panel - dptos_clima
print(f'Departamentos en panel sin match en clima: {len(sin_clima)}')
if sin_clima:
    print(f'  {sin_clima}')


# In[9]:


# Left join panel con clima
cols_clima = ['temperatura_c', 'precipitacion_mm', 'ndvi', 'dewpoint_c']

df_clima_merge = df_clima.rename(columns={'dpto_norm': 'dpto_clima_key'})

panel = panel.merge(
    df_clima_merge[['dpto_clima_key', 'ano', 'mes'] + cols_clima],
    on=['dpto_clima_key', 'ano', 'mes'],
    how='left'
)

# Cobertura
cobertura_clima = panel[cols_clima[0]].notna().mean() * 100
print(f'Cobertura de clima: {cobertura_clima:.1f}%')
print(f'Filas sin datos climaticos: {panel[cols_clima[0]].isna().sum():,}')

# Verificar dimensiones
print(f'\nPanel tras merge con clima: {panel.shape[0]:,} filas x {panel.shape[1]} columnas')


# ## 5. Integrar poblacion DANE

# In[10]:


# Cargar DANE
df_dane = cargar_dane()
print(f'DANE: {df_dane.shape[0]:,} municipios')

# Fix: cod_mpio viene como '5001.0' (float → str)
df_dane['cod_mpio'] = df_dane['cod_mpio'].astype(str).str.replace('.0', '', regex=False).str.zfill(5)
print(f'Ejemplo cod_mpio: {df_dane["cod_mpio"].head().tolist()}')

# Melt a formato largo
pob_cols = [c for c in df_dane.columns if c.startswith('pob_')]
df_pob = df_dane.melt(
    id_vars=['cod_mpio'],
    value_vars=pob_cols,
    var_name='ano_pob',
    value_name='poblacion'
)
df_pob['ano_pob'] = df_pob['ano_pob'].str.replace('pob_', '').astype(int)

# Mapeo de anos de estudio a anos de poblacion disponibles
mapeo_ano_pob = {2010: 2010, 2016: 2016, 2019: 2019, 2022: 2020, 2024: 2020}
panel['ano_pob'] = panel['ano'].map(mapeo_ano_pob)

print(f'\nMapeo de anos para poblacion: {mapeo_ano_pob}')
print(f'Poblacion en formato largo: {df_pob.shape[0]:,} filas')


# In[11]:


# Left join con panel
panel = panel.merge(
    df_pob[['cod_mpio', 'ano_pob', 'poblacion']],
    left_on=['cod_mun_n_str', 'ano_pob'],
    right_on=['cod_mpio', 'ano_pob'],
    how='left'
).drop(columns=['cod_mpio'])

# Calcular tasa de incidencia por 100,000 habitantes
panel['tasa_incidencia'] = np.where(
    panel['poblacion'] > 0,
    panel['casos_total'] / panel['poblacion'] * 100_000,
    np.nan
)

cobertura_pob = panel['poblacion'].notna().mean() * 100
print(f'Cobertura de poblacion: {cobertura_pob:.1f}%')
print(f'Filas sin poblacion: {panel["poblacion"].isna().sum():,}')
print(f'\nPanel tras merge con poblacion: {panel.shape[0]:,} filas x {panel.shape[1]} columnas')


# ## 6. Variable objetivo: exceso epidemico

# In[12]:


# Calcular media y std historica de casos por municipio
stats_mun = panel.groupby('cod_mun_n_str')['casos_total'].agg(['mean', 'std']).reset_index()
stats_mun.columns = ['cod_mun_n_str', 'media_hist', 'std_hist']
stats_mun['std_hist'] = stats_mun['std_hist'].fillna(0)

# Umbral de exceso: media + 2*std
stats_mun['umbral_exceso'] = stats_mun['media_hist'] + 2 * stats_mun['std_hist']

panel = panel.merge(stats_mun, on='cod_mun_n_str', how='left')

# Exceso: 1 si casos > umbral
# Para municipios con std=0 (casi siempre 0 casos): cualquier caso > media es exceso
panel['exceso'] = np.where(
    panel['std_hist'] > 0,
    (panel['casos_total'] > panel['umbral_exceso']).astype(int),
    (panel['casos_total'] > panel['media_hist']).astype(int)
)

# Version basada en tasa de incidencia
stats_tasa = panel.groupby('cod_mun_n_str')['tasa_incidencia'].agg(['mean', 'std']).reset_index()
stats_tasa.columns = ['cod_mun_n_str', 'media_tasa_hist', 'std_tasa_hist']
stats_tasa['std_tasa_hist'] = stats_tasa['std_tasa_hist'].fillna(0)
stats_tasa['umbral_exceso_tasa'] = stats_tasa['media_tasa_hist'] + 2 * stats_tasa['std_tasa_hist']

panel = panel.merge(stats_tasa, on='cod_mun_n_str', how='left')

panel['exceso_tasa'] = np.where(
    panel['std_tasa_hist'] > 0,
    (panel['tasa_incidencia'] > panel['umbral_exceso_tasa']).astype(int),
    (panel['tasa_incidencia'] > panel['media_tasa_hist']).astype(int)
)
# Si tasa_incidencia es NaN, exceso_tasa = 0
panel['exceso_tasa'] = panel['exceso_tasa'].fillna(0).astype(int)

print(f'Prevalencia de exceso (casos): {panel["exceso"].mean()*100:.1f}%')
print(f'Prevalencia de exceso (tasa):  {panel["exceso_tasa"].mean()*100:.1f}%')
print(f'\nDistribucion exceso:')
print(panel['exceso'].value_counts().to_string())


# ## 7. Lag features (rezagos temporales)

# In[13]:


# Variables a rezagar
vars_lag = ['temperatura_c', 'precipitacion_mm', 'ndvi', 'dewpoint_c',
            'casos_total', 'tasa_incidencia']
lags = [1, 2, 3]

# Ordenar por municipio, ano, mes
panel = panel.sort_values(['cod_mun_n_str', 'ano', 'mes']).reset_index(drop=True)

# Lags DENTRO de cada ano por municipio (no cruzar limites de ano)
for var in vars_lag:
    for lag in lags:
        col_name = f'{var}_lag{lag}'
        panel[col_name] = panel.groupby(['cod_mun_n_str', 'ano'])[var].shift(lag)

# Media movil de 3 meses (within-year)
for var in vars_lag:
    col_name = f'{var}_mm3'
    panel[col_name] = panel.groupby(['cod_mun_n_str', 'ano'])[var].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

# Contar features nuevos
cols_lag = [c for c in panel.columns if '_lag' in c]
cols_mm = [c for c in panel.columns if '_mm3' in c]
print(f'Features de lag creados: {len(cols_lag)}')
print(f'Features de media movil: {len(cols_mm)}')
print(f'Total features temporales: {len(cols_lag) + len(cols_mm)}')
print(f'\nPanel final: {panel.shape[0]:,} filas x {panel.shape[1]} columnas')


# ## 8. Validacion y exportacion

# In[ ]:


# Verificar no duplicados en (municipio, ano, mes) - post lags
dup = panel.duplicated(subset=['cod_mun_n_str', 'ano', 'mes'], keep=False).sum()
print(f'Duplicados en (municipio, ano, mes): {dup}')
assert dup == 0, f'HAY {dup} DUPLICADOS - revisar pipeline'

# Resumen de NaN
print(f'\nResumen de NaN por columna:')
nans = panel.isnull().sum()
nans_pct = (nans / len(panel) * 100).round(1)
nan_summary = pd.DataFrame({'nulos': nans, 'pct': nans_pct})
nan_summary = nan_summary[nan_summary['nulos'] > 0].sort_values('pct', ascending=False)
print(nan_summary.to_string())

# Estadisticas descriptivas
print(f'\nEstadisticas descriptivas del panel:')
print(panel.describe().T.to_string())


# In[15]:


# Eliminar columnas auxiliares de merge
cols_drop = ['dpto_norm', 'dpto_clima_key', 'ano_pob']
panel = panel.drop(columns=[c for c in cols_drop if c in panel.columns])

# Crear directorio de salida si no existe
output_dir = PROJECT_ROOT / 'data' / 'processed'
output_dir.mkdir(parents=True, exist_ok=True)

# Exportar
panel.to_parquet(output_dir / 'panel_municipal_mensual.parquet', index=False)
panel.to_csv(output_dir / 'panel_municipal_mensual.csv', index=False)

print(f'Panel exportado exitosamente:')
print(f'  {output_dir / "panel_municipal_mensual.parquet"}')
print(f'  {output_dir / "panel_municipal_mensual.csv"}')
print(f'\nDimensiones finales: {panel.shape[0]:,} filas x {panel.shape[1]} columnas')
print(f'Columnas: {list(panel.columns)}')

resumen_dataframe(panel, 'Panel Municipal-Mensual Final')

