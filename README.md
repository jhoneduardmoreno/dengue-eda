# Análisis Exploratorio de Datos: Dengue en Colombia

Análisis exploratorio de datos (EDA) sobre la incidencia de dengue y dengue grave en Colombia, integrando datos epidemiológicos del SIVIGILA, proyecciones poblacionales del DANE y variables climáticas obtenidas de Google Earth Engine.

## Contexto académico

Proyecto desarrollado en el marco de la **Maestría en Inteligencia Artificial** de la **Universidad Autónoma de Occidente**, para la materia **Desarrollo de Soluciones**.

## Autores

- Jhon Edwar Salazar
- Santiago Castaño Orozco
- David Alejandro Burbano Getial

## Fuentes de datos

| Fuente | Descripción |
|--------|-------------|
| **SIVIGILA** | Casos de dengue (evento 210) y dengue grave (evento 220) reportados al sistema de vigilancia en salud pública |
| **DANE** | Proyecciones de población municipal para el cálculo de tasas de incidencia |
| **Google Earth Engine** | Variables climáticas (temperatura, precipitación, humedad) agregadas a nivel departamental |

## Años de estudio

2010, 2016, 2019, 2022 y 2024.

## Estructura del repositorio

```
dengue-eda/
├── data/
│   ├── clima/              # Variables climáticas por año (CSV)
│   ├── dane/               # Proyecciones poblacionales (XLS)
│   ├── dengue/             # Casos de dengue regular - evento 210 (XLSX)
│   └── dengue fuerte/      # Casos de dengue grave - evento 220 (XLS/XLSX)
├── notebooks/
│   ├── 00_descarga_clima_gee.ipynb
│   ├── 01_carga_y_limpieza.ipynb
│   ├── 02_eda_dengue.ipynb
│   ├── 03_eda_dengue_grave.ipynb
│   └── 04_eda_comparativo.ipynb
├── src/
│   ├── generar_reporte.py   # Generación automática de reporte PDF
│   └── utils.py             # Funciones auxiliares compartidas
└── results_graphs/
    ├── comparative/         # Gráficos comparativos dengue regular vs grave
    └── *.png                # Gráficos individuales por tipo de análisis
```
