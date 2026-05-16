# Mejoras al proyecto de dengue tras la reunión con Carlos (2026-04-24)

> Documento de planeación derivado de la sesión de seguimiento con el director del proyecto. Ver transcripción completa en [`reuniones/2026-04-24_seguimiento_carlos.md`](reuniones/2026-04-24_seguimiento_carlos.md).

## Contexto

El modelo actual entrena un único clasificador sobre los 993 municipios del país. Resultado en test (2024): **Precision 43% / Recall 88% / ROC-AUC 0.92**. Carlos lo calificó como un modelo *"alarmista"*: detecta casi todos los excesos reales, pero genera demasiadas falsas alarmas. La causa principal es que un solo modelo no puede capturar la heterogeneidad climática, cultural y socioeconómica de Colombia.

Este documento lista los cambios concretos a implementar.

---

## 🎯 Cambio estructural #1 — Enfoque regional, no nacional

**Problema actual:** un único modelo con los 993 municipios.

**Mejora:** entrenar **un modelo independiente por municipio**, sobre los 5–10 municipios con mayor **incidencia × 100.000 hab**.

**Municipios foco** (top de incidencia según `Selección geografía.docx`):

| # | Departamento | Municipio | Incidencia (×100k) |
|---|---|---|---|
| 1 | Córdoba | Valencia | 1.306 |
| 2 | Magdalena | Fundación | 959 |
| 3 | Guaviare | El Retorno | 899 |
| 4 | Cesar | San Alberto | 885 |
| 5 | Córdoba | Pueblo Nuevo | 834 |
| 6 | Huila | Aipe | 736 |
| 7 | Guaviare | San José del Guaviare | 686 |
| 8 | Atlántico | Polo Nuevo | 678 |
| 9 | Cesar | Agustín Codazzi | 606 |
| 10 | N. Santander | Villa del Rosario | 524 |

**Justificación de Carlos:** *"asuman que cada unidad espacial va a ser distinta y el modelo va a ajustarse de manera diferente"*. Trabajar por incidencia (no por casos absolutos) evita el sesgo de "donde hay más gente hay más casos" y prioriza comunidades vulnerables.

---

## 📅 Cambio estructural #2 — Partición cronológica del test

**Problema actual:** Train = 2010/2016/2019/2022 (años discretos) · Test = 2024. Hay riesgo de fuga temporal y no evalúa capacidad de predecir el *futuro*.

**Mejora:**
- Split temporal estricto: **train = primer 80% de los meses · test = último 20%** (≈ últimos 3–4 años).
- Validar `max(fecha_train) < min(fecha_test)`.
- Para validación interna usar `sklearn.model_selection.TimeSeriesSplit`, **nunca K-Fold aleatorio**.

**Justificación:** *"Cuando hago la partición, normalmente lo que vamos a hacer es querer predecir de aquí en adelante. Por eso muchas veces lo que se hace es coger el último 20% de los datos como el test"* — Carlos.

---

## 📊 Cambio estructural #3 — Serie temporal continua

**Problema actual:** solo 5 años discretos (2010/2016/2019/2022/2024).

**Mejora:**
- Ampliar el panel a **2007–2024 continuo, mensual**.
- Re-descargar clima de Google Earth Engine para los años faltantes.
- Reprocesar SIVIGILA para todos los años en el rango.
- Regenerar `data/processed/panel_municipal_mensual.parquet`.

---

## 📈 Cambio en métricas

**Mejora:** priorizar **Precision y Recall** sobre el test como métricas de decisión.

| Métrica | Uso | Prioridad |
|---|---|---|
| Accuracy | Reportar, no decisión | Baja |
| **Precision** | **Decisión principal** | **Alta** |
| **Recall** | **Decisión principal** | **Alta** |
| F1 | Complemento | Media |
| ROC-AUC | Complemento | Media |

**Regla agregación entre municipios:** si se reportan métricas globales, **sumar TP/FP/TN/FN** antes de calcular Precision/Recall, no promediar las métricas individuales.

---

## 🧹 Limpieza de modelos

- ❌ **Eliminar Random Forest** del comparativo final (Recall 2.4% — inservible para la clase minoritaria).
- ✅ **Mantener Regresión Logística (regularizada) + XGBoost** como baseline definitivo.
- ⚠️ **No invertir tiempo en deep learning** (LSTM/TFT/Transformers):
  - Solo ≈ 780 datos semanales por municipio (15 años × 52 semanas).
  - Un LSTM típico (hidden_size = 64) ya tiene >200 parámetros → overfitting casi garantizado.
  - Si insisten en probarlo, usar `hidden_size` ≤ 8.

---

## 🚫 Decisiones explícitas de NO hacer

| Decisión | Justificación |
|---|---|
| No esperar datos SIVIGILA 2025 | No se espera que estén publicados antes del cierre del proyecto |
| No integrar API en vivo de Google Earth Engine en el despliegue | Es prototipo; entrada por Excel ya procesado es suficiente |
| No incluir camas hospitalarias del DANE como feature | El propio análisis previo del grupo mostró que no mejora la explicación estadística |
| No usar redes neuronales profundas como modelo final | Pocos datos para aprovecharlas, overfitting esperado |
| No usar K-Fold aleatorio sobre meses | Genera fuga temporal en series de tiempo |

---

## ✅ Lo que se conserva (ya está bien)

- Ingeniería de features: lags climáticos (1–3 meses), medias móviles de 3 meses, variables del ciclo del mosquito y período de incubación.
- Manejo de desbalanceo: `class_weight='balanced'` para LogReg, `scale_pos_weight` para XGBoost (recalcular por municipio).
- Estructura del panel mensual por municipio.
- Pipeline de descarga (GEE) y limpieza (SIVIGILA + DANE).

---

## 📋 Plan de ejecución sugerido

| # | Tarea | Archivos afectados |
|---|---|---|
| 1 | Ampliar descarga de clima GEE a 2007–2024 continuo | `notebooks/00_descarga_clima_gee.ipynb` |
| 2 | Cargar SIVIGILA todos los años en el rango | `notebooks/01_carga_y_limpieza.ipynb` |
| 3 | Regenerar `panel_municipal_mensual.parquet` | `notebooks/05_feature_engineering.ipynb` |
| 4 | Definir lista `MUNICIPIOS_FOCO` con códigos DIVIPOLA | `notebooks/06_modelado.ipynb` |
| 5 | Refactor: loop por municipio + split cronológico (80/20) | `notebooks/06_modelado.ipynb` |
| 6 | Quitar Random Forest del comparativo | `notebooks/06_modelado.ipynb` |
| 7 | Reportar Precision/Recall + matriz de confusión por municipio | `notebooks/06_modelado.ipynb` |
| 8 | Guardar artefactos por municipio: `{cod_mun}_{modelo}.joblib` | `models/` |
| 9 | Adaptar reporte a estructura por municipio | `src/generar_reporte_modelado.py` |
| 10 | Dashboard prototipo con selector de municipio | (a definir) |

---

## 🎯 Objetivo cuantitativo

En al menos **la mitad de los municipios foco**, sobre el test cronológico:
- **Precision > 43%** (baseline nacional actual).
- **Recall > 60%** mantenido.

---

## Referencias

- Reunión transcrita: [`reuniones/2026-04-24_seguimiento_carlos.md`](reuniones/2026-04-24_seguimiento_carlos.md)
- Análisis de selección geográfica del grupo: `Selección geografía.docx` (local)
- Notebook actual de modelado: `notebooks/06_modelado.ipynb`
- Marco Geoestadístico Nacional — DANE (referencia opcional para futuras extensiones)
