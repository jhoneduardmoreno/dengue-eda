# Seguimiento de avance del proyecto de dengue con Carlos

- **Fecha:** 2026-04-24
- **Participantes:** Jhon Edwar Moreno Díaz (y miembros del grupo: Hernán, Ruby, Danilo), Carlos (director), grupos de Andrés y Amaya.
- **Objetivo:** Hacer seguimiento al informe de avance, resolver dudas y recibir retroalimentación para encaminar el producto final.

---

## Resumen ejecutivo (puntos clave de la retroalimentación de Carlos)

1. **Enfocar el modelado en una región específica** (uno o pocos departamentos/municipios), no en todo el país, debido a la enorme heterogeneidad climática, biológica, cultural y socioeconómica de Colombia.
2. **Entrenar modelos independientes por unidad espacial** (un modelo por municipio/departamento), en lugar de un modelo general con todas las regiones.
3. **Partición temporal de los datos** (no aleatoria): usar el último ~20% de la serie como test (p. ej. 2019–2024) para evaluar realmente la capacidad predictiva hacia el futuro.
4. **Métricas recomendadas para clasificación binaria (exceso / no exceso):** principalmente **Recall** y **Precision** sobre el conjunto de test. Accuracy es útil reportarlo, pero menos confiable con desbalanceo.
5. **Modelos suficientes:** regresión logística (con regularización), árboles (XGBoost) son una baseline excelente. Deep learning (LSTM, TFT, Transformers) es opcional y suele *no* aportar con tan pocos datos.
6. **Despliegue / dashboard como prototipo:** priorizar tener un entregable funcional simple (entrada por tabla Excel ya procesada) sobre integraciones avanzadas con APIs (Google Earth Engine, etc.).
7. **Datos 2025 no disponibles:** trabajar hasta 2024 está bien; si llegan los de 2025 antes del cierre, usarlos como un segundo test, pero no es requisito.

---

## Discusión por grupo

### Grupo 1 — Jhon Edwar, Hernán, Ruby, Danilo

**Lo que presentaron:**
- Datos de todas las regiones del país.
- Años de entrenamiento seleccionados de forma representativa (≈ 2007, 2008, 2010, 2012).
- Año de prueba: **2024**.
- Modelos: Regresión Logística, Random Forest, XGBoost.
- Métrica priorizada: **Recall** (por tratarse de datos desbalanceados, con ponderación de clase negativa).
- Mejor modelo según recall: **Regresión Logística**.

**Preguntas planteadas a Carlos:**
- ¿Aplicar técnicas de balanceo en los datos?
- ¿Probar modelos temporales?
- ¿Enfocarse en regiones específicas?
- ¿Conviene usar todos los años en lugar de años representativos?

**Feedback de Carlos:**

- **Matriz de confusión actual:** la precisión real para "exceso" es ~43% y el recall ~87%. Es un modelo *alarmista*: detecta la mayoría de los excesos reales, pero también genera muchas falsas alarmas. Esto se debe principalmente a que se está mezclando toda la heterogeneidad espacial del país.
- **Recomendación principal:** **enfocarse en una única región** (un departamento, un municipio, o un conjunto pequeño de municipios) y/o entrenar modelos independientes por región. Esto evita tener que modelar explícitamente la heterogeneidad espacial.
- **Usar toda la serie de años, no solo años "representativos".**
- **Modelos:** los actuales (regresión logística regularizada, XGBoost) son **suficientes y meritorios**. Deep learning (RNN, Transformers) opcional como extra.
- **Partición de datos en series de tiempo:** **no se hace al azar**. Se debe coger el último ~20% como test. Ejemplo: entrenar 2007–2018 y testear 2019–2024.
- **Métricas a reportar:**
  - **Accuracy:** útil pero sesgada por desbalanceo. Reportarla pero no confiar plenamente.
  - **Precision** (TP / (TP + FP)): "cuando el modelo dice que hay exceso, ¿qué tan confiable es?"
  - **Recall** (TP / (TP + FN)): "de todos los excesos reales, ¿cuántos detecta el modelo?"
  - Las **dos clave** son **Precision y Recall** sobre el test.

**Pregunta de Hernán — ¿cómo elegir la región?**
- **Opción A:** lugares con **mayor número absoluto de casos** (Ibagué, Girardot, etc.). Útil para mitigar el mayor número de casos.
- **Opción B:** lugares con **mayor incidencia por habitante** (casos / población). Útil para enfocarse en comunidades vulnerables y de mayor riesgo (Amazonas, Vichada, etc.). Evita el sesgo de "solo donde hay más gente hay más casos".
- **Opción C:** lugares de **interés particular** (por una motivación específica, comunidad olvidada, etc.) — también válido.
- **Recurso recomendado:** *Marco Geoestadístico Nacional* del **DANE** — contiene datos georreferenciados de hospitales por municipio, útil para cruzar con disponibilidad hospitalaria.

**Pregunta sobre Transformers (curiosidad de Jhon):**
- Los Transformers "clásicos" ya están siendo reemplazados/evolucionados en arquitecturas más complejas (apilamiento de capas de atención, múltiples cabezas, atención cruzada, enmascarada, etc.). La filosofía sigue vigente, pero las arquitecturas modernas (GPT-3+, multimodales) son mucho más complejas. Referencia visual: `bbycroft.net` (visualizador de LLMs).

---

### Grupo de Andrés

**Lo que presentaron / preguntaron:**
- Estado de datos 2025: aún no publicados en SIVIGILA.
- Avance en integraciones con Google Earth Engine y el Marco Geoestadístico Nacional.
- Modelos en uso: **Poisson** y **LightGBM** (LightGBM con mejores resultados; posible uso solo para explicabilidad).
- Exploración de **LSTM** y **TFT** (Temporal Fusion Transformer) en PyTorch — sin superar a LightGBM.
- Tres municipios trabajados: Ibagué, Girardot, Espinal.

**Feedback de Carlos:**

- **Datos 2025:** no esperarlos. Usar hasta 2024 es lo esperado por el proyecto. Si llegan, usarlos como segundo test.
- **Despliegue:** mantenerlo simple como **prototipo**. La entrada debe ser la **tabla Excel ya procesada**, no APIs en vivo de Google Earth Engine. Lo de la API es un extra, no requisito.
- **Deep learning con pocos datos — explicación clave:**
  - 2007–2022 entrenamiento ≈ 15 años × 52 semanas ≈ **780 datos**.
  - Un LSTM con hidden_size = 64 ya tiene >200 parámetros → **overfitting casi garantizado** con tan pocos datos.
  - Recomendación: bajar `hidden_size` a **2, 4 u 8** si insisten en deep learning.
  - El grupo confirma: con hidden_size 250–500, train mejora pero test se degrada (overfit, agravado por 2024 atípico).
  - El deep learning brillaría si tuvieran ~30 años a escala diaria, o muchos municipios × semanas (~702.000 datos).
- **Modelos independientes por municipio:**
  - Idealmente, **un modelo entrenado por cada municipio** (mismos hiperparámetros pueden cambiar entre municipios). Cada municipio = unidad independiente.
  - Andrés aclara que entrenan con todos los municipios e infieren sobre tres — Carlos sugiere idealmente independizarlos, pero **si ya está avanzado el pipeline, no echar atrás**.
- **Métrica agregada entre municipios:** si se trabajan como un solo bloque, **sumar** los errores en lugar de promediarlos (es más coherente cuando se tratan como una sola unidad).
- **Prioridad:** terminar primero un entregable funcional (aunque sea borrador), luego pulir.

---

### Grupo de Amaya

**Lo que presentaron:**
- Dos modelos comparativos:
  1. Modelo **nacional** (~900 municipios).
  2. Modelo con **3 departamentos** representativos.
- Incluyeron ciclo de vida del mosquito y período de incubación de la enfermedad como variables.
- Resultados: **el modelo de 3 departamentos da mejores métricas** (Recall, Precision, F1, ROC) que el nacional.

**Feedback de Carlos:**

- Es un fenómeno típico de **escala espacial**: a escala nacional, las variables relevantes se promedian y se pierden las dinámicas locales (clima, cultura, comportamiento). A escala departamental, ciertas variables (sociales, climáticas locales) cobran más peso.
- **Recomendación:** quedarse con el modelo a escala departamental (el que está dando mejor desempeño) para construir el despliegue.
- Sería interesante explorar también escala nacional si queda tiempo, pero priorizar el de mejor performance.

---

## Acciones / próximos pasos para el grupo de Jhon

1. **Re-entrenar enfocándose en una región específica** (a definir como grupo según criterio: mayor número absoluto de casos, mayor incidencia por habitante, o interés particular).
2. **Usar toda la serie temporal disponible** (no solo años representativos).
3. **Reajustar la partición train/test:** último ~20% del rango temporal como test (p. ej. 2019–2024 test, anteriores como train).
4. **Reportar Precision y Recall** sobre el test como métricas principales (además de Accuracy y F1).
5. **Mantener regresión logística y XGBoost como modelos base**; deep learning queda opcional y, si se intenta, con `hidden_size` muy pequeño (2–8).
6. **Considerar cruzar con disponibilidad hospitalaria** usando el Marco Geoestadístico Nacional del DANE.
7. **Construir un dashboard prototipo** con entrada por archivo Excel ya procesado; conexiones por API son opcionales.

---

## Referencias mencionadas

- **Marco Geoestadístico Nacional — DANE**: datos georreferenciados de municipios y hospitales en Colombia.
- **bbycroft.net** — visualizador interactivo de arquitecturas de LLMs/Transformers.
- **SIVIGILA** — fuente de datos epidemiológicos de dengue (aún sin datos 2025 al momento de la reunión).
