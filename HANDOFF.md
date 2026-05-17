# Handoff al repo `dashboard_dengue`

Este documento describe el traspaso de artefactos del repo `dengue-eda`
(modelado y datos) al repo `dashboard_dengue` (Streamlit + FastAPI), y
contiene el prompt listo para usar en una sesión de Claude Code en el
otro repo.

---

## Estado del traspaso

El script `scripts/exportar_artefactos_dashboard.py` ya copió los artefactos
necesarios a `../microproyecto_entrega 2/dashboard_dengue/`:

| Archivo | Tamaño | Propósito |
|---|---:|---|
| `foco_models.joblib` | ~676 KB | Bundle con `{cod_mpio: {municipio, logistic, xgboost}}` para los 3 foco |
| `panel_municipal_mensual.csv` | ~200 KB | Panel filtrado: 3 mpios × 18 años × 12 meses = 648 filas × 42 cols |
| `predicciones_test.csv` | ~12 KB | Predicciones cacheadas (test 2020–2024) por los 3 modelos |
| `docs/decisiones_proyecto.md` | ~22 KB | Copia de la metodología (D1–D17) |
| `backup_pre_foco/<timestamp>/` | — | Backup de los archivos viejos sobrescritos |

El archivo viejo `logistic_regression.joblib` (modelo nacional único de la
Entrega 2) **sigue presente en el destino** y debe eliminarse durante la
adaptación del código.

---

## Prompt para iniciar la sesión en el otro repo

Pega este bloque como primer mensaje al abrir Claude Code en
`dashboard_dengue/`.

```text
Necesito adaptar este dashboard al nuevo enfoque per-municipio del
proyecto dengue. Los artefactos ya están copiados al repo. Lo que falta
es modificar el código.

CONTEXTO

Este repo era la Entrega 2: un Streamlit + FastAPI sobre un único modelo
nacional (`logistic_regression.joblib`) aplicado a ~1000 municipios. El
proyecto evolucionó: ahora se entrenan modelos independientes por
municipio sobre 3 foco (Valencia, Fundación, El Retorno), con datos
2007-2024 unificados y target de exceso revisado. Toda la metodología
está en `docs/decisiones_proyecto.md` (D1-D17). El acta del director del
proyecto (Carlos, 2026-04-24) que originó las decisiones también está
en el repo de origen `../dengue-eda/docs/reuniones/`.

ARTEFACTOS NUEVOS YA PRESENTES EN ESTE REPO

  - foco_models.joblib      Bundle consolidado, estructura:
                            {cod_mpio (str): {
                              "municipio": str,
                              "logistic": {model, scaler, features},
                              "xgboost":  {model, scaler, features, best_params}
                            }}
                            3 mpios: "23855" Valencia (Córdoba),
                                     "47288" Fundación (Magdalena),
                                     "95025" El Retorno (Guaviare).
  - panel_municipal_mensual.csv   648 filas × 42 cols, ya filtrado a foco.
                                  Columnas clave: cod_mpio, municipio, ano,
                                  mes, casos_total, casos_regular,
                                  casos_grave, poblacion, incidencia_x100k,
                                  exceso (0/1).
  - predicciones_test.csv         Predicciones cacheadas para 2020-2024
                                  por los 3 modelos por mpio:
                                  pred_baseline, pred_logistic,
                                  proba_logistic, pred_xgboost,
                                  proba_xgboost.
  - docs/decisiones_proyecto.md   Metodología (D1-D17) ya copiada.
  - backup_pre_foco/<timestamp>/  Archivos viejos respaldados.
  - logistic_regression.joblib    VIEJO de Entrega 2, DEBE SER BORRADO.

LO QUE NECESITO QUE HAGAS

1. Adaptar `app.py`:
   - Cargar `foco_models.joblib` en lugar de `logistic_regression.joblib`.
   - Restringir el dashboard a los 3 municipios foco (filtrar/ocultar
     municipios no foco en los selectores).
   - Modelo por defecto que aplica predicciones: XGBoost. Permitir ver
     también Logística y Baseline para comparación.
   - Para el municipio seleccionado, leer su modelo correspondiente del
     bundle, escalar con su scaler, predecir con sus features.
   - Métricas y matrices de confusión: calcular desde
     `predicciones_test.csv` (no recomputar en cada render — usar cache).
   - Mantener la lógica de niveles de alerta
     (Normal / Riesgo / Alerta) con thresholds 0.3 / 0.6.

2. Adaptar `api.py`:
   - El endpoint `/predict` debe recibir `cod_mpio` además del dict de
     features. Validar que `cod_mpio` esté en el bundle.
   - Cargar `foco_models.joblib` en `lifespan`.
   - Considerar un endpoint `/municipios` que liste los 3 con metadata
     (nombre, depto, región).

3. Borrar `logistic_regression.joblib`.

4. Actualizar `README.md` y `CLAUDE.md`:
   - Reflejar nuevo alcance (3 foco).
   - Referenciar `docs/decisiones_proyecto.md` para metodología.

5. Smoke test:
   - `streamlit run app.py` y verificar:
     a) El dashboard carga sin errores.
     b) Los 3 municipios aparecen en el selector.
     c) Al seleccionar uno, se ven sus métricas (Precision/Recall) y la
        serie temporal con predicciones del test 2020-2024.
   - `uvicorn api:app --reload` y `curl http://localhost:8000/health`
     debe devolver información correcta del bundle.

REFERENCIAS PARA LEER

  - foco_models.joblib (joblib.load para inspeccionar estructura)
  - panel_municipal_mensual.csv (head para entender columnas)
  - docs/decisiones_proyecto.md (D4 métricas, D5 modelos, D12 target,
    D15 features, D16 tuning, D17 baseline trivial)

ENTREGABLES ESPERADOS

  - app.py y api.py adaptados.
  - logistic_regression.joblib borrado.
  - README + CLAUDE.md actualizados.
  - Commit + push con mensaje descriptivo (si el repo tiene origin).
```

---

## Si necesitas re-exportar artefactos desde `dengue-eda`

Si re-entrenas los modelos o regeneras el panel y quieres actualizar los
artefactos del dashboard:

```bash
conda run -n dengue-eda python scripts/exportar_artefactos_dashboard.py
```

(Soporta `--dry-run` para preview y `--destino` si la ruta del dashboard
cambia.)

---

## Referencia inversa

El repo `dengue-eda` está en GitHub:
https://github.com/jhoneduardmoreno/dengue-eda

Cualquier ajuste futuro a la metodología debe hacerse en `dengue-eda` y
re-exportarse al dashboard, no editar el dashboard a mano.
