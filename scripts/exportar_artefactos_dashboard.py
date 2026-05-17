"""
Empaqueta y copia los artefactos del proyecto dengue-eda al repo
dashboard_dengue para consumo desde el Streamlit/FastAPI.

Artefactos exportados:
  1. foco_models.joblib — bundle consolidado {cod: {logistic, xgboost, municipio}}
  2. panel_municipal_mensual.csv — panel filtrado a los 3 foco
  3. predicciones_test.csv — predicciones cacheadas (test 2020-2024)

Los archivos preexistentes del destino con el mismo nombre se respaldan en
una subcarpeta `backup_pre_foco/` con timestamp.

Uso:
  python scripts/exportar_artefactos_dashboard.py
  python scripts/exportar_artefactos_dashboard.py --dry-run
  python scripts/exportar_artefactos_dashboard.py --destino "ruta/personalizada"
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import MUNICIPIOS_FOCO, PROJECT_ROOT  # noqa: E402


DEFAULT_DESTINO = (
    PROJECT_ROOT.parent
    / "microproyecto_entrega 2"
    / "dashboard_dengue"
)


# ---------------------------------------------------------------------------
# Bundle de modelos
# ---------------------------------------------------------------------------
def construir_bundle() -> dict:
    """Lee los 6 joblibs individuales y arma un dict consolidado."""
    models_dir = PROJECT_ROOT / "models"
    bundle = {}
    for cod, nom in MUNICIPIOS_FOCO.items():
        log_path = models_dir / f"{cod}_logistic.joblib"
        xgb_path = models_dir / f"{cod}_xgboost.joblib"
        if not log_path.exists() or not xgb_path.exists():
            raise FileNotFoundError(
                f"Faltan joblibs para {cod} ({nom}). "
                f"Corre primero notebooks/07_modelado.ipynb."
            )
        bundle[cod] = {
            "municipio": nom,
            "logistic": joblib.load(log_path),
            "xgboost":  joblib.load(xgb_path),
        }
    return bundle


# ---------------------------------------------------------------------------
# Copia segura (con backup)
# ---------------------------------------------------------------------------
def copiar_con_backup(src: Path, dst: Path, backup_dir: Path, dry_run: bool):
    """Copia src → dst. Si dst ya existe, lo mueve a backup_dir antes."""
    accion = []
    if dst.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_target = backup_dir / dst.name
        accion.append(f"backup: {dst.name} → backup_pre_foco/{dst.name}")
        if not dry_run:
            shutil.move(str(dst), str(backup_target))
    accion.append(f"copy:   {src.name} → {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    return accion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path, default=DEFAULT_DESTINO,
                        help=f"Carpeta destino (default: {DEFAULT_DESTINO})")
    parser.add_argument("--dry-run", action="store_true",
                        help="No copia archivos, solo muestra lo que haría.")
    args = parser.parse_args()

    destino: Path = args.destino
    if not destino.exists():
        print(f"ERROR: la carpeta destino no existe → {destino}")
        sys.exit(1)
    print(f"Destino: {destino}")
    if args.dry_run:
        print("(modo --dry-run: no se modifica nada)\n")

    # 1) Bundle de modelos
    print("1) Construyendo bundle de modelos consolidado…")
    bundle = construir_bundle()
    for cod, nom in MUNICIPIOS_FOCO.items():
        print(f"   ✓ {cod} {nom}: logistic + xgboost cargados")

    bundle_tmp = PROJECT_ROOT / "models" / "foco_models.joblib"
    if not args.dry_run:
        joblib.dump(bundle, bundle_tmp)
    print(f"   → guardado temporalmente en {bundle_tmp.relative_to(PROJECT_ROOT)}")

    # 2) Definir copias
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = destino / "backup_pre_foco" / timestamp

    copias = [
        (bundle_tmp,
         destino / "foco_models.joblib"),
        (PROJECT_ROOT / "data" / "processed" / "panel_municipal_mensual.csv",
         destino / "panel_municipal_mensual.csv"),
        (PROJECT_ROOT / "data" / "processed" / "predicciones_test.csv",
         destino / "predicciones_test.csv"),
    ]

    # 3) Ejecutar copias
    print("\n2) Copiando artefactos al destino…")
    for src, dst in tqdm(copias, desc="Exportando"):
        # En dry-run el bundle no existe físicamente; las otras fuentes sí deben existir.
        if not src.exists() and not (args.dry_run and src.name == "foco_models.joblib"):
            print(f"   ⚠️ FALTA origen: {src} — se omite")
            continue
        for line in copiar_con_backup(src, dst, backup_dir, args.dry_run):
            print(f"   {line}")

    # 4) Resumen
    print("\n3) Resumen")
    print(f"   Bundle:     foco_models.joblib ({len(bundle)} municipios × 2 modelos)")
    print(f"   Panel:      panel_municipal_mensual.csv (648 filas, 3 foco × 18 años)")
    print(f"   Predicciones: predicciones_test.csv (test 2020-2024)")
    if args.dry_run:
        print("\n   ⚠️ Dry-run: no se modificó nada en disco.")
    else:
        if backup_dir.exists():
            print(f"\n   Backups previos en: {backup_dir.relative_to(destino)}")
        print("\n   ✅ Listo. Próximo paso: adaptar app.py y api.py en el repo dashboard_dengue")
        print("      para cargar foco_models.joblib y filtrar a los 3 municipios.")


if __name__ == "__main__":
    main()
