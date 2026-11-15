# Monthly prediction export (added 2026-11-14)

The original `04_Secondary_Metrics/calculate_RMSE_MAPE_revised_main.py`
is retained UNCHANGED.

* `export_main_monthly_predictions_only_*.py`: exports the 2,952
  original `revised_cv_monthly_paired.csv` files, each with 132 months, into
  one ZIP containing `MAIN_MONTHLY_PREDICTIONS_ALL_AEs.csv` (389,664 rows).
  This does NOT rerun any models and does NOT recalculate RMSE/MAPE.
* `calculate_RMSE_MAPE_revised_main_WITH_MONTHLY_EXPORT*.py`: alternative
  unified RMSE/MAPE recalculation + monthly export. Run this INSTEAD of
  (not in addition to) the original RMSE/MAPE script if recalculation
  is desired.

Input: saved main-forecast archive `SciRep_GitHub_REVISED_20261111_ENVFIX.zip`
(or in the integrated portable version, a supported output directory).
The complete saved forecast archive is NOT in this code-only package.
Use the separate ResultFiles/ForecastData archives to share numerical data.

The personal-PC versions retain the historical user input path (editable);
the public versions DO NOT contain any user-specific paths.
Outputs are placed next to the added Python script except where the integrated
portable RMSE/MAPE script uses its documented default subfolder.

The exporter performs no rounding or numeric transformation: it copies the
original monthly CSV content and checks total source files and rows.
