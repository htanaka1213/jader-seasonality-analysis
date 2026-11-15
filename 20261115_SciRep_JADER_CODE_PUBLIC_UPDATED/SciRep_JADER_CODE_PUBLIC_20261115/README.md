# Scientific Reports JADER seasonality — updated computational code, 2026-11-15

This package supplements the uploaded November 13 code archive. **All numerical forecasting algorithms are unchanged.** Three Python files
were edited on 2026-11-15: two AE English display labels and one BH validation
message. The completely unmodified 2026-05-07 `supplementary_top10_suspected_drugs.py`
is retained in `80_Original_Submission/historical_unmodified/`. No refitting
or change in the primary classification was performed.

## Added on 2026-11-14

1. `06_BH_FDR_by_Method/` — method-specific Benjamini–Hochberg correction
   separately for each (dataset, method) family; 6 families. Run it after
   main inference CSV generation. 5 of 10 vaccine/Prophet nominal candidates
   retain q<0.05; all other families retain 0. Existing joint IUT-BH is
   independent and unchanged.
2. `04_Secondary_Metrics/export_main_monthly_predictions_only_*.py` — exports
   existing 132-month forecasts for all 2,952 AE/method entries to one CSV ZIP.
   This does NOT fit forecasting models or re-estimate RMSE/MAPE.
3. `04_Secondary_Metrics/calculate_RMSE_MAPE_revised_main_WITH_MONTHLY_EXPORT*.py`
   — optional integrated alternative that runs the original secondary metrics
   AND monthly export. Keep original `calculate_RMSE_MAPE_revised_main.py`.
4. `06_BH_FDR_by_Method/reference_results/` — reproducible numerical results.

**P-value clarification:** `p_direction_adjusted_two_sided` is the **two-sided**
t-test p value for positive Delta MAE; for nonpositive Delta MAE it is set
to 1.0. It is not the conventional one-sided p value. Its use reproduces
5 vaccine/Prophet BH detections; using standard one-sided p values instead
would produce 6, changing the manuscript. The source main analysis uses the
conservative two-sided/directional p-value definition. This supplementary
BH analysis does not modify the primary CI-based seasonality rule.

## Data and source files

- This is a **code archive**, not the complete paper's data release.
- Official PMDA JADER raw data and the main 11-origin prediction ZIP are
  NOT included. Submit these separately as appropriate.
- The three main inference CSVs and derived BH output CSVs are included
  as convenience inputs/verification for method-specific BH analysis.
- Final manuscript figures, tables, and the external weather data are
  maintained separately and must be submitted as separate files when ready.
- The personal archive also retains the original private code snapshots.

## ZIP file names

All filenames within the ZIP are ASCII-only to avoid mojibake with certain
Windows ZIP extractors. The historical Japanese-language documentation content
is preserved, though a small number of archived document FILE NAMES were
renamed for portability. CSV Japanese values and Python source encodings
are preserved; no data values were transliterated.

Detailed instructions: `06_BH_FDR_by_Method/README_BH_FDR.md` and
`04_Secondary_Metrics/README_MONTHLY_EXPORT.md`.

## Public / professor version

- This release contains no hard-coded personal Windows paths.
- The BH script can run immediately using the three bundled CSV files in
  `06_BH_FDR_by_Method/input/`. Outputs appear next to the BH script.
- For monthly export, place the saved full forecast ZIP next to the
  exporter or edit `SOURCE_ZIP` to its actual location. Output appears
  next to the monthly exporter, named
  `MAIN_MONTHLY_PREDICTIONS_ALL_AEs.zip`.
- For integrated RMSE/MAPE + export, use the _WITH_MONTHLY_EXPORT.py
  script and set `PRIMARY_RESULT_SOURCE` when auto-discovery fails.

## Update on 2026-11-15 (manuscript v12)

- English display label corrected from **Appetite loss** to **Decreased appetite**
  for JADER term `食欲減退` in the current `05_Supplementary_S8/` code
  and in `80_Original_Submission/supplementary_top10_suspected_drugs.py`.
  The untouched historical code is saved in the `historical_unmodified/` folder.
  No AE definition, counting, numeric output, model or classification criterion changed.
- `P + Q > 0` is already enforced for SARIMA model candidates in
  `02_Main_Analysis/scirep_revision_cv_core_20261111.py` and
  `03_Sensitivity_Analysis/run_full_ae_sensitivity_USER_ORIGINALS_FIXED.py`.
  The main model-fitting algorithms were **not** changed in this update.
- The separately calculated BH-FDR code under `06_BH_FDR_by_Method/`
  was re-run against the bundled three inference input CSVs. All three
  generated outputs were identical to the prior `reference_results/`
  (2,952 inference rows, 6 summary rows, 5 surviving vaccine/Prophet rows).
  The code's final PASS message now refers to manuscript v12.
  The independent historical joint IUT/BH routine is not the new
  dataset-by-method FDR analysis; it has deliberately been preserved.
- See `CHANGELOG_20261115.md` for exact filenames, rationale and validation.
