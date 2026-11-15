# Additional analysis scripts, 2026-11-14

Package type: public / professor.

Original 33 Python scripts from the supplied November 13 ZIP
are preserved with identical SHA256 digests. Main prediction, sensitivity,
original RMSE/MAPE, data preparation, and S8 analysis are NOT changed.

New items:
- Separate BH-FDR correction (data-set x model, 6 testing families)
- Three input main inference CSVs and reference outputs for validation
- Standalone monthly-forecast exporter (2,952 originals, 389,664 months)
- Optional combined RMSE/MAPE and monthly-export script
- README files, new code manifest, explanatory p-value clarification

The original `README.md` and `CODE_MANIFEST.csv` are preserved under
`docs_archived/`. Old Japanese filenames used in historical documentation
were translated to ASCII filenames **without changing their content**.

The original code here uses a direction-constrained TWO-SIDED p-value;
this is important for interpreting the 5 surviving vaccine/Prophet candidates.
