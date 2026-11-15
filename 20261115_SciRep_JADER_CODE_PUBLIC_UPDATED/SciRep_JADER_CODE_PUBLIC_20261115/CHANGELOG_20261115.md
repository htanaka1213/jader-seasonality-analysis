# Reproducibility update, 2026-11-15 — manuscript v12

## Scope

1. `05_Supplementary_S8/supplementary_table_S8_top10_suspected_drugs_REVISED.py`: the English **output display label** for JADER `食欲減退` is now `Decreased appetite` (was `Appetite loss`).
2. `80_Original_Submission/supplementary_top10_suspected_drugs.py`: same English label-only correction for convenient execution; original unmodified historical source is preserved at `80_Original_Submission/historical_unmodified/supplementary_top10_suspected_drugs_20260507_original.py`. Historical outputs bearing the old label remain attributable to the unmodified original.
3. `06_BH_FDR_by_Method/calculate_BH_FDR_by_method_standalone.py`: *only* updates the final test success message from manuscript v7 to v12; numerical formula unchanged.

## Verified, not changed

- Main `SEAS_ORDERS` explicitly requires `P + Q > 0`; sensitivity `candidate_orders` explicitly skips `P + Q == 0`. Therefore no SARIMA algorithm edit is required.
- Standalone per-(dataset,method) BH correction uses the existing two-sided t-test p-value for positive Delta MAE and p=1 otherwise, applying BH separately to six families. The `02_Main_Analysis/summarize_IUT_BH...` historical joint procedure is distinct and was not modified.
- BH outputs regenerated from all three included inference CSVs, with each row/cell matching the prior reference outputs. Summary: overall/Prophet 720/7/0, overall/ARIMA-SARIMA 720/9/0, non-vaccine/Prophet 692/4/0, non-vaccine/ARIMA-SARIMA 692/7/0, vaccine/Prophet 47/10/5, vaccine/ARIMA-SARIMA 47/2/0 (number tested / CI-positive candidates / candidates surviving BH).

## Verification limits

- No full ARIMA/SARIMA or Prophet refitting was performed.
- Raw JADER input is not included in this public ZIP. Therefore the suspected-drug scripts have not been run end-to-end on raw JADER input in this update. The AE English-label substitution was verified by exact source-code inspection; the Japanese lookup term and selection logic are unchanged.
- Runtime used for standalone BH reproduction: Python 3.13, pandas 2.2.3, statsmodels 0.14.6 (the manuscript's main run environment is stated in the code and manuscript).
