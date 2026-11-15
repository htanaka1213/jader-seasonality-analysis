ARCHIVED NOTE: This document records an earlier local workflow. For paths and execution steps in this packaged repository, follow the root README.md instead.

# Peak month and peak-to-mean ratio — revised JADER study

This is a **separate descriptive analysis** that accompanies the 11-origin Prophet and ARIMA/SARIMA scripts. It does **not** refit forecasting models and does not modify MAE, IUT or FDR analysis.

## Files

- `compute_peak_month_REVISED_20261113.py`: reproducible standalone analysis
- `PEAK_MONTH_ALL_AEs_20261113.csv`: output generated with the same three 2005–2019 monthly count inputs as the revised manuscript

## Definition

For each AE and dataset, sum reports over 2005–2019 separately for January, February, ..., December. Peak month is the largest of these 12 totals, with all tied months reported. Peak-to-mean ratio = (peak month total) / (15-year AE total / 12). This is a descriptive ratio, not an estimate of AE incidence or a test of seasonality.

The code screens for AEs with at least 100 reports across the 180 monthly observations, as in the revised main analysis. Expected rows: overall 727; non-vaccine 698; vaccine 51; total 1,476 AE–dataset pairs. For auditing, the CSV includes 12 month-of-year totals, peak-month reports and tied-peak counts.

## Running

Python 3.12; numpy and pandas are required.

The easiest **portable** workflow is to put these three input CSV files together in a `data/` folder **next to the script**:

- `20250112_yearly_counts_filtered.csv`
- `20260411_yearly_counts_filtered_non_vaccine.csv`
- `20260411_yearly_counts_filtered_vaccine_only.csv`

Then run `python compute_peak_month_REVISED_20261113.py` or press F5 in Spyder. The output will be saved in `output_peak_month/` next to the script.

Alternative: set the input folder on the command line:

```
python compute_peak_month_REVISED_20261113.py --data-dir /path/to/monthly/csvs
```

Alternatively, place this script in the same directory as `scirep_revision_config_20261111.py` and run it; it reads `INPUT_FILES` from the existing main-analysis configuration. The original user's OneDrive source locations are a final fallback, but **should not be relied on for a public repository**.

The `--output-dir` argument overrides the default output folder.

## Validation

Independently verified all 1,476 output rows against the three original monthly count CSVs, comparing 12 month totals, 15-year totals, peak months (including ties) and ratios. **Zero mismatches**. This portable script's output equals the uploaded original script's output (including all 1,476 rows).

## Limitations / manuscript clarity

The result describes report-count concentrations; it does not establish seasonal variation in AE risk or causal effects. Note that the revised Supplementary Tables S1–S6 currently list forecast metrics but do not contain columns named `peak_month` or `peak_to_mean_ratio`. If the manuscript says these values for all AEs are provided in those tables, add the columns or explicitly refer to this CSV as supplementary data.
