# Method-specific BH-FDR analysis (added 2026-11-14)

`06_BH_FDR_by_Method/calculate_BH_FDR_by_method*.py` is an ADDITIONAL
supplementary analysis. It does NOT modify or replace the joint IUT/BH
procedure in `02_Main_Analysis/summarize_IUT_BH_REVISED_20261111.py`.

Input: `INFERENCE_BY_METHOD_{overall,non_vaccine,vaccine}.csv`
Output (next to the Python file):
- `BH_FDR_BY_METHOD_ALL_AEs.csv`: per-AE p, q, and flags
- `BH_FDR_BY_METHOD_SUMMARY.csv`: six dataset x model summaries
- `BH_FDR_BY_METHOD_SURVIVING_CANDIDATES.csv`: five vaccine/Prophet rows

Correct p-value definition (IMPORTANT): original `p_direction_adjusted_two_sided`
is the TWO-SIDED t-test p-value if Delta MAE is positive, otherwise 1.0.
It is NOT the standard one-sided p-value (p_two_sided / 2).
Use this exact direction-constrained, conservative two-sided p definition
for the reported 5/10 vaccine/Prophet result. In the archived actual data,
using a conventional one-sided p-value yields 6/10, not 5/10.

BH correction is performed separately over evaluable AEs for each of:
(overall, Prophet), (overall, ARIMA/SARIMA), (non-vaccine, Prophet),
(non-vaccine, ARIMA/SARIMA), (vaccine, Prophet), (vaccine, ARIMA/SARIMA).
Counts tested: 720/720, 692/692, 47/47. q < 0.05.
Expected retained: 0, 0, 0, 0, 5, 0 respectively.

This exploratory supplementary analysis does not change the nominal
95% CI > 0 candidate rule, forecast models, or the joint IUT results.
The 11 origin-year Delta MAE estimates may not be independent.

