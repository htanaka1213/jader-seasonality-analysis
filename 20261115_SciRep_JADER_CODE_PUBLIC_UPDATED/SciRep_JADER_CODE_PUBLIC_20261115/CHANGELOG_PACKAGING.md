# Changes made while packaging — November 13, 2026

1. `create_analysis_data.py`: support source `drug202409.csv` decoded as CP932 if `drug202409_utf8.csv` is absent; read `識別番号` in REAC and DRUG as matching strings (prevents numeric/string ID mismatch). Other vaccine, onset date and AE counting rules unchanged.
2. `run_full_ae_sensitivity_USER_ORIGINALS_FIXED.py`: corrected `continueycy7` -> `continue` in invalid-variance rejection branch. The supplied `Python_code (2).zip` still contained the typo; corrected code is the working copy in both archives. The *private* archive also includes the untouched original in `99_Original_Unmodified_Code`.
3. Public archive only: configure project-relative data/output paths for raw data preparation, revised main analysis, sensitivity analysis, secondary metrics, peak-month analysis and S8; retain model-selection/statistical calculations.
4. Public old submission scripts: configure nonempty input/output paths for portability; keep older 5-fold/Prophet inference definitions as historical only. Legacy vaccine output filename corrected to `ARIMA_SARIMA_vaccine.csv` (file name only).
5. Public AIC diagnostic scripts: substitute project-relative paths; retain them as diagnostics not primary code.
6. Both archives include 6 legacy original-submission scripts, 2 data preparation scripts, 6 revised run launchers, a shared revised analysis core/config, IUT/BH summary code, full AE sensitivity, RMSE/MAPE, peak month, revised suspected-drug S8, diagnostic scripts and tests.
7. Data and model output archives are **not** automatically bundled. See `data/README_INPUTS.md` for preparation details.
