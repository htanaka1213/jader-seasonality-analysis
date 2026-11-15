# Verification and remaining limitations (2026-11-13)

- All packaged Python scripts pass syntax compilation (public: 33, private: 66 including the original snapshots).
- A scan of public `.py`, `.md`, `.txt`, `.csv` and `.json` content found no private home-directory paths or identifying account names.
- The public sensitivity script completed `--mode validate` using the previously supplied 3 monthly datasets and 3 main inference tables (727 / 698 / 51 AEs; 180 months each; all relationships passed).
- The public peak-month script reproduced the previous 1,476-row peak-month output with identical values.
- The public revised-main logic regression test passed without refitting all 1,476 AEs.
- The public secondary-metrics script processed 2,952 monthly prediction files from the previously saved main archive with zero MAE QC mismatches. All metrics matched the prior secondary CSV, except a few MAPE exploratory t-CI endpoints with a maximum absolute difference about `2.3e-8` under the test environment's different SciPy version. This does not affect MAE, IUT, BH or published numerical precision.
- Synthetic CP932 raw DRUG vs UTF-8 DRUG tests yielded identical vaccine case classification; the revised preparation code now reads case IDs as strings in REAC and DRUG.
- The public S8 script ran against synthetic CP932 REAC/DRUG inputs using repository-relative data paths and wrote a CSV. Synthetic case counts intentionally differ from the full JADER study.

**Not yet verified**: direct reproduction of all 3 analytical monthly datasets from the original complete PMDA JADER 2024-09 raw REAC/DRUG files; full numerical re-estimation across all 1,476 AE-dataset series using this newly reorganized repository; reproduction of every Excel/PowerPoint figure and manually prepared table from scripts alone. The source JADER records and complete model result archive are not included in this code ZIP.

Original pre-modification files are kept byte-for-byte in the private archive's `99_Original_Unmodified_Code/`.
