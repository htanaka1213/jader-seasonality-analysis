ARCHIVED NOTE: This document records an earlier local workflow. For paths and execution steps in this packaged repository, follow the root README.md instead.

# Change log — JADER seasonality Scientific Reports revision (2026-11-11)

The original submission code is **not** changed or overwritten by this release.

- Aligned ARIMA/SARIMA and Prophet to 11 expanding-year training origins and 12 complete calendar-month forecasts per origin (132 evaluation months, 2009–2019).
- Replaced ARIMA/SARIMA order search from original 8 + 64 grid with controlled `d=1,D=0` fixed-differencing 4 + 12 candidate selection, excluding SARIMA candidates with no seasonal terms, and rejecting nonconverged and noncomparable-likelihood fits.
- Kept submitted Prophet yearly seasonality on/off comparison and default forecasting settings; replaced day-based cross-validation with a fixed calendar-month evaluation design.
- Exported paired per-month actuals, predictions, errors and per-origin model diagnostics and input/environment hashes.
- Added documented all-zero training history handling. AE with any such origin is kept for descriptive results but excluded from primary inference **for both model families**.
- Added cutoff-year t-based exploratory CI and p values, monthly HAC(12) CI sensitivity, joint improvement IUT and per-dataset BH-FDR.
- Added partial-run safety: no finalized q values until every eligible AE in a dataset has been audited for both approaches.
- Added 6 standalone Spyder launchers with names corresponding to the 6 submitted GitHub files, one all-in-one launcher, tests and a methods README.

Known limitations: the cutoff-level t assumption and BH's dependence assumptions are not empirically established as exact for JADER; q values must be interpreted as exploratory. Model specification choice should be ratified before any full-analysis run.


## Hotfix: stable CmdStan fingerprint
- Initialize Prophet's default Stan backend before reading `cmdstanpy.cmdstan_version()`; reject unresolved `None` rather than persisting it into a run manifest.
- Use an isolated default output directory (`..._ENVFIX`) to avoid mixing old and patched run fingerprints.
- Add a no-Prophet regression test with a simulated lazy backend initialization.
