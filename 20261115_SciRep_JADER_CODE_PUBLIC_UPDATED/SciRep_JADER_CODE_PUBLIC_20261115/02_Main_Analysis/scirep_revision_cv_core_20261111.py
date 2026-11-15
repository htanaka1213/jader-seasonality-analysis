# -*- coding: utf-8 -*-
"""Revised forecast extraction for the submitted JADER ARIMA/SARIMA/Prophet code.

Original GitHub source names:
  run_prophet_{overall,non_vaccine,vaccine}_analysis.py
  run_arima_sarima_{overall,non_vaccine,vaccine}_analysis.py

This revision makes evaluation windows consistent at 11 expanding origins x
12 forecast months (Jan 2009 - Dec 2019). It retains submitted Prophet model
hyperparameters, but adopts the separately piloted controlled SARIMAX AIC grid:
d=1, D=0, equal likelihood sample across candidates. NOT a replica of the
original 5-fold ARIMA/SARIMA or the Prophet 365-day cutoffs.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import platform
import re
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

YEARS = tuple(range(2009, 2020))
SCHEMA = "JADER_11x12_REVISED_20261111_schema_v1"
AR_ORDERS = tuple((p, 1, q) for p, q in itertools.product((0, 1), repeat=2))
SEAS_ORDERS = tuple((P, 0, Q, 12) for P, Q in itertools.product((0, 1), repeat=2) if P + Q > 0)
ARIMA_SPEC = {"d": 1, "D": 0, "trend": "n", "simple_differencing": False,
              "enforce_stationarity": True, "enforce_invertibility": True,
              "reject_nonconverged": True, "period_months": 12,
              "candidate_arima": 4, "candidate_sarima": 12}


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def safe_slug(event: str) -> str:
    prefix = re.sub(r"[^\w\-]+", "_", str(event), flags=re.UNICODE).strip("_")[:48]
    return f"{prefix}_{hashlib.sha1(str(event).encode('utf-8')).hexdigest()[:12]}"


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.16g")
    os.replace(tmp, path)


def atomic_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def env_versions():
    import scipy
    result = {"python": platform.python_version(), "numpy": np.__version__,
              "pandas": pd.__version__, "scipy": scipy.__version__}
    for mod in ("statsmodels", "prophet", "cmdstanpy"):
        try:
            import importlib
            obj = importlib.import_module(mod)
            result[mod] = getattr(obj, "__version__", "unknown")
        except ImportError:
            result[mod] = "NOT_INSTALLED"
    # Prophet's bundled CmdStan path may not be registered with cmdstanpy
    # until the Prophet Stan backend is first instantiated.  Calling
    # cmdstanpy.cmdstan_version() earlier can yield None, whereas the same
    # process reports e.g. (2, 37) after the first fit.  Since this value is
    # included in RUN_MANIFEST's fingerprint, proactively initialize exactly
    # the default backend used for later forecasts BEFORE obtaining its version.
    # This only constructs a model/backend; it does not fit any AE data.
    if result["prophet"] == "NOT_INSTALLED":
        result["cmdstan"] = "NOT_INSTALLED"
    else:
        try:
            from prophet import Prophet
            import cmdstanpy
            _version_probe = Prophet(yearly_seasonality=False)
            version = cmdstanpy.cmdstan_version()
            if version is None:
                raise RuntimeError("cmdstan_version() returned None after Prophet backend initialization")
            result["cmdstan"] = str(version)
        except Exception as exc:
            raise RuntimeError(
                "Failed to resolve CmdStan version for a reproducible run "
                "fingerprint. Check the Prophet/CmdStanPy installation."
            ) from exc
    return result


def load_source(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"入力月次CSVがありません: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "年月" not in df:
        raise ValueError("入力CSVに '年月' 列がありません")
    dates = pd.to_datetime(df.pop("年月").astype(str), format="%Y%m", errors="raise")
    expected = pd.date_range("2005-01-01", "2019-12-01", freq="MS")
    if not pd.DatetimeIndex(dates).equals(expected):
        raise ValueError("年月が2005-01～2019-12の連続180か月ではありません")
    if df.columns.duplicated().any():
        raise ValueError("AE列名の重複があります")
    df.index = expected
    # Match >=100 screening from the submitted scripts. No event names are translated.
    counts = df.apply(pd.to_numeric, errors="raise")
    if counts.isna().any().any() or (counts < 0).any().any():
        raise ValueError("入力CSVに欠損・負の報告件数があります")
    eligible = counts.loc[:, counts.sum(axis=0) >= 100].copy()
    if not np.isfinite(eligible.to_numpy(float)).all():
        raise ValueError("AE列に非有限の数値があります")
    return eligible


def split_origin(series: pd.Series, year: int):
    if year not in YEARS:
        raise ValueError(year)
    train = series.loc[:pd.Timestamp(year - 1, 12, 1)]
    test = series.loc[pd.Timestamp(year, 1, 1):pd.Timestamp(year, 12, 1)]
    if len(train) != 48 + 12 * (year - 2009) or len(test) != 12:
        raise AssertionError("11-origin design is not aligned")
    return train, test


def fit_prophet_pair(train: pd.Series, test: pd.Series):
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    training = pd.DataFrame({"ds": train.index, "y": train.to_numpy(float)})
    future = pd.DataFrame({"ds": test.index})
    forecasts, info = {}, {}
    for key, flag in (("baseline", False), ("seasonal", True)):
        model = Prophet(yearly_seasonality=flag)  # same defaults as submission
        model.fit(training)
        pred = model.predict(future)
        if not pd.DatetimeIndex(pd.to_datetime(pred["ds"])).equals(pd.DatetimeIndex(test.index)):
            raise AssertionError("Prophet predicted dates do not match next 12 calendar months")
        values = np.asarray(pred["yhat"], dtype=float)
        if len(values) != 12 or not np.isfinite(values).all():
            raise RuntimeError("Invalid Prophet forecast")
        forecasts[key] = values
        info[key] = {"yearly_seasonality": flag}
    return forecasts, info


def aic_grid_search(train: pd.Series, *, seasonal: bool):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    combos = [(order, seas) for order in AR_ORDERS
              for seas in (SEAS_ORDERS if seasonal else [(0, 0, 0, 0)])]
    accepted = []
    failures = 0
    not_converged = 0
    bad_burn = 0
    for order, seas in combos:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(train, order=order, seasonal_order=seas, trend="n",
                              simple_differencing=False, enforce_stationarity=True,
                              enforce_invertibility=True).fit(disp=False)
            if not bool(fit.mle_retvals.get("converged", False)):
                not_converged += 1
                continue
            if not np.isfinite(fit.aic):
                failures += 1
                continue
            # AIC comparability: identical likelihood burn and same training sample.
            if int(fit.loglikelihood_burn) != 1 or int(fit.nobs_effective) != len(train)-1:
                bad_burn += 1
                continue
            accepted.append((float(fit.aic), order, seas, fit))
        except Exception:
            failures += 1
    if not accepted:
        return None, {"status": "NO_VALID_CONVERGED_CANDIDATE", "n_success": 0,
                      "n_failures": failures, "n_nonconverged": not_converged,
                      "n_bad_burn": bad_burn, "n_attempted": len(combos)}
    aic, order, seas, fit = min(accepted, key=lambda item: item[0])
    return fit, {"status": "FITTED", "aic": aic, "order": list(order),
                 "seasonal_order": list(seas), "loglikelihood_burn": int(fit.loglikelihood_burn),
                 "nobs_effective": int(fit.nobs_effective), "n_success": len(accepted),
                 "n_failures": failures, "n_nonconverged": not_converged,
                 "n_bad_burn": bad_burn, "n_attempted": len(combos)}


def fit_arima_pair(train: pd.Series, test: pd.Series):
    predictions, details = {}, {}
    for label, seasonal in (("baseline", False), ("seasonal", True)):
        fit, info = aic_grid_search(train, seasonal=seasonal)
        details[label] = info
        if fit is None:
            predictions[label] = np.full(len(test), np.nan)
            continue
        values = np.asarray(fit.get_forecast(steps=len(test)).predicted_mean, dtype=float)
        if len(values) != len(test) or not np.isfinite(values).all():
            predictions[label] = np.full(len(test), np.nan)
            info["status"] = "INVALID_PREDICTION"
        else:
            predictions[label] = values
    return predictions, details


def forecast_origin(series: pd.Series, dataset: str, engine: str, year: int, fingerprint: str):
    train, test = split_origin(series, year)
    t0 = time.perf_counter()
    y = test.to_numpy(float)
    n_nonzero = int((train.to_numpy(float) != 0).sum())
    all_zero = n_nonzero == 0
    note = ""
    info = {}
    if all_zero:
        predictions = {"baseline": np.zeros(12), "seasonal": np.zeros(12)}
        status = "ZERO_HISTORY_BASELINE_NOT_MODEL_FITTED"
        note = "Both forecasts set to 0; descriptive only, no model estimated."
    else:
        try:
            if engine == "prophet":
                predictions, info = fit_prophet_pair(train, test)
            elif engine == "arima_sarima":
                predictions, info = fit_arima_pair(train, test)
            else:
                raise ValueError(engine)
            status = "FITTED" if all(np.isfinite(predictions[k]).all() for k in ("baseline", "seasonal")) else "UNRESOLVED"
        except Exception as exc:
            status = "UNRESOLVED"
            note = f"{type(exc).__name__}: {exc}"
            info["error_traceback"] = traceback.format_exc(limit=4)
            predictions = {"baseline": np.full(12, np.nan), "seasonal": np.full(12, np.nan)}
    a, s = predictions["baseline"], predictions["seasonal"]
    rows = pd.DataFrame({"dataset": dataset, "event": series.name, "engine": engine,
        "origin_year": year, "ds": test.index.strftime("%Y-%m-%d"),
        "horizon_month": np.arange(1, 13), "train_months": len(train),
        "train_reports": float(train.sum()), "train_nonzero_months": n_nonzero,
        "origin_status": status, "y": y, "yhat_baseline": a, "yhat_seasonal": s,
        "ae_baseline": np.abs(y - a), "ae_seasonal": np.abs(y - s),
        "d_abs_error": np.abs(y - a) - np.abs(y - s)})
    details = {"schema": SCHEMA, "dataset": dataset, "event": str(series.name),
               "engine": engine, "origin_year": year, "origin_status": status,
               "train_months": len(train), "train_reports": float(train.sum()),
               "train_nonzero_months": n_nonzero, "status_note": note,
               "models": info, "time_sec": time.perf_counter() - t0,
               "config_fingerprint": fingerprint}
    return rows, details


def validate_checkpoint(rows: pd.DataFrame, meta: dict, *, event: str, dataset: str,
                        engine: str, year: int, fingerprint: str, expected_y: np.ndarray):
    if meta.get("config_fingerprint") != fingerprint or meta.get("origin_year") != year:
        raise ValueError("Origin checkpoint fingerprint/year mismatch")
    if meta.get("event") != event or meta.get("dataset") != dataset or meta.get("engine") != engine:
        raise ValueError("Origin checkpoint event/dataset/engine mismatch")
    if len(rows) != 12 or not rows["origin_year"].eq(year).all():
        raise ValueError("Checkpoint has an incorrect number of forecast months")
    dates = list(pd.to_datetime(rows["ds"]))
    if dates != list(pd.date_range(f"{year}-01-01", periods=12, freq="MS")):
        raise ValueError("Checkpoint forecast months differ from fixed 11x12 design")
    if not np.allclose(rows["y"].to_numpy(float), expected_y, atol=1e-8, rtol=0):
        raise ValueError("Checkpoint actual values differ from input monthly CSV")
    if not np.array_equal(rows["horizon_month"].to_numpy(int), np.arange(1,13)):
        raise ValueError("Checkpoint horizons differ from 1-12 months")
    for col in ("yhat_baseline", "yhat_seasonal", "ae_baseline", "ae_seasonal", "d_abs_error"):
        if col not in rows:
            raise ValueError(f"Checkpoint is missing column: {col}")
    if meta["origin_status"] in ("FITTED", "ZERO_HISTORY_BASELINE_NOT_MODEL_FITTED"):
        if not np.isfinite(rows[["yhat_baseline", "yhat_seasonal", "d_abs_error"]].to_numpy(float)).all():
            raise ValueError("Successful checkpoint has NaN forecasts")
        calc = rows["ae_baseline"].to_numpy(float) - rows["ae_seasonal"].to_numpy(float)
        if not np.allclose(calc, rows["d_abs_error"].to_numpy(float), atol=1e-9):
            raise ValueError("Checkpoint loss difference is inconsistent")


def run_event(series: pd.Series, dataset: str, engine: str, folder: Path,
              fingerprint: str, retry_unresolved: bool):
    folder.mkdir(parents=True, exist_ok=True)
    year_dir = folder / "years"
    year_dir.mkdir(exist_ok=True)
    pieces, meta_list = [], []
    for year in YEARS:
        csv_path = year_dir / f"{year}_predictions.csv"
        json_path = year_dir / f"{year}_metrics.json"
        reuse = csv_path.exists() and json_path.exists()
        if reuse:
            old = json.loads(json_path.read_text(encoding="utf-8"))
            if old.get("config_fingerprint") != fingerprint:
                raise ValueError(f"Stored results are from different settings: {json_path}")
            if retry_unresolved and old["origin_status"] == "UNRESOLVED":
                reuse = False
        if reuse:
            rows = pd.read_csv(csv_path, encoding="utf-8-sig")
            train, test = split_origin(series, year)
            validate_checkpoint(rows, old, event=str(series.name), dataset=dataset,
                                engine=engine, year=year, fingerprint=fingerprint,
                                expected_y=test.to_numpy(float))
            detail = old
            print(f"  {year}: reuse {detail['origin_status']}", flush=True)
        else:
            print(f"  {year}: fitting {engine} (train {48+12*(year-2009)} months)", flush=True)
            rows, detail = forecast_origin(series, dataset, engine, year, fingerprint)
            atomic_csv(rows, csv_path)
            atomic_json(detail, json_path)
            print(f"  {year}: {detail['origin_status']}, {detail['time_sec']:.1f}s", flush=True)
        pieces.append(rows)
        meta_list.append(detail)
    paired = pd.concat(pieces, ignore_index=True).sort_values("ds").reset_index(drop=True)
    required = pd.date_range("2009-01-01", "2019-12-01", freq="MS")
    if len(paired) != 132 or not pd.DatetimeIndex(pd.to_datetime(paired["ds"])).equals(required):
        raise ValueError("Combined forecasts are not 132 continuous months")
    origin_statuses = [x["origin_status"] for x in meta_list]
    initial_all_zero = bool((series.iloc[:48].to_numpy(float) == 0).all())
    n_missing = int(paired["d_abs_error"].isna().sum())
    if initial_all_zero:
        status = "INFERENCE_INELIGIBLE_ZERO_HISTORY"
    elif "UNRESOLVED" in origin_statuses or n_missing:
        status = "INFERENCE_INELIGIBLE_UNRESOLVED"
    else:
        status = "INFERENCE_READY"
    summary = {"dataset": dataset, "event": str(series.name), "engine": engine,
        "status": status, "n_reports_2005_2019": int(series.sum()),
        "initial_48_month_reports": int(series.iloc[:48].sum()),
        "initial_48_nonzero_months": int((series.iloc[:48]>0).sum()),
        "n_origins": 11, "n_origin_fitted": origin_statuses.count("FITTED"),
        "n_origin_zero_history": origin_statuses.count("ZERO_HISTORY_BASELINE_NOT_MODEL_FITTED"),
        "n_origin_unresolved": origin_statuses.count("UNRESOLVED"),
        "n_valid_months": int(paired["d_abs_error"].notna().sum()),
        "n_missing_months": n_missing, "delta_mae_descriptive": float(paired["d_abs_error"].mean()) if n_missing <132 else None,
        "mae_baseline_descriptive": float(paired["ae_baseline"].mean()) if n_missing <132 else None,
        "mae_seasonal_descriptive": float(paired["ae_seasonal"].mean()) if n_missing <132 else None,
        "config_fingerprint": fingerprint}
    atomic_csv(paired, folder / "revised_cv_monthly_paired.csv")
    atomic_csv(pd.DataFrame(meta_list), folder / "revised_origin_metrics.csv")
    atomic_csv(pd.DataFrame([summary]), folder / "revised_event_summary.csv")
    return summary


def run_dataset_engine(config: dict, dataset: str, engine: str):
    if dataset not in config["INPUT_FILES"] or engine not in ("prophet", "arima_sarima"):
        raise ValueError(f"Unknown dataset/engine: {dataset}/{engine}")
    source_path = Path(config["INPUT_FILES"][dataset])
    output_dir = Path(config["OUTPUT_ROOT"]) / dataset / engine
    source = load_source(source_path)
    expected_total = {"overall":727, "non_vaccine":698, "vaccine":51}[dataset]
    if source.shape[1] != expected_total:
        raise ValueError(f"Dataset eligibility count changed: {source.shape[1]} != {expected_total}")
    if config["ARIMA_SPECIFICATION"] != "controlled_d1_D0_v2":
        raise ValueError("Unexpected ARIMA model selection specification")
    env = env_versions()
    if engine == "prophet" and env["prophet"] == "NOT_INSTALLED":
        raise ImportError("Prophet not installed; please install from requirements.txt")
    if engine == "arima_sarima" and env["statsmodels"] == "NOT_INSTALLED":
        raise ImportError("statsmodels not installed")
    config_payload = {"revision_id": config["REVISION_ID"], "schema": SCHEMA,
                      "core_sha256": sha256_path(Path(__file__)),
                      "source_sha256": sha256_path(source_path), "model_spec": ARIMA_SPEC,
                      "env": env, "engine": engine, "dataset": dataset,
                      "min_reports": config["MIN_REPORTS_15Y"],
                      "zero_history_policy": config["ZERO_HISTORY_POLICY"]}
    fingerprint = hashlib.sha256(json.dumps(config_payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "RUN_MANIFEST.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise RuntimeError(f"Existing output differs in input/code/environment: {output_dir}. Use a NEW OUTPUT_ROOT.")
    else:
        atomic_json({"fingerprint": fingerprint, **config_payload,
                     "input_csv": str(source_path), "n_expected_events": len(source.columns)}, manifest_path)
    events = list(source.columns)
    requested = config.get("SELECTED_EVENTS")
    if requested is not None:
        missing = set(requested) - set(events)
        if missing:
            raise ValueError(f"Events not in eligible input CSV: {sorted(missing)}")
        events = [e for e in events if e in set(requested)]
    max_events = config.get("MAX_EVENTS_PER_DATASET")
    if max_events is not None:
        events = events[:max_events]
    print(f"\nSTART: {dataset} / {engine}; {len(events)}/{len(source.columns)} AE", flush=True)
    print("OUTPUT:", output_dir, flush=True)
    statuses = []
    for i,event in enumerate(events,1):
        print(f"[{i}/{len(events)}] {event}", flush=True)
        try:
            series = source[event].astype(float).rename(event)
            summary = run_event(series, dataset, engine,
                output_dir / safe_slug(event), fingerprint,
                config["RETRY_UNRESOLVED_ORIGINS"])
        except Exception as exc:
            print(f"  ERROR {event}: {type(exc).__name__}: {exc}",flush=True)
            traceback.print_exc(limit=2)
            summary = {"dataset":dataset,"engine":engine,"event":event,
                       "status":"FATAL_ERROR", "error":f"{type(exc).__name__}: {exc}"}
        statuses.append(summary)
        atomic_csv(pd.DataFrame(statuses), output_dir / "RUN_STATUS.csv")
    print(f"FINISHED {dataset}/{engine}; processed {len(statuses)} events",flush=True)
    return statuses
