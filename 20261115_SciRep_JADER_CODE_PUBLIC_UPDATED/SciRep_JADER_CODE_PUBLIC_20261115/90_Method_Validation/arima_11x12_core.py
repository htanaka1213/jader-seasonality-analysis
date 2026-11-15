# -*- coding: utf-8 -*-
"""11 rolling origins x 12 months: ARIMA vs SARIMA comparison.

Original methodological source: GitHub_SourceCode/run_arima_sarima_*_analysis.py
Only the CV split and downstream exports/diagnostics are changed.
Do not use exploratory CIs as validated definitive coverage.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "arima-sarima-aligned-11x12-v1"
YEARS = tuple(range(2009, 2020))
ARIMA_ORDERS = tuple(itertools.product(range(2), repeat=3))
SARIMA_ORDERS = tuple(itertools.product(range(2), repeat=3))
SEASONAL_ORDERS = tuple(itertools.product(range(2), repeat=3))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def slugify(event: str) -> str:
    head = re.sub(r"[^\w\-]+", "_", event, flags=re.UNICODE).strip("_")[:55]
    digest = hashlib.sha1(event.encode("utf-8")).hexdigest()[:10]
    return f"{head}_{digest}"


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False, encoding="utf-8-sig", float_format="%.17g")
    os.replace(temp, path)


def write_json_atomic(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_source(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"入力ファイルがありません: {path}")
    x = pd.read_csv(path, encoding="utf-8-sig")
    if "年月" not in x:
        raise ValueError("'年月' 列がありません")
    x["年月"] = pd.to_datetime(x["年月"].astype(str), format="%Y%m", errors="raise")
    if x["年月"].duplicated().any():
        raise ValueError("年月に重複があります")
    x = x.set_index("年月")
    expected = pd.date_range("2005-01-01", "2019-12-01", freq="MS")
    if not x.index.equals(expected):
        raise ValueError("2005-01～2019-12の180か月で、年月が昇順で連続している必要があります")
    # The submitted source retains all AE columns until a >=100 total filter is applied.
    return x


def validate_event(source: pd.DataFrame, event: str) -> pd.Series:
    if event not in source.columns:
        raise ValueError(f"AE列が存在しません: {event}")
    y = pd.to_numeric(source[event], errors="raise")
    if len(y) != 180 or y.isna().any() or (~np.isfinite(y.to_numpy(dtype=float))).any():
        raise ValueError(f"180か月の欠損なし数値系列が必要です: {event}")
    if (y < 0).any():
        raise ValueError(f"負の月次件数があります: {event}")
    if y.sum() < 100:
        raise ValueError(f"15年間の合計が100未満です: {event} ({y.sum()})")
    return y.astype(float)


def split_aligned_year(series: pd.Series, forecast_year: int):
    if forecast_year not in YEARS:
        raise ValueError("対象外の年")
    train = series.loc[:pd.Timestamp(forecast_year-1, 12, 1)]
    test = series.loc[pd.Timestamp(forecast_year, 1, 1):pd.Timestamp(forecast_year, 12, 1)]
    assert len(train) == 48 + 12*(forecast_year-2009)
    assert len(test) == 12
    assert train.index.max() == pd.Timestamp(forecast_year-1, 12, 1)
    assert test.index.equals(pd.date_range(f"{forecast_year}-01-01", periods=12, freq="MS"))
    return train, test


def original_aic_search(train: pd.Series, engine: str):
    """Exact submitted AIC search choices; do not change the candidate grids.

    Original SARIMAX: enforce_stationarity=False, enforce_invertibility=False;
    .fit(disp=False), no trend term, default optimizer/settings.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    best = None
    best_aic = np.inf
    best_order, best_seasonal = None, None
    failed, succeeded, nonconverged = 0, 0, 0
    start = time.perf_counter()
    if engine == "ARIMA":
        choices = ((order, (0, 0, 0, 0)) for order in ARIMA_ORDERS)
    elif engine == "SARIMA":
        choices = ((order, seasonal + (12,))
                   for order in SARIMA_ORDERS for seasonal in SEASONAL_ORDERS)
    else:
        raise ValueError(engine)
    # Original bare except:continue silently skips fits that raise.
    # Retain that behavior while recording counts and hiding repetitive warnings.
    for order, seasonal_order in choices:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False)
            succeeded += 1
            if not fit.mle_retvals.get("converged", True):
                nonconverged += 1
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                best, best_aic = fit, float(fit.aic)
                best_order, best_seasonal = order, seasonal_order
        except Exception:
            failed += 1
    return {
        "fit": best, "order": best_order, "seasonal_order": best_seasonal,
        "aic": best_aic, "bic": float(best.bic) if best is not None else np.nan,
        "candidates_succeeded": succeeded, "candidates_failed": failed,
        "candidates_nonconverged": nonconverged,
        "elapsed_sec": time.perf_counter() - start,
    }


def run_one_year(series: pd.Series, dataset: str, event: str, year: int, fingerprint: str):
    train, test = split_aligned_year(series, year)
    result = {}
    for engine in ("ARIMA", "SARIMA"):
        result[engine] = original_aic_search(train, engine)
    forecast = {}
    statuses = {}
    for engine in ("ARIMA", "SARIMA"):
        fit = result[engine]["fit"]
        try:
            if fit is None:
                raise RuntimeError("No finite-AIC candidate model succeeded")
            pred = np.asarray(fit.get_forecast(steps=len(test)).predicted_mean, dtype=float)
            if pred.shape != (12,) or not np.isfinite(pred).all():
                raise ValueError("Prediction missing/non-finite/wrong size")
            forecast[engine] = pred
            statuses[engine] = "ok"
        except Exception as exc:
            forecast[engine] = np.full(12, np.nan)
            statuses[engine] = f"FAILED: {type(exc).__name__}: {exc}"
    y = test.to_numpy(dtype=float)
    a, s = forecast["ARIMA"], forecast["SARIMA"]
    rows = pd.DataFrame({
        "dataset": dataset, "event": event, "origin_year": year,
        "origin_train_end": train.index.max().strftime("%Y-%m-%d"),
        "prophet_cutoff_equivalent": "year-end training endpoint (monthly data)",
        "ds": test.index.strftime("%Y-%m-%d"),
        "horizon_month": np.arange(1, 13), "y": y,
        "yhat_arima": a, "yhat_sarima": s,
        "ae_arima": np.abs(y-a), "ae_sarima": np.abs(y-s),
        "d_abs_error": np.abs(y-a)-np.abs(y-s),
        "se_arima": (y-a)**2, "se_sarima": (y-s)**2,
        "arima_order": str(result["ARIMA"]["order"]),
        "sarima_order": str(result["SARIMA"]["order"]),
        "sarima_seasonal_order": str(result["SARIMA"]["seasonal_order"]),
    })
    detail = {
        "schema": SCHEMA_VERSION, "config_fingerprint": fingerprint,
        "dataset": dataset, "event": event, "origin_year": year,
        "n_train": len(train), "n_test": 12,
        "train_start": str(train.index[0].date()),
        "train_end": str(train.index[-1].date()),
        "test_start": str(test.index[0].date()),
        "test_end": str(test.index[-1].date()),
        "status_arima": statuses["ARIMA"],
        "status_sarima": statuses["SARIMA"],
    }
    for engine in ("ARIMA", "SARIMA"):
        fit = result[engine]
        detail.update({
            f"{engine.lower()}_order": str(fit["order"]),
            f"{engine.lower()}_seasonal_order": str(fit["seasonal_order"]),
            f"{engine.lower()}_aic": fit["aic"] if np.isfinite(fit["aic"]) else None,
            f"{engine.lower()}_bic": fit["bic"] if np.isfinite(fit["bic"]) else None,
            f"{engine.lower()}_candidates_succeeded": fit["candidates_succeeded"],
            f"{engine.lower()}_candidates_failed": fit["candidates_failed"],
            f"{engine.lower()}_candidates_nonconverged": fit["candidates_nonconverged"],
            f"{engine.lower()}_fit_seconds": fit["elapsed_sec"],
        })
    if all(st == "ok" for st in statuses.values()):
        detail["mae_arima"] = float(np.mean(np.abs(y-a)))
        detail["mae_sarima"] = float(np.mean(np.abs(y-s)))
        detail["delta_mae"] = detail["mae_arima"]-detail["mae_sarima"]
        detail["complete"] = True
    else:
        detail["complete"] = False
    return rows, detail


def diagnostic_ci(x: np.ndarray):
    """Exploratory year-wise diagnostics only; NOT a validated primary CI."""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    if len(x) != 11 or not np.isfinite(x).all():
        raise ValueError("11 nonmissing yearly ΔMAE values required")
    mean = float(x.mean())
    se = float(x.std(ddof=1)/np.sqrt(len(x)))
    if se == 0:
        low = high = mean
        p_two = 0.0 if mean != 0 else 1.0
    else:
        tcrit = stats.t.ppf(.975, len(x)-1)
        low, high = mean-tcrit*se, mean+tcrit*se
        p_two = float(2*stats.t.sf(abs(mean/se), len(x)-1))
    npositive = int((x > 0).sum())
    nnegative = int((x < 0).sum())
    n_nonzero = npositive + nnegative
    p_sign = float(stats.binomtest(npositive, n_nonzero, .5).pvalue) if n_nonzero else np.nan
    return {
        "group_t95_low_exploratory": low,
        "group_t95_high_exploratory": high,
        "group_t_p_two_sided_exploratory": p_two,
        "group_t_se_exploratory": se,
        "year_positive_count": npositive,
        "year_negative_count": nnegative,
        "year_sign_test_p_two_sided": p_sign,
        "leave_one_year_out_delta_min": float(min(np.delete(x,i).mean() for i in range(len(x)))),
        "leave_one_year_out_delta_max": float(max(np.delete(x,i).mean() for i in range(len(x)))),
        "year_lag1_autocorr": float(pd.Series(x).autocorr(lag=1)),
    }


def summarize_one(paired: pd.DataFrame, years: list[dict], dataset: str, event: str,
                  input_sha256: str, fingerprint: str):
    ordered = paired.sort_values("ds").reset_index(drop=True)
    expected = pd.date_range("2009-01-01", "2019-12-01", freq="MS")
    dates = pd.DatetimeIndex(pd.to_datetime(ordered["ds"]))
    if not (len(ordered)==132 and dates.is_unique and dates.equals(expected)
            and ordered["d_abs_error"].notna().all()):
        raise ValueError("132 months Jan 2009 through Dec 2019, one prediction per month, required")
    g = ordered.groupby("origin_year", sort=True).agg(
        n_months=("ds","size"), mae_arima=("ae_arima","mean"),
        mae_sarima=("ae_sarima","mean"), delta_mae=("d_abs_error","mean")
    ).reset_index()
    if g["origin_year"].tolist() != list(YEARS) or not (g["n_months"] == 12).all():
        raise ValueError("Each of 11 years must have 12 months")
    a = float(ordered["ae_arima"].mean())
    s = float(ordered["ae_sarima"].mean())
    d = float(ordered["d_abs_error"].mean())
    assert np.isclose(d,a-s,atol=1e-10)
    if not np.isclose(d,float(g["delta_mae"].mean()),atol=1e-10):
        raise AssertionError("monthly/yearly delta mismatch")
    tdiag=diagnostic_ci(g["delta_mae"].to_numpy(float))
    joined_prophet={}
    row = {
        "dataset": dataset, "engine": "arima_sarima_aligned", "event": event,
        "n_reports_2005_2019": None, # caller fills
        "input_sha256": input_sha256,
        "config_fingerprint": fingerprint,
        "n_origins": 11, "n_predicted_months": 132,
        "first_month": "2009-01", "last_month": "2019-12",
        "mae_arima_raw": a, "mae_sarima_raw": s,
        "delta_mae_arima_minus_sarima_raw": d,
        "relative_mae_improvement": d/a if a else np.nan,
        "rmse_arima_raw": float(np.sqrt(ordered["se_arima"].mean())),
        "rmse_sarima_raw": float(np.sqrt(ordered["se_sarima"].mean())),
        "all_11_origins_valid": True,
        **tdiag
    }
    return row, g


def verify_prophet_alignment(pair_path: Path, arima_paired: pd.DataFrame):
    """Optional record-level alignment check, never blend one model with other engine's errors."""
    prophet = pd.read_csv(pair_path, encoding="utf-8-sig")
    required = {"ds", "y", "d_abs_error", "ae_seasonal", "ae_nonseasonal"}
    if not required.issubset(prophet.columns):
        raise ValueError(f"Prophetペアに列がありません: {required-set(prophet.columns)}")
    pp=prophet.copy()
    aa=arima_paired.copy()
    pp["ds"]=pd.to_datetime(pp["ds"])
    aa["ds"]=pd.to_datetime(aa["ds"])
    if pp["ds"].duplicated().any() or aa["ds"].duplicated().any():
        raise ValueError("Prophet/ARIMA の予測月に重複があります")
    both=aa.merge(pp[["ds","y","ae_seasonal","ae_nonseasonal","d_abs_error"]],
                  on="ds",how="outer",validate="one_to_one",indicator=True,
                  suffixes=("_arima","_prophet"))
    if (len(both)!=132 or (both["_merge"]!="both").any() or
        not np.allclose(both["y_arima"], both["y_prophet"],rtol=0,atol=1e-8)):
        raise ValueError("ProphetとARIMAの予測月または実測値が一致しません")
    both=both.drop(columns=["_merge"])
    both=both.rename(columns={"d_abs_error_arima":"d_arima_minus_sarima",
                              "d_abs_error_prophet":"d_prophet_nonseason_minus_season"})
    return both
