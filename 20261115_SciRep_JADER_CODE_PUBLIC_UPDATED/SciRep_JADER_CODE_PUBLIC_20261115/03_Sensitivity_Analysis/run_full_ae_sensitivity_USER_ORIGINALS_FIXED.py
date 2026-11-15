# -*- coding: utf-8 -*-
"""JADER full-AE ARIMA/SARIMA model-selection sensitivity analysis.

Scientific Reports revision, November 2026.
Author-ready, auditable cross-validation: 11 expanding origins (2009--2019),
12 forecast months per origin; year-wise DeltaMAE t CI, Prophet + ARIMA/SARIMA
IUT and per-dataset BH correction.

Crucially: alternate differencing specifications are NOT compared by AIC.
Within each (scenario, year, family), AIC is only compared between models
with identical effective-likelihood sample size. Across scenarios, comparisons
use truly held-out prediction error.

README_ユーザー原本の実行方法.md documents study design, limitations and setup.
"""
from __future__ import annotations

# Set BLAS thread caps before importing numpy/statsmodels to avoid CPU oversubscription.
import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels

# -----------------------------------------------------------
# SPYDER SETTINGS: edit these four fields, then press F5.
# -----------------------------------------------------------
RUN_MODE = "validate"  # "validate" -> "pilot" -> "all" -> "collect".
SCENARIO_NAMES = ["d1D0_reference", "d0D0_intercept", "d1D1_seasonal_difference"]
DATASET_NAMES = ["overall", "non_vaccine", "vaccine"]
WORKERS = 1  # Spyder/Windows: 1 is safest. Command line: --workers 2 (or 3).
PILOT_AE_PER_DATASET = 1

# Place this script in the folder with all six supplied CSV files.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
RESULTS_DIR = ROOT.parent / "outputs" / "sensitivity"

INPUT_FILES = {
    "overall": "20250112_yearly_counts_filtered.csv",
    "non_vaccine": "20260411_yearly_counts_filtered_non_vaccine.csv",
    "vaccine": "20260411_yearly_counts_filtered_vaccine_only.csv",
}
PROPHET_FILES = {
    "overall": "INFERENCE_BY_METHOD_overall.csv",
    "non_vaccine": "INFERENCE_BY_METHOD_non_vaccine.csv",
    "vaccine": "INFERENCE_BY_METHOD_vaccine.csv",
}
# --- 2026-11-12: use the USER'S SIX ORIGINAL FILES directly on Windows. ---
# All six originals uploaded to this conversation were byte-for-byte identical
# to the input files in the previous ZIP (SHA-256 verification).
USE_USER_ORIGINAL_PATHS = False   # True: Windows originals; False: bundled input/ CSVs
VERIFY_ORIGINAL_SHA256 = False    # Require exact match with files verified 2026-11-12

USER_ORIGINAL_PATHS = {"monthly": {}, "inference": {}}  # local mode only

# SHA-256 fingerprints calculated from the six files the user uploaded.
EXPECTED_SOURCE_SHA256 = {
    "monthly": {
        "overall": "40a5eea83ae93646a3476f18aa58923a0927dcbf66b515f2ff9fc4b57416c77b",
        "non_vaccine": "d6c0b42e6021508c87476c2fb86db332909f908b45a56f23b38d2af71a142bc8",
        "vaccine": "a82b4348b61fa8f14d10a37bd38153b0130e50500b8de1efb399e602cd4c7a52",
    },
    "inference": {
        "overall": "b41fb9001702af0de88dbbbcba01eaefed455a915b53577303f20310b9a1ed3e",
        "non_vaccine": "5383b2fa92a75c3b59c7f42b9849d9e0c9b543c1e8a199b3b7b4f99d5e530bf8",
        "vaccine": "ec3febed8159d08980ae789d0f6c8daa258efb51df5d7f12efeffc5612c421ce",
    },
}


def source_file(kind: str, dataset: str) -> Path:
    """Canonical input resolver. Never silently substitute another dataset."""
    if kind not in ("monthly", "inference") or dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown source kind/dataset: {kind}/{dataset}")
    if USE_USER_ORIGINAL_PATHS:
        path = Path(USER_ORIGINAL_PATHS[kind][dataset])
    elif kind == "monthly":
        path = DATA_DIR / "monthly" / INPUT_FILES[dataset]
    else:
        path = DATA_DIR / "inference" / PROPHET_FILES[dataset]
        if not path.is_file():
            path = ROOT.parent / "outputs" / "main" / PROPHET_FILES[dataset]
    if not path.is_file():
        raise FileNotFoundError(f"指定した原本ファイルがありません: {path}")
    if VERIFY_ORIGINAL_SHA256:
        actual = sha256_file(path)
        expected = EXPECTED_SOURCE_SHA256[kind][dataset]
        if actual != expected:
            raise ValueError(
                f"原本のハッシュ値がアップロード時と一致しません: {path}\n"
                f"Expected SHA256={expected}\nActual SHA256={actual}\n"
                "元データの内容が更新された可能性があります。解析を停止して比較してください。"
            )
    return path

EXPECTED_TOTAL_AE = {"overall": 727, "non_vaccine": 698, "vaccine": 51}
EXPECTED_ZERO48 = {"overall": 7, "non_vaccine": 6, "vaccine": 4}
YEAR_START, YEAR_END = 2009, 2019
FORECAST_MONTHS = 12
MIN_REPORTS_15Y = 100
ALPHA = 0.05
# Flag implausible forecasts *without silently changing their values*.
# A forecast >20 x max(1, historical maximum), absolute, prompts examination.
EXTREME_FORECAST_RATIO = 20.0
MAXITER = 50
REJECT_DEGENERATE_ZERO_LOGLIK = True

# Each scenario fixes a coherent differencing specification.
# d=0 stationary models include an intercept; integrated d=1 models no drift.
# Sensitivity: seasonal differencing D=1 applies only to SARIMA family;
# nonseasonal ARIMA retains seasonal_order=(0,0,0,0).
SCENARIOS = {
    "d1D0_reference": {
        "d": 1, "D": 0, "trend": "n", "pq": [0, 1], "PQ": [0, 1],
        "description": "Re-fit primary specification d=1,D=0, no drift; 4 ARIMA + 12 SARIMA",
    },
    "d0D0_intercept": {
        "d": 0, "D": 0, "trend": "c", "pq": [0, 1], "PQ": [0, 1],
        "description": "Alternative stationary-level d=0,D=0 with intercept; 4+12",
    },
    "d1D1_seasonal_difference": {
        "d": 1, "D": 1, "trend": "n", "pq": [0, 1], "PQ": [0, 1],
        "description": "Alternative SARIMA d=1,D=1; ARIMA d=1; no drift; 4+12",
    },
    "d1D0_expanded_QC": {
        "d": 1, "D": 0, "trend": "n", "pq": [0, 1, 2], "PQ": [0, 1],
        "description": "Optional expanded p,q=0,1,2; P,Q=0,1; 9+27 with numerical QC",
    },
}

# -------------------------------------
# Stable output/configuration utilities
# -------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_config(spec: dict, input_sha: str, event: str) -> str:
    defining = {
        "spec": spec, "input_sha": input_sha, "event": event,
        "start": YEAR_START, "end": YEAR_END, "forecast_months": FORECAST_MONTHS,
        "maxiter": MAXITER, "reject_zero_llf": REJECT_DEGENERATE_ZERO_LOGLIK,
        "statsmodels_version": statsmodels.__version__,
        "scipy_version": __import__("scipy").__version__,
        "numpy_version": np.__version__,
    }
    return hashlib.sha256(json.dumps(defining, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def dump_atomic(path: Path, obj: dict) -> None:
    import time
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False, indent=None)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(12):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))   # OneDrive などのロックが外れるのを待つ
    os.replace(temp, path)  # それでも失敗する場合はエラーで停止


def identity_case(dataset: str, event: str) -> str:
    return hashlib.sha256((dataset + "::" + event).encode("utf-8")).hexdigest()[:20]


def checkpoint_path(scenario: str, dataset: str, event: str) -> Path:
    return RESULTS_DIR / "checkpoints" / scenario / dataset / (identity_case(dataset, event) + ".json")


def read_input(dataset: str) -> tuple[pd.DataFrame, str]:
    path = source_file("monthly", dataset)
    if not path.is_file():
        raise FileNotFoundError(f"入力ファイルなし: {path}")
    sha = sha256_file(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.columns[0] != "年月":
        raise ValueError(f"先頭列は'年月'が必要です: {path}")
    idx = pd.to_datetime(df["年月"].astype(str), format="%Y%m", errors="raise")
    expected = pd.date_range("2005-01-01", "2019-12-01", freq="MS")
    if len(df) != 180 or not idx.equals(pd.Series(expected)):
        raise ValueError("月次系列が2005-01～2019-12の180か月と一致しません")
    df.index = expected
    sums = df.drop(columns=["年月"]).sum(numeric_only=True)
    events = list(sums.index[sums >= MIN_REPORTS_15Y])
    if len(events) != EXPECTED_TOTAL_AE[dataset]:
        raise ValueError(f"対象AE件数不一致: {dataset}: {len(events)} != {EXPECTED_TOTAL_AE[dataset]}")
    initial_zero = sum(float(df[e].iloc[:48].sum()) == 0 for e in events)
    if initial_zero != EXPECTED_ZERO48[dataset]:
        raise ValueError(f"初期48か月0件のAE数不一致: {initial_zero} != {EXPECTED_ZERO48[dataset]}")
    for name in events:
        s = pd.to_numeric(df[name], errors="raise")
        if s.isna().any() or not np.isfinite(s.to_numpy(float)).all() or (s < 0).any():
            raise ValueError(f"AEの欠損/非有限/負の報告件数: {dataset}, {name}")
    print(f"[INPUT] {dataset}: {len(events)} AEs, {initial_zero} zero-history; SHA256={sha[:12]}", flush=True)
    return df[events], sha


def candidate_orders(seasonal: bool, spec: dict):
    pqs = spec["pq"]
    for p, q in itertools.product(pqs, pqs):
        order = (p, int(spec["d"]), q)
        if seasonal:
            for P, Q in itertools.product(spec["PQ"], spec["PQ"]):
                if P + Q == 0:  # force a genuine 12-month seasonal parameter
                    continue
                yield order, (P, int(spec["D"]), Q, 12)
        else:
            yield order, (0, 0, 0, 0)


def _fitted_variance(fit) -> float | None:
    names = list(fit.param_names)
    if "sigma2" in names:
        return float(np.asarray(fit.params)[names.index("sigma2")])
    return None


def select_and_forecast(train: np.ndarray, family: str, spec: dict) -> dict:
    """Select by AIC *within* same d/D model family & effective sample.

    Forecasts are kept on the original report-count scale and are NOT clipped.
    Zero-loglikelihood degenerate fits are rejected using training-only QC.
    """
    use_seasonal = family == "SARIMA"
    n_candidates = 0
    counts = {"converged": 0, "nonconverged": 0, "fit_exception": 0,
              "nonfinite_aic": 0, "bad_effective_n": 0,
              "zero_llf_degenerate": 0, "invalid_variance": 0,
              "forecast_failed": 0}
    best = None
    best_aic = float("inf")
    selected = None
    accepted_burns = set()
    expected_burn = int(spec["d"]) + (12 * int(spec["D"]) if use_seasonal else 0)

    for order, seasonal_order in candidate_orders(use_seasonal, spec):
        n_candidates += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = SARIMAX(
                    train, order=order, seasonal_order=seasonal_order,
                    trend=spec["trend"], simple_differencing=False,
                    enforce_stationarity=True, enforce_invertibility=True,
                )
                fit = m.fit(disp=False, maxiter=MAXITER)
        except Exception:
            counts["fit_exception"] += 1
            continue
        if not bool(fit.mle_retvals.get("converged", False)):
            counts["nonconverged"] += 1
            continue
        if not np.isfinite(fit.aic) or not np.isfinite(fit.llf):
            counts["nonfinite_aic"] += 1
            continue
        burn = int(fit.loglikelihood_burn)
        eff = int(fit.nobs_effective)
        if burn != expected_burn or eff != len(train) - expected_burn:
            counts["bad_effective_n"] += 1
            continue
        variance = _fitted_variance(fit)
        if variance is not None and (not np.isfinite(variance) or variance <= 0):
            counts["invalid_variance"] += 1
            continue
        if REJECT_DEGENERATE_ZERO_LOGLIK and float(fit.llf) == 0.0 and float(np.std(train)) > 0:
            # AIC ≈ 2*k with nonconstant data can be a numerical filtering failure.
            # Investigate separately; do not let it win the AIC search.
            counts["zero_llf_degenerate"] += 1
            continue
        counts["converged"] += 1
        accepted_burns.add(burn)
        if float(fit.aic) < best_aic:
            best = fit
            best_aic = float(fit.aic)
            selected = {"order": list(order), "seasonal_order": list(seasonal_order),
                        "aic": best_aic, "llf": float(fit.llf),
                        "effective_n": eff, "burn": burn, "sigma2": variance}

    if best is None:
        return {"status": "NO_ACCEPTED_MODEL", "n_candidates": n_candidates,
                "counts": counts, "expected_burn": expected_burn}
    if len(accepted_burns) != 1:
        raise AssertionError("AIC比較に異なる有効標本数を混入させています")
    try:
        forecast = np.asarray(best.get_forecast(steps=FORECAST_MONTHS).predicted_mean, dtype=float)
    except Exception as exc:
        counts["forecast_failed"] += 1
        return {"status": "FORECAST_ERROR", "error": str(exc),
                "n_candidates": n_candidates, "counts": counts}
    if forecast.shape != (FORECAST_MONTHS,) or not np.isfinite(forecast).all():
        return {"status": "NONFINITE_FORECAST", "n_candidates": n_candidates,
                "counts": counts, "selected": selected}
    scale = max(1.0, float(train.max()))
    extreme = bool(np.max(np.abs(forecast)) > EXTREME_FORECAST_RATIO * scale)
    # Do not alter or remove extreme predictions; merely flag them for audit.
    return {"status": "OK", "forecast": forecast.tolist(),
            "n_candidates": n_candidates, "counts": counts, "selected": selected,
            "qc_extreme_forecast": extreme, "qc_max_abs_prediction_over_max_train": float(np.max(np.abs(forecast))/scale)}


def _empty_or_load_case(path: Path, dataset: str, event: str,
                        scenario: str, spec: dict, source_sha: str, y: np.ndarray) -> dict:
    config_hash = digest_config(spec, source_sha, event)
    if path.is_file():
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("config_sha256") != config_hash:
            raise RuntimeError(f"チェックポイントの条件が変わりました。上書きしません: {path}")
        return obj
    obj = {"dataset": dataset, "event": event, "scenario": scenario,
           "config_sha256": config_hash, "data_sha256": source_sha,
           "n_reports": int(y.sum()), "initial_48_reports": int(y[:48].sum()),
           "year_results": {}, "status": "IN_PROGRESS"}
    if float(y[:48].sum()) == 0:
        obj["status"] = "UNTESTABLE_ZERO_INITIAL_48"
    dump_atomic(path, obj)
    return obj


def work_one(task: tuple) -> dict:
    scenario, dataset, event, ylist, input_sha = task
    spec = SCENARIOS[scenario]
    y = np.asarray(ylist, dtype=float)
    path = checkpoint_path(scenario, dataset, event)
    obj = _empty_or_load_case(path, dataset, event, scenario, spec, input_sha, y)
    if obj["status"] == "UNTESTABLE_ZERO_INITIAL_48":
        return {"dataset": dataset, "event": event, "scenario": scenario,
                "status": obj["status"], "year_count": 0}
    if len(y) != 180 or not np.isfinite(y).all() or np.min(y) < 0:
        raise ValueError(f"invalid AE series: {dataset}/{event}")
    for yr in range(YEAR_START, YEAR_END+1):
        key = str(yr)
        if obj["year_results"].get(key, {}).get("status") == "OK":
            continue
        n_train = 48 + 12*(yr - YEAR_START)
        train, actual = y[:n_train], y[n_train:n_train+12]
        if len(actual) != 12 or float(train.sum()) <= 0:
            raise RuntimeError(f"Unexpected zero-history or year length: {dataset}/{event}/{yr}")
        arima = select_and_forecast(train, "ARIMA", spec)
        sarima = select_and_forecast(train, "SARIMA", spec)
        if arima["status"] == sarima["status"] == "OK":
            e0 = np.abs(actual-np.asarray(arima["forecast"],dtype=float))
            e1 = np.abs(actual-np.asarray(sarima["forecast"],dtype=float))
            rec = {"status": "OK", "train_months": len(train),
                   "observed": actual.tolist(),
                   "mae_arima": float(e0.mean()), "mae_sarima": float(e1.mean()),
                   "delta_mae": float((e0-e1).mean()),
                   "arima": arima, "sarima": sarima,
                   "any_extreme_forecast": bool(arima["qc_extreme_forecast"] or sarima["qc_extreme_forecast"])}
        else:
            rec = {"status": "FIT_OR_FORECAST_FAILED", "train_months": len(train),
                   "arima": arima, "sarima": sarima}
        obj["year_results"][key] = rec
        obj["status"] = "IN_PROGRESS"
        dump_atomic(path, obj)
        print(f"[CHECKPOINT] {scenario} / {dataset} / {event} / {yr}: {rec['status']}",flush=True)
    successes = sum(o.get("status")=="OK" for o in obj["year_results"].values())
    obj["status"] = "COMPLETE" if successes == 11 else "INCOMPLETE_REVIEW_FAILURES"
    dump_atomic(path, obj)
    return {"dataset": dataset, "event": event, "scenario": scenario,
            "status": obj["status"], "year_count": successes}


def yearwise_stats(year_delta: np.ndarray) -> dict:
    """Year-level DeltaMAE inference; undefined t-statistic is NOT p=0.

    A constant series of year-level deltas has zero estimated standard error.
    This is not statistically testable using the prespecified year-level t test.
    """
    assert len(year_delta) == 11 and np.isfinite(year_delta).all()
    mean = float(np.mean(year_delta))
    sd = float(np.std(year_delta, ddof=1))
    se = sd / math.sqrt(11)
    common = {"delta_mae":mean,
              "n_positive_years":int(np.sum(year_delta>0)),
              "n_negative_years":int(np.sum(year_delta<0)),
              "year_sd":sd,"year_se":se}
    if not np.isfinite(se) or se == 0.0:
        # No inferential p-value/CI; keep descriptive delta for audit only.
        return {**common,"inference_status":"UNTESTABLE_ZERO_VARIANCE",
                "ci95_low":float("nan"),"ci95_high":float("nan"),
                "p_two_sided":float("nan"),
                "p_positive_direction":float("nan"),
                "nominal_positive":False,"nominal_negative":False}
    low = float(mean-stats.t.ppf(.975,10)*se)
    high = float(mean+stats.t.ppf(.975,10)*se)
    p2 = float(2*stats.t.sf(abs(mean/se),10))
    return {**common,"inference_status":"INFERENCE_READY",
            "ci95_low":low,"ci95_high":high,
            "p_two_sided":p2,
            "p_positive_direction":p2 if mean>0 else 1.0,
            "nominal_positive":bool(low>0),
            "nominal_negative":bool(high<0)}


def load_prophet_summary(dataset: str) -> pd.DataFrame:
    file = source_file("inference", dataset)
    if not file.is_file():
        raise FileNotFoundError(f"Prophet既存結果なし: {file}")
    df = pd.read_csv(file, encoding="utf-8-sig")
    df = df[df["engine"].astype(str).str.lower() == "prophet"].copy()
    if len(df) != EXPECTED_TOTAL_AE[dataset] or df["event"].duplicated().any():
        raise ValueError(f"Prophet集計が期待件数と異なります: {dataset}")
    return df.set_index("event")


def collect_data(selected_scenarios: list[str], selected_datasets: list[str],
                 allow_partial: bool = True) -> None:
    """Rebuild all exports from saved origin-level checkpoints.

    BH q-values are supplied only after every AE is in a terminal state:
    tested, initial-zero ineligible, zero-variance ineligible, or fit failed.
    In-progress and never-started AEs block BH calculation. Failed AEs are
    excluded from the valid-test BH family and their number is reported.
    """
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    rows=[]; years=[]; model_audits=[]
    per_dataset_source = {}
    source_hashes={}
    for dataset in selected_datasets:
        df,sha=read_input(dataset)
        per_dataset_source[dataset]=df
        source_hashes[dataset]=sha
    for scenario in selected_scenarios:
        spec=SCENARIOS[scenario]
        for dataset in selected_datasets:
            df=per_dataset_source[dataset]
            for event in df.columns:
                y=df[event].to_numpy(float)
                file=checkpoint_path(scenario,dataset,event)
                base={"scenario":scenario,"dataset":dataset,"event":event,
                      "n_reports":int(y.sum()),"initial_48_reports":int(y[:48].sum())}
                if not file.exists():
                    rows.append(dict(base,status="NOT_RUN"))
                    continue
                obj=json.loads(file.read_text(encoding="utf-8"))
                if obj.get("config_sha256")!=digest_config(spec,source_hashes[dataset],event):
                    raise RuntimeError(f"チェックポイント環境・データ不一致: {file}")
                if obj["status"]=="UNTESTABLE_ZERO_INITIAL_48":
                    rows.append(dict(base,status="UNTESTABLE_ZERO_INITIAL_48"))
                    continue
                yearly=obj["year_results"]
                years_completed = len(yearly)==11 and all(str(yr) in yearly for yr in range(2009,2020))
                years_all_ok = years_completed and all(yearly[str(yr)].get('status')=='OK' for yr in range(2009,2020))
                if obj.get('status') == 'COMPLETE' and not years_all_ok:
                    raise RuntimeError(f"破損したCOMPLETEチェックポイント: {file}")
                if not years_all_ok or obj.get('status') != 'COMPLETE':
                    # Only a formally terminal failure is excluded from BH.
                    # An unfinished/restartable AE remains IN_PROGRESS and blocks BH.
                    terminal_failure = bool(obj.get('status') == 'INCOMPLETE_REVIEW_FAILURES'
                                            and years_completed and not years_all_ok)
                    status = 'INCOMPLETE_OR_FAILED' if terminal_failure else 'IN_PROGRESS'
                    rows.append(dict(base,status=status,
                                     n_success_years=sum(v.get('status')=='OK' for v in yearly.values()),
                                     n_failed_years=sum(v.get('status') not in ('OK',) for v in yearly.values()) if terminal_failure else np.nan))
                    # Include failed AEs in the model-selection/QC audit too.
                    for ykey, obs in yearly.items():
                        for family, key in [('ARIMA','arima'),('SARIMA','sarima')]:
                            model = obs.get(key, {})
                            if not model:continue
                            sel = model.get('selected') or {}
                            model_audits.append(dict(base,year=int(ykey),family=family,
                                year_status=obs.get('status'),model_status=model.get('status'),
                                order=str(sel.get('order')),seasonal_order=str(sel.get('seasonal_order')),
                                aic=sel.get('aic'),llf=sel.get('llf'),
                                nobs_effective=sel.get('effective_n'),
                                loglikelihood_burn=sel.get('burn'),
                                n_candidates=model.get('n_candidates'),
                                accepted=model.get('counts',{}).get('converged'),
                                nonconverged=model.get('counts',{}).get('nonconverged'),
                                zero_llf_degenerate=model.get('counts',{}).get('zero_llf_degenerate'),
                                invalid_variance=model.get('counts',{}).get('invalid_variance'),
                                qc_extreme_forecast=model.get('qc_extreme_forecast'),
                                max_forecast_vs_train=model.get('qc_max_abs_prediction_over_max_train')))
                    continue
                deltas=np.asarray([yearly[str(yr)]["delta_mae"] for yr in range(2009,2020)],dtype=float)
                inf=yearwise_stats(deltas)
                if inf['inference_status'] != 'INFERENCE_READY':
                    rows.append(dict(base,status=inf['inference_status'],
                                     n_success_years=11,n_prediction_months=132,**inf))
                    continue
                months=[]
                extreme_years=[]
                for yr in range(2009,2020):
                    obs=yearly[str(yr)]
                    months.extend((yr, i+1,float(obs["observed"][i]),
                        float(obs['arima']['forecast'][i]),float(obs['sarima']['forecast'][i])) for i in range(12))
                    years.append(dict(base,year=yr,delta_mae=obs["delta_mae"],
                        mae_arima=obs["mae_arima"],mae_sarima=obs["mae_sarima"],
                        qc_extreme_forecast=obs['any_extreme_forecast']))
                    if obs['any_extreme_forecast']:extreme_years.append(yr)
                    for family,key in [("ARIMA","arima"),("SARIMA","sarima")]:
                        model=obs[key]
                        sel=model.get('selected',{})
                        model_audits.append(dict(base,year=yr,family=family,
                            year_status='OK',model_status=model.get('status'),
                            order=str(sel.get('order')),seasonal_order=str(sel.get('seasonal_order')),
                            aic=sel.get('aic'),llf=sel.get('llf'),
                            nobs_effective=sel.get('effective_n'),
                            loglikelihood_burn=sel.get('burn'),
                            n_candidates=model.get('n_candidates'),
                            accepted=model.get('counts',{}).get('converged'),
                            nonconverged=model.get('counts',{}).get('nonconverged'),
                            zero_llf_degenerate=model.get('counts',{}).get('zero_llf_degenerate'),
                            invalid_variance=model.get('counts',{}).get('invalid_variance'),
                            qc_extreme_forecast=model.get('qc_extreme_forecast'),
                            max_forecast_vs_train=model.get('qc_max_abs_prediction_over_max_train')))
                arr=np.asarray(months,object)
                obs=np.array([q[2] for q in months]); pr0=np.array([q[3] for q in months]); pr1=np.array([q[4] for q in months])
                m0=float(np.abs(obs-pr0).mean()); m1=float(np.abs(obs-pr1).mean())
                if not np.isclose(m0-m1,inf['delta_mae'],rtol=1e-7,atol=1e-7):
                    raise AssertionError(f"132か月と11年のΔMAEが不一致: {scenario}/{dataset}/{event}")
                rows.append(dict(base,status="INFERENCE_READY",n_success_years=11,
                    n_prediction_months=132,mae_arima=m0,mae_sarima=m1,
                    qc_extreme_forecast_years=len(extreme_years),
                    qc_extreme_year_list=','.join(map(str,extreme_years)),
                    **{k:v for k,v in inf.items() if k != 'inference_status'}))
    s=pd.DataFrame(rows)
    if s.empty:
        raise RuntimeError("集計対象なし")
    proph_all=pd.concat([load_prophet_summary(d).assign(dataset=d) for d in selected_datasets],axis=0)
    proph_all=proph_all.reset_index().rename(columns={'index':'event'})
    proph_all=proph_all[['dataset','event','status','delta_mae','ci95_low','ci95_high','p_two_sided','p_direction_adjusted_two_sided']]
    proph_all=proph_all.rename(columns={'status':'prophet_status','delta_mae':'prophet_delta_mae',
       'ci95_low':'prophet_ci95_low','ci95_high':'prophet_ci95_high',
       'p_two_sided':'prophet_p_two_sided','p_direction_adjusted_two_sided':'prophet_p_positive_direction'})
    s=s.merge(proph_all,on=['dataset','event'],how='left',validate='many_to_one')
    if s['prophet_status'].isna().any():
        raise RuntimeError('Prophet結果のAE名が一致しません')
    s['p_IUT_joint']=np.nan
    s['q_BH_dataset_IUT']=np.nan
    s['nominal_IUT_positive']=False
    s['fdr_joint_q_lt_005']=False
    s['group_fdr_complete']=False
    group_records=[]
    for (scenario,dataset),ix in s.groupby(['scenario','dataset']).groups.items():
        indices=list(ix); part=s.loc[indices]
        valid=part['status'].eq('INFERENCE_READY') & part['prophet_status'].eq('INFERENCE_READY')
        both=part[valid]
        # Initial-zero and estimation failures are not testable, and are
        # excluded from the BH family. We do not silently treat missing as p=0.
        correct_n=EXPECTED_TOTAL_AE[dataset]-EXPECTED_ZERO48[dataset]
        terminal_statuses={"INFERENCE_READY", "UNTESTABLE_ZERO_INITIAL_48",
                           "INCOMPLETE_OR_FAILED", "UNTESTABLE_ZERO_VARIANCE"}
        n_zero=int(part['status'].eq('UNTESTABLE_ZERO_INITIAL_48').sum())
        n_failed=int(part['status'].eq('INCOMPLETE_OR_FAILED').sum())
        n_zero_sd=int(part['status'].eq('UNTESTABLE_ZERO_VARIANCE').sum())
        n_pending=int(part['status'].eq('NOT_RUN').sum())
        n_in_progress=int(part['status'].eq('IN_PROGRESS').sum())
        complete=bool(len(part)==EXPECTED_TOTAL_AE[dataset] and
                      n_zero==EXPECTED_ZERO48[dataset] and
                      part['status'].isin(terminal_statuses).all() and
                      n_pending==0 and n_in_progress==0 and
                      len(both)+n_failed+n_zero_sd==correct_n)
        if complete and bool((part.loc[part.status=='INFERENCE_READY','prophet_status'] != 'INFERENCE_READY').any()):
            raise RuntimeError(f'Prophet inferential status mismatch: {scenario}/{dataset}')
        if complete:
            s.loc[indices,'group_fdr_complete']=True
        if len(both):
            p_arima=both['p_positive_direction'].to_numpy(float)
            p_prophet=both['prophet_p_positive_direction'].to_numpy(float)
            joint=np.maximum(p_arima,p_prophet)
            if not np.isfinite(joint).all():
                raise RuntimeError(f'nonfinite joint p: {scenario}/{dataset}')
            s.loc[both.index,'p_IUT_joint']=joint
            s.loc[both.index,'nominal_IUT_positive']=(both['ci95_low'].to_numpy(float)>0) & (both['prophet_ci95_low'].to_numpy(float)>0)
            if complete:
                _,q,_,_=multipletests(joint,alpha=ALPHA,method='fdr_bh')
                s.loc[both.index,'q_BH_dataset_IUT']=q
                s.loc[both.index,'fdr_joint_q_lt_005']=q<ALPHA
        group_records.append(dict(scenario=scenario,dataset=dataset,n_total=len(part),n_valid=len(both),
            n_expected_inferable=correct_n,n_zero_initial=n_zero,
            n_pending=n_pending,n_in_progress=n_in_progress,
            n_failed=n_failed,n_zero_variance=n_zero_sd,
            group_complete=complete,
            n_nominal_ARIMA_SARIMA_improved=int((part.ci95_low>0).sum()),
            n_nominal_ARIMA_SARIMA_worse=int((part.ci95_high<0).sum()),
            n_nominal_IUT_positive=int(s.loc[indices,'nominal_IUT_positive'].sum()),
            n_FDR_IUT_positive=int(s.loc[indices,'fdr_joint_q_lt_005'].sum()) if complete else None,
            n_ae_with_extreme_forecast=int((part.qc_extreme_forecast_years>0).sum())))
    s.to_csv(RESULTS_DIR/'FULL_AE_SUMMARY_AND_JOINT_IUT.csv',index=False,encoding='utf-8-sig')
    # Detect rather than hide numerical discrepancies with the already supplied
    # November 11 baseline, including changes caused by software versions.
    if 'd1D0_reference' in selected_scenarios:
        primary = []
        for d in selected_datasets:
            old = pd.read_csv(source_file('inference', d), encoding='utf-8-sig')
            old = old[old.engine.eq('arima_sarima')]
            old = old[['event','status','delta_mae','ci95_low','ci95_high']].copy()
            old['dataset'] = d
            primary.append(old)
        old_primary = pd.concat(primary, ignore_index=True)
        old_primary = old_primary.rename(columns={
            'status':'saved_main_status','delta_mae':'saved_main_delta_mae',
            'ci95_low':'saved_main_ci95_low','ci95_high':'saved_main_ci95_high'})
        renewed=s[s.scenario.eq('d1D0_reference')].merge(
            old_primary, on=['dataset','event'],how='left',validate='one_to_one')
        renewed['abs_delta_difference']=abs(renewed.delta_mae-renewed.saved_main_delta_mae)
        renewed['nominal_positive_saved'] = renewed.saved_main_ci95_low>0
        renewed['nominal_positive_changed_vs_saved'] = (
            renewed.nominal_positive != renewed.nominal_positive_saved).astype('boolean')
        # Do not classify an untested/unfinished comparison as 'changed'.
        renewed.loc[renewed.status!='INFERENCE_READY','nominal_positive_changed_vs_saved']=pd.NA
        renewed[['dataset','event','status','saved_main_status',
          'delta_mae','saved_main_delta_mae','abs_delta_difference',
          'ci95_low','ci95_high','saved_main_ci95_low','saved_main_ci95_high',
          'nominal_positive','nominal_positive_saved',
          'nominal_positive_changed_vs_saved']].to_csv(
            RESULTS_DIR/'REFITTED_REFERENCE_VS_SAVED_MAIN.csv',index=False,encoding='utf-8-sig')

    pd.DataFrame(group_records).to_csv(RESULTS_DIR/'GROUP_COMPLETENESS_AND_COUNTS.csv',index=False,encoding='utf-8-sig')
    if years:pd.DataFrame(years).to_csv(RESULTS_DIR/'YEARLY_DELTA_ALL.csv',index=False,encoding='utf-8-sig')
    if model_audits:pd.DataFrame(model_audits).to_csv(RESULTS_DIR/'SELECTED_MODELS_AND_QC.csv',index=False,encoding='utf-8-sig')
    # Keep a separate table comparing every alternative directly to the re-fitted reference.
    if 'd1D0_reference' in selected_scenarios:
        base=s[s.scenario=='d1D0_reference'].set_index(['dataset','event'])
        changes=[]
        for scenario in selected_scenarios:
            if scenario=='d1D0_reference':continue
            a=s[s.scenario==scenario].set_index(['dataset','event'])
            x=a.join(base[['status','delta_mae','ci95_low','ci95_high','nominal_IUT_positive',
                           'q_BH_dataset_IUT','fdr_joint_q_lt_005']],rsuffix='_reference')
            x['delta_mae_difference_from_reference']=x['delta_mae']-x['delta_mae_reference']
            x['nominal_IUT_changed']=(x['nominal_IUT_positive']!=x['nominal_IUT_positive_reference']).astype('boolean')
            x.loc[(x.status!='INFERENCE_READY') | (x.status_reference!='INFERENCE_READY'),
                  'nominal_IUT_changed']=pd.NA
            x=x.reset_index()
            changes.append(x[['dataset','event','scenario','status','status_reference',
                'delta_mae','delta_mae_reference','delta_mae_difference_from_reference',
                'ci95_low','ci95_high','nominal_IUT_positive','nominal_IUT_positive_reference',
                'nominal_IUT_changed','q_BH_dataset_IUT','q_BH_dataset_IUT_reference',
                'fdr_joint_q_lt_005','fdr_joint_q_lt_005_reference','qc_extreme_forecast_years']])
        if changes:
            pd.concat(changes,ignore_index=True).to_csv(RESULTS_DIR/'ALTERNATIVES_VS_REFERENCE.csv',index=False,encoding='utf-8-sig')
    focus={("overall","低ナトリウム血症"),("non_vaccine","低ナトリウム血症"),
       ("non_vaccine","発熱"),("vaccine","注射部位腫脹"),("vaccine","注射部位紅斑"),
       ("overall","ギラン・バレー症候群"),("overall","誤嚥"),
       ("vaccine","ギラン・バレー症候群"),("vaccine","急性散在性脳脊髄炎")}
    s[s.apply(lambda r: (r['dataset'],r['event']) in focus,axis=1)].to_csv(RESULTS_DIR/'FOCUS_9_RESULTS.csv',index=False,encoding='utf-8-sig')
    manifest={'utc_generated':datetime.now(timezone.utc).isoformat(),
      'software':{'python':sys.version,'platform':platform.platform(),
       'statsmodels':statsmodels.__version__,'numpy':np.__version__,
       'scipy':__import__('scipy').__version__,'pandas':pd.__version__},
      'source_sha256':source_hashes,
      'inference_source_sha256':{d:sha256_file(source_file('inference',d)) for d in selected_datasets},
      'input_file_paths':{d:{'monthly':str(source_file('monthly',d)),'inference':str(source_file('inference',d))} for d in selected_datasets},
      'scenarios':{k:SCENARIOS[k] for k in selected_scenarios},
      'note':'BH computed only for terminal groups; estimation failures and zero-variance tests are excluded and counted; inspect group_complete and n_failed.'}
    (RESULTS_DIR/'RUN_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n[COLLECT] Completed; please inspect GROUP_COMPLETENESS_AND_COUNTS.csv',flush=True)
    print(pd.DataFrame(group_records).to_string(index=False),flush=True)


def execute(mode: str, scenarios: list[str], datasets: list[str],workers:int,pilot_n:int) -> None:
    if mode not in ("validate","pilot","all","collect"):
        raise ValueError('mode must be validate/pilot/all/collect')
    if workers<1 or workers>32:
        raise ValueError('workers must be 1..32')
    if any(c not in SCENARIOS for c in scenarios):
        raise ValueError('Unknown scenario')
    if any(c not in INPUT_FILES for c in datasets):
        raise ValueError('Unknown dataset')
    if mode=='collect':
        return collect_data(scenarios,datasets)
    if mode=='validate':
        for dataset in datasets:
            df, sha=read_input(dataset)
            print(f"[VALIDATE] {dataset}: {len(df.columns)} AE, 180 months; source={sha}",flush=True)
            p=load_prophet_summary(dataset)
            print(f'[VALIDATE] Monthly source: {source_file("monthly",dataset)}',flush=True)
            print(f'[VALIDATE] Inference source: {source_file("inference",dataset)}',flush=True)
            if set(df.columns)!=set(p.index):
                raise ValueError(f"Prophet results AE list mismatch: {dataset}")
            # Cross-check the monthly count data against BOTH method summaries.
            both = pd.read_csv(source_file("inference", dataset), encoding="utf-8-sig")
            if set(both["engine"]) != {"prophet", "arima_sarima"}:
                raise ValueError(f"期待する2手法が揃っていません: {dataset}")
            if both.duplicated(["event", "engine"]).any():
                raise ValueError(f"AE×手法の重複があります: {dataset}")
            actual_total = df.sum(axis=0)
            actual_initial = df.iloc[:48].sum(axis=0)
            for engine in ("prophet", "arima_sarima"):
                sub = both[both.engine.eq(engine)].set_index("event")
                if set(sub.index) != set(df.columns):
                    raise ValueError(f"{dataset}/{engine}: AE名が月次データと異なります")
                sub = sub.reindex(df.columns)
                for col, expected in (("n_reports", actual_total),
                                      ("initial_48_month_reports", actual_initial)):
                    actual = pd.to_numeric(sub[col], errors="raise")
                    different = actual.to_numpy(float) != expected.to_numpy(float)
                    if different.any():
                        offending = expected.index[different].tolist()[:5]
                        raise ValueError(f"{dataset}/{engine}: {col} 不一致: {offending}")
                expected_status = np.where(actual_initial.to_numpy(float) == 0,
                                           "INFERENCE_INELIGIBLE_ZERO_HISTORY", "INFERENCE_READY")
                if not np.array_equal(sub["status"].to_numpy(str),expected_status):
                    raise ValueError(f"{dataset}/{engine}: 推論対象外の扱いが一致しません")
            print(f"[VALIDATE] BOTH methods: AE names, 15-year totals, 48-month totals, inference eligibility match ({len(p)} AEs).",flush=True)
        print('[VALIDATE] ALL INPUT FILES PASSED. Next set RUN_MODE="pilot".',flush=True)
        return
    all_tasks=[]
    for dataset in datasets:
        df,sha=read_input(dataset)
        events=list(df.columns)
        if mode=='pilot':
            # Prioritize high-value scientific cases in the small test.
            preferred = {'overall':['低ナトリウム血症','ギラン・バレー症候群','誤嚥'],
                         'non_vaccine':['低ナトリウム血症','発熱','食欲減退'],
                         'vaccine':['ギラン・バレー症候群','急性散在性脳脊髄炎','注射部位腫脹']}.get(dataset,[])
            events=([x for x in preferred if x in events]+[x for x in events if x not in preferred])[:pilot_n]
        for name in scenarios:
            for event in events:
                all_tasks.append((name,dataset,event,df[event].to_numpy(dtype=float).tolist(),sha))
    print(f'[RUN] mode={mode} scenarios={scenarios} datasets={datasets} tasks={len(all_tasks)} workers={workers}',flush=True)
    print('[IMPORTANT] AIC is never compared across different d/D; scenario conclusions use held-out MAE.',flush=True)
    t0=time.perf_counter()
    completed=0
    if workers==1:
        for task in all_tasks:
            r=work_one(task)
            completed+=1
            print(f'[CASE {completed}/{len(all_tasks)}] {r}',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures=[ex.submit(work_one,task) for task in all_tasks]
            for fut in as_completed(futures):
                r=fut.result()
                completed+=1
                print(f'[CASE {completed}/{len(all_tasks)}] {r}',flush=True)
    print(f'[RUN FINISHED] cases={completed}, seconds={time.perf_counter()-t0:.1f}',flush=True)
    collect_data(scenarios,datasets)


def main():
    parser=argparse.ArgumentParser(description='Run full-AE ARIMA/SARIMA sensitivity analysis')
    parser.add_argument('--mode',choices=['validate','pilot','all','collect'],default=RUN_MODE)
    parser.add_argument('--scenarios',nargs='+',choices=list(SCENARIOS),default=SCENARIO_NAMES)
    parser.add_argument('--datasets',nargs='+',choices=list(INPUT_FILES),default=DATASET_NAMES)
    parser.add_argument('--workers',type=int,default=WORKERS)
    parser.add_argument('--pilot-ae',type=int,default=PILOT_AE_PER_DATASET)
    args=parser.parse_args()
    execute(args.mode,args.scenarios,args.datasets,args.workers,args.pilot_ae)


if __name__=='__main__':
    # Spyder's ipykernel may pass extra arguments, hence avoid main() if running
    # as an interactive runfile with notebook-specific argv.
    if 'ipykernel' in sys.modules or 'spyder_kernels' in sys.modules:
        execute(RUN_MODE, SCENARIO_NAMES, DATASET_NAMES,WORKERS,PILOT_AE_PER_DATASET)
    else:
        main()
