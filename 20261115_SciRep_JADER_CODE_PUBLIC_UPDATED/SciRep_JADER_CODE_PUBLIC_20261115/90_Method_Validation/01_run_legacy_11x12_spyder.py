# -*- coding: utf-8 -*-
"""Spyder / F5: representative ARIMA/SARIMA 11 origins x 12-month CV.

Open this file, press F5. No command-line args. This is the pilot only.
Each year has a durable checkpoint; restarting skips completed years.
"""
from __future__ import annotations
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Strong local-module import: avoids stale Spyder modules named arima_11x12_core.
import importlib.util
_module_name = "_scirep_arima_aligned_core_local"
_spec = importlib.util.spec_from_file_location(_module_name, HERE/"arima_11x12_core.py")
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)

INPUT_FILES = {
    "overall": (HERE.parent / "data" / "monthly" / "20250112_yearly_counts_filtered.csv"),
    "non_vaccine": (HERE.parent / "data" / "monthly" / "20260411_yearly_counts_filtered_non_vaccine.csv"),
    "vaccine": (HERE.parent / "data" / "monthly" / "20260411_yearly_counts_filtered_vaccine_only.csv"),
}
PROPHET_ROOT = (HERE.parent / "outputs" / "historical_pilot")
OUTPUT = PROPHET_ROOT / "ARIMA_SARIMA_11x12_PILOT"
CASES_FILE = HERE / "representative_AEs_9.csv"

# Recommended first invocation: 1 AE to check runtime. Then switch to 8.
MAX_CASES = 9  # 1 => smoke test; 9 / None => all 9 selected cases
ALLOW_MISSING_PROPHET = True  # if no Prophet export exists, ARIMA will still run
VERIFY_EXISTING_PROPHET = True
FORCE_REDO = False  # Do NOT overwrite valid checkpoints unless necessary.


def env_info():
    import statsmodels
    import scipy
    return {"python":platform.python_version(),"numpy":np.__version__,
            "pandas":pd.__version__,"scipy":scipy.__version__,
            "statsmodels":statsmodels.__version__}


def fingerprint_for_source(input_file:Path, input_sha: str) -> str:
    payload = {"schema":core.SCHEMA_VERSION,
               "input_sha256":input_sha,
               "core_sha256":core.file_hash(HERE/"arima_11x12_core.py"),
               "test_years": list(core.YEARS),
               "arima_orders": [list(x) for x in core.ARIMA_ORDERS],
               "sarima_orders": [list(x) for x in core.SARIMA_ORDERS],
               "seasonal_orders": [list(x) for x in core.SEASONAL_ORDERS],
               "environment":env_info()}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode("utf-8")).hexdigest()


def process_case(dataset, event, source, input_path, input_sha):
    series = core.validate_event(source,event)
    folder = OUTPUT/dataset/core.slugify(event)
    folder.mkdir(parents=True, exist_ok=True)
    years_dir = folder/"years"
    years_dir.mkdir(exist_ok=True)
    fp = fingerprint_for_source(input_path,input_sha)
    manifest_file = folder/"manifest.json"
    manifest = {"schema":core.SCHEMA_VERSION, "dataset":dataset,"event":event,
                "input_file":str(input_path),"input_sha256":input_sha,
                "config_fingerprint":fp,"environment":env_info(),
                "source_original_code":"run_arima_sarima_*_analysis.py",
                "change":"5 x 30-month TimeSeriesSplit -> 11 x 12-month calendar-year origins; AIC grids identical",
                "inference":"all reported intervals are sensitivity diagnostics; validity not established"}
    if manifest_file.exists():
        old=json.loads(manifest_file.read_text(encoding="utf-8"))
        if old.get("config_fingerprint")!=fp:
            raise ValueError("既存フォルダーと入力CSV・コード・環境が異なります。別の出力フォルダーを指定してください。")
    else:
        core.write_json_atomic(manifest,manifest_file)
    pieces, metrics, failures = [],[],[]
    for year in core.YEARS:
        path = years_dir/f"{year}_predictions.csv"
        metapath = years_dir/f"{year}_metrics.json"
        complete = False
        if path.exists() and metapath.exists() and not FORCE_REDO:
            old=json.loads(metapath.read_text(encoding="utf-8"))
            if old.get("config_fingerprint")!=fp:
                raise ValueError(f"年別キャッシュに設定の不一致: {metapath}")
            if old.get("complete"):
                rows=pd.read_csv(path,encoding="utf-8-sig")
                if (len(rows)==12 and rows.ds.nunique()==12 and
                    rows.d_abs_error.notna().all() and
                    rows.origin_year.eq(year).all() and
                    list(pd.to_datetime(rows.ds))==list(pd.date_range(f"{year}-01-01",periods=12,freq="MS"))):
                    print(f"  {year}: skip completed",flush=True)
                    complete = True
                    detail=old
                else:
                    print(f"  {year}: cached file incomplete; rerun",flush=True)
        if not complete:
            train,test=core.split_aligned_year(series,year)
            print(f"  {year}: train {len(train)} months, forecast 12 months (ARIMA 8 + SARIMA 64 AIC candidates)",flush=True)
            rows,detail=core.run_one_year(series,dataset,event,year,fp)
            core.write_csv_atomic(rows,path)
            core.write_json_atomic(detail,metapath)
            print("   status:",detail["status_arima"],"/",detail["status_sarima"],
                  "elapsed(s):",round(detail["arima_fit_seconds"]+detail["sarima_fit_seconds"],1),flush=True)
        pieces.append(rows)
        metrics.append(detail)
        if not detail.get("complete"):
            failures.append(year)
    paired=pd.concat(pieces, ignore_index=True)
    core.write_csv_atomic(paired,folder/"arima_11x12_cv_paired.csv")
    metric_table=pd.DataFrame(metrics)
    core.write_csv_atomic(metric_table,folder/"arima_11x12_origin_metrics.csv")
    if failures:
        return {"dataset":dataset,"event":event,"status":"INCOMPLETE",
                "n_origin_ok":11-len(failures), "failed_origins":",".join(map(str,failures))}
    row, yearly = core.summarize_one(paired,metrics,dataset,event,input_sha,fp)
    row["n_reports_2005_2019"]=int(series.sum())
    row["status"]="OK"
    core.write_csv_atomic(yearly,folder/"arima_11x12_yearly_delta.csv")
    core.write_csv_atomic(pd.DataFrame([row]),folder/"arima_11x12_summary.csv")
    if VERIFY_EXISTING_PROPHET:
        pp = PROPHET_ROOT/dataset/"prophet"/core.slugify(event)/"prophet_cv_paired.csv"
        if pp.exists():
            common = core.verify_prophet_alignment(pp, paired)
            core.write_csv_atomic(common,folder/"prophet_arima_monthly_alignment.csv")
            print("   Prophet: 132 forecast months + observed monthly counts MATCH",flush=True)
        elif not ALLOW_MISSING_PROPHET:
            raise FileNotFoundError(f"Prophetの既存出力なし: {pp}")
        else:
            print("   Prophet data not found: no pairing (ARIMA result still saved)",flush=True)
    return row


def main():
    print("=== ARIMA/SARIMA aligned 11 x 12-month sensitivity analysis ===",flush=True)
    print("Core:",HERE/"arima_11x12_core.py")
    print("Output:",OUTPUT)
    print("First, validate one event; then set MAX_CASES=9 or None.",flush=True)
    cases=pd.read_csv(CASES_FILE,encoding="utf-8-sig")
    required={"dataset","event","selection_role"}
    if not required.issubset(cases.columns):
        raise ValueError(f"case CSV missing: {required-set(cases.columns)}")
    if cases.duplicated(["dataset","event"]).any():
        raise ValueError("Duplicate dataset-event in list")
    if MAX_CASES is not None:
        cases=cases.iloc[:MAX_CASES]
    results=[]
    source_cache={}
    for i,r in enumerate(cases.itertuples(index=False),1):
        ds,event=r.dataset,r.event
        print(f"\n[{i}/{len(cases)}] {ds} / {event}",flush=True)
        try:
            if ds not in INPUT_FILES:
                raise ValueError(f"Unknown dataset: {ds}")
            if ds not in source_cache:
                source_cache[ds]=(core.load_source(INPUT_FILES[ds]),core.file_hash(INPUT_FILES[ds]))
            source,sha=source_cache[ds]
            output=process_case(ds,event,source,INPUT_FILES[ds],sha)
            output["selection_role"]=r.selection_role
            results.append(output)
            print("  Result:",output["status"],flush=True)
        except Exception as exc:
            print("  FAILED:",str(exc),flush=True)
            traceback.print_exc()
            results.append({"dataset":ds,"event":event,"selection_role":r.selection_role,
                            "status":"FAILED","error":f"{type(exc).__name__}: {exc}"})
        OUTPUT.mkdir(parents=True,exist_ok=True)
        core.write_csv_atomic(pd.DataFrame(results),OUTPUT/"RUN_STATUS.csv")
    all_summaries=[]
    for path in OUTPUT.glob("*/*/arima_11x12_summary.csv"):
        row=pd.read_csv(path,encoding="utf-8-sig")
        all_summaries.append(row)
    if all_summaries:
        core.write_csv_atomic(pd.concat(all_summaries,ignore_index=True),
                              OUTPUT/"PILOT_SUMMARY_11x12.csv")
    print(f"\nFinished: {sum(x['status']=='OK' for x in results)}/{len(results)} complete; ",
          OUTPUT/"RUN_STATUS.csv",flush=True)
    print("These are exploratory intervals; do not use for BH/FDR or final significance without inference review.")

if __name__=="__main__":
    main()
