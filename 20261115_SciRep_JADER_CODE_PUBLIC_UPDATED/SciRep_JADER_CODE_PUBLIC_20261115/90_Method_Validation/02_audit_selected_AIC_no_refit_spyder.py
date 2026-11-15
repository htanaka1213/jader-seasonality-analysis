# -*- coding: utf-8 -*-
"""Spyder: completed 11x12 results only; AIC likelihood burn and seasonal-order audit.
This does not refit any models. SARIMAX model constructors are used to inspect
initialization/loglikelihood burn of the *selected* orders on the same input.
"""
from __future__ import annotations
import ast
import importlib.util
import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("_scirep_legacy_core_for_aic_audit",HERE/"arima_11x12_core.py")
core=importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
INPUT_FILES={
 "overall":(HERE.parent / "data" / "monthly" / "20250112_yearly_counts_filtered.csv"),
 "non_vaccine":(HERE.parent / "data" / "monthly" / "20260411_yearly_counts_filtered_non_vaccine.csv"),
 "vaccine":(HERE.parent / "data" / "monthly" / "20260411_yearly_counts_filtered_vaccine_only.csv"),
}
BASE=(HERE.parent / "outputs" / "historical_pilot")
ROOTS={
 "legacy":BASE/"ARIMA_SARIMA_11x12_PILOT",
 "controlled_d1D0":BASE/"ARIMA_SARIMA_11x12_CONTROLLED_PILOT",
}

def read_order(raw):
    value=ast.literal_eval(raw) if isinstance(raw,str) else raw
    if not isinstance(value,tuple) or not all(isinstance(i,int) for i in value):
        raise ValueError(f"Unexpected order: {raw}")
    return value


def main():
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    output=[]
    sources={}
    for mode,root in ROOTS.items():
        if not root.exists():
            print(f"SKIP (folder missing): {root}")
            continue
        for meta_path in sorted(root.glob("*/*/years/*_metrics.json")):
            try:
                record=json.loads(meta_path.read_text(encoding="utf-8"))
                ds,event,year=record["dataset"],record["event"],int(record["origin_year"])
                if not record.get("complete",False):
                    continue
                if ds not in sources:
                    sources[ds]=core.load_source(INPUT_FILES[ds])
                train,_=core.split_aligned_year(core.validate_event(sources[ds],event),year)
                flags={"enforce_stationarity":mode!="legacy",
                       "enforce_invertibility":mode!="legacy",
                       "trend":"n","simple_differencing":False}
                orders={
                   "arima":(read_order(record["arima_order"]),read_order(record["arima_seasonal_order"])),
                   "sarima":(read_order(record["sarima_order"]),read_order(record["sarima_seasonal_order"])),
                }
                row={"mode":mode,"dataset":ds,"event":event,"year":year,
                     "training_months":len(train),"SARIMA_has_nonzero_seasonal_AR_or_MA":bool(orders['sarima'][1][0] or orders['sarima'][1][2]),
                     "SARIMA_has_seasonal_difference":bool(orders['sarima'][1][1])}
                for engine,(order,seasonal_order) in orders.items():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m=SARIMAX(train,order=order,seasonal_order=seasonal_order,**flags)
                    burn=int(m.loglikelihood_burn)
                    row.update({f"{engine}_order":str(order),
                                f"{engine}_seasonal_order":str(seasonal_order),
                                f"{engine}_aic_cached":record.get(f"{engine}_aic"),
                                f"{engine}_burn":burn,
                                f"{engine}_nobs_effective":len(train)-burn,
                                f"{engine}_nonconverged_candidates":record.get(f"{engine}_candidates_nonconverged")})
                row["effective_nobs_difference_SARIMA_minus_ARIMA"]=(row['sarima_nobs_effective']-row['arima_nobs_effective'])
                row["AIC_same_training_likelihood_sample"]=(row['arima_burn']==row['sarima_burn'])
                output.append(row)
            except Exception as exc:
                print(f"AUDIT ERROR: {meta_path}: {type(exc).__name__}: {exc}")
                output.append({"mode":mode,"file":str(meta_path),"error":str(exc)})
    if not output:
        print("Completed model results not found. Run 01 first.")
        return
    detail=pd.DataFrame(output)
    dst=BASE/"ARIMA_11x12_AIC_AUDIT"
    core.write_csv_atomic(detail,dst/"SELECTED_MODEL_AIC_BURN_BY_ORIGIN.csv")
    valid=detail[detail["year"].notna()].copy()
    if len(valid):
        summary=valid.groupby(["mode","dataset","event"],dropna=False).agg(
            completed_origins=("year","size"),
            selected_seasonal_ar_or_ma_origins=("SARIMA_has_nonzero_seasonal_AR_or_MA","sum"),
            selected_D1_origins=("SARIMA_has_seasonal_difference","sum"),
            origins_with_equal_nobs=("AIC_same_training_likelihood_sample","sum"),
            min_effective_nobs_difference=("effective_nobs_difference_SARIMA_minus_ARIMA","min"),
            max_effective_nobs_difference=("effective_nobs_difference_SARIMA_minus_ARIMA","max"),
        ).reset_index()
        core.write_csv_atomic(summary,dst/"SELECTED_MODEL_AIC_BURN_SUMMARY.csv")
        print(summary.to_string(index=False),flush=True)
    print("Saved:",dst,flush=True)

if __name__=="__main__":main()
