# -*- coding: utf-8 -*-
"""Compute 11-cutoff CI, IUT and dataset-wise BH after forecast extraction.

Start this script using Spyder F5. It will NOT compute q values for any dataset
until ALL its eligible AEs have both fully audited method outputs. Pilot runs
produce QC_PROGRESS.csv only; no interim FDR results are mislabelled as final.

Important: cutoff-year t inference assumes approximate independence of 11
origin-wise mean losses; sharing expanding training data may violate this.
HAC intervals are sensitivity diagnostics, NOT a guaranteed valid calibration.
"""
from __future__ import annotations
import importlib.util
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent

def local_module(file_name: str, module_name: str):
    spec=importlib.util.spec_from_file_location(module_name,HERE/file_name)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[module_name]=mod
    spec.loader.exec_module(mod)
    return mod


def bh_qvalues(pvals):
    p = np.asarray(pvals,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("BH requires complete valid p values")
    m=len(p)
    if m==0:
        return np.array([],float)
    idx=np.argsort(p,kind="stable")
    ranked=p[idx]*m/np.arange(1,m+1)
    qrank=np.clip(np.minimum.accumulate(ranked[::-1])[::-1],0,1)
    q=np.empty(m,float)
    q[idx]=qrank
    return q


def t_ci_11(delta):
    x = np.asarray(delta,float)
    if len(x)!=11 or not np.isfinite(x).all():
        return {"valid":False,"reason":"NEED_11_FINITE_ORIGIN_DIFFERENCES"}
    mean=float(x.mean())
    se=float(np.std(x,ddof=1)/np.sqrt(len(x)))
    # A zero-variance positive difference is not used as proof of significance.
    if se <= 1e-12 or not np.isfinite(se):
        return {"valid":False,"reason":"DEGENERATE_ORIGIN_VARIANCE", "delta_mae":mean}
    crit=float(stats.t.ppf(.975,df=10))
    p_two=float(2*stats.t.sf(abs(mean/se), df=10))
    return {"valid":True,"delta_mae":mean,"se_year":se,"ci95_low":mean-crit*se,
            "ci95_high":mean+crit*se,"p_two_sided":p_two,
            "p_direction_adjusted_two_sided": p_two if mean>0 else 1.0,
            "n_positive_origins":int(np.sum(x>0)),
            "n_negative_origins":int(np.sum(x<0)),
            "min_loo_delta":float(min(np.delete(x,i).mean() for i in range(11))),
            "max_loo_delta":float(max(np.delete(x,i).mean() for i in range(11))),
            "reason":"VALID_APPROXIMATE_T"}


def hac_ci(values, lag: int=12):
    """Bartlett HAC, normal approximation, sensitivity only."""
    x=np.asarray(values,float)
    if len(x)!=132 or not np.isfinite(x).all():
        return None, None
    e=x-x.mean()
    n=len(x)
    var=float(np.dot(e,e)/n)
    for j in range(1,min(lag,n-1)+1):
        var+=2*(1-j/(lag+1))*float(np.dot(e[:-j],e[j:])/n)
    se=float(np.sqrt(max(var,0)/n))
    crit=float(stats.norm.ppf(.975))
    return float(x.mean()-crit*se),float(x.mean()+crit*se)


def analyze_method(core, output_root:Path, dataset:str, engine:str, event:str, expected_y:np.ndarray, source_sha:str):
    folder=output_root/dataset/engine/core.safe_slug(event)
    summary_file=folder/"revised_event_summary.csv"
    csv_file=folder/"revised_cv_monthly_paired.csv"
    if not (summary_file.is_file() and csv_file.is_file()):
        return {"dataset":dataset,"event":event,"engine":engine,"status":"NOT_RUN"}
    try:
        manifest_path = output_root / dataset / engine / "RUN_MANIFEST.json"
        if not manifest_path.is_file():
            raise ValueError("RUN_MANIFEST_MISSING")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_sha256") != source_sha:
            raise ValueError("INPUT_CSV_SHA256_MISMATCH")
        if manifest.get("core_sha256") != core.sha256_path(HERE / "scirep_revision_cv_core_20261111.py"):
            raise ValueError("MODEL_CORE_SHA256_MISMATCH")
        if manifest.get("schema") != core.SCHEMA or manifest.get("engine") != engine or manifest.get("dataset") != dataset:
            raise ValueError("MODEL_SPECIFICATION_MISMATCH")
        s=pd.read_csv(summary_file,encoding="utf-8-sig").iloc[0].to_dict()
        if s.get("config_fingerprint") != manifest.get("fingerprint"):
            raise ValueError("EVENT_SUMMARY_FINGERPRINT_MISMATCH")
        for year in core.YEARS:
            yy = folder / "years" / f"{year}_metrics.json"
            cc = folder / "years" / f"{year}_predictions.csv"
            if not yy.is_file() or not cc.is_file():
                raise ValueError(f"CHECKPOINT_MISSING_{year}")
            yydata = json.loads(yy.read_text(encoding="utf-8"))
            if yydata.get("config_fingerprint") != manifest.get("fingerprint"):
                raise ValueError(f"CHECKPOINT_FINGERPRINT_MISMATCH_{year}")
        
        data=pd.read_csv(csv_file,encoding="utf-8-sig")
        if len(data)!=132 or pd.DatetimeIndex(pd.to_datetime(data["ds"])).tolist()!=list(pd.date_range("2009-01-01","2019-12-01",freq="MS")):
            raise ValueError("MONTHLY_DATE_MISMATCH")
        if not (data["dataset"].eq(dataset).all() and data["event"].eq(event).all() and data["engine"].eq(engine).all()):
            raise ValueError("DATASET_EVENT_ENGINE_MISMATCH")
        if not np.allclose(data["y"].to_numpy(float), expected_y,rtol=0,atol=1e-8):
            raise ValueError("ACTUALS_DIFFER_FROM_SOURCE_CSV")
        if not np.array_equal(data["origin_year"].to_numpy(int), np.repeat(np.arange(2009,2020),12)):
            raise ValueError("ORIGIN_ASSIGNMENT_MISMATCH")
        status=str(s["status"])
        result={"dataset":dataset,"event":event,"engine":engine,"status":status,
                "n_reports":int(s["n_reports_2005_2019"]),
                "initial_48_month_reports":int(s["initial_48_month_reports"]),
                "n_origin_fitted":int(s["n_origin_fitted"]),
                "n_origin_zero_history":int(s["n_origin_zero_history"]),
                "n_origin_unresolved":int(s["n_origin_unresolved"]),
                "n_valid_months":int(s["n_valid_months"]),
                "mae_baseline_descriptive":s["mae_baseline_descriptive"],
                "mae_seasonal_descriptive":s["mae_seasonal_descriptive"],
                "delta_mae_descriptive":s["delta_mae_descriptive"]}
        # A fully reproducible 11 x 12 forecast grid may contain descriptive zeros,
        # but these are not used for p values in the primary joint analysis.
        if status!="INFERENCE_READY":
            result["inference_reason"]=status
            return result
        y=data["y"].to_numpy(float)
        baseline=data["yhat_baseline"].to_numpy(float)
        seasonal=data["yhat_seasonal"].to_numpy(float)
        d=data["d_abs_error"].to_numpy(float)
        if not np.isfinite(np.c_[y,baseline,seasonal,d]).all():
            raise ValueError("MISSING_MONTHLY_FORECAST_FOR_READY_CASE")
        if not np.allclose(np.abs(y-baseline)-np.abs(y-seasonal),d,atol=1e-8,rtol=0):
            raise ValueError("LOSS_DIFFERENCE_INCONSISTENT")
        group=data.groupby("origin_year",sort=True)
        yearly=group.agg(n_months=("ds","size"), d=("d_abs_error","mean"))
        if yearly.index.tolist()!=list(core.YEARS) or not yearly.n_months.eq(12).all():
            raise ValueError("BAD_ORIGIN_GROUP_COUNTS")
        diag=t_ci_11(yearly["d"].to_numpy(float))
        if not diag["valid"]:
            result["status"]="INFERENCE_INELIGIBLE_"+diag["reason"]
            result["inference_reason"]=diag["reason"]
            return result
        hlow,hhigh=hac_ci(d,12)
        result.update({k:v for k,v in diag.items() if k not in {"valid","reason"}})
        result.update({"inference_reason":"VALID_APPROXIMATE_T",
                       "hac12_ci95_low_sensitivity":hlow,
                       "hac12_ci95_high_sensitivity":hhigh})
        return result
    except Exception as exc:
        return {"dataset":dataset,"event":event,"engine":engine,"status":"AUDIT_ERROR",
                "inference_reason":f"{type(exc).__name__}: {exc}"}


def combine_pair(a,b):
    row={"dataset":a["dataset"],"event":a["event"],
         "status_prophet":a["status"],"status_arima_sarima":b["status"],
         "n_reports_2005_2019":a.get("n_reports",b.get("n_reports")),
         "initial_48_month_reports":a.get("initial_48_month_reports",b.get("initial_48_month_reports"))}
    for tag, d in (("prophet",a),("arima_sarima",b)):
        for col in ("delta_mae_descriptive","mae_baseline_descriptive","mae_seasonal_descriptive",
                    "n_origin_fitted","n_origin_zero_history","n_origin_unresolved",
                    "n_valid_months","ci95_low","ci95_high","p_two_sided",
                    "p_direction_adjusted_two_sided","hac12_ci95_low_sensitivity",
                    "hac12_ci95_high_sensitivity", "inference_reason"):
            row[f"{tag}_{col}"]=d.get(col,np.nan)
    if (a["status"]==b["status"]=="INFERENCE_READY" and
        all(pd.notna(d.get("p_direction_adjusted_two_sided")) for d in (a,b))):
        # IUT of joint improvement in both procedures.
        p_joint=max(float(a["p_direction_adjusted_two_sided"]),
                    float(b["p_direction_adjusted_two_sided"]))
        row["p_IUT_joint"]=p_joint
        row["nominal_both_lower95_positive"] = bool(a["ci95_low"]>0 and b["ci95_low"]>0)
        row["joint_status"]="VALID_IUT"
    else:
        row["joint_status"]="NOT_TESTABLE"
        row["p_IUT_joint"]=np.nan
        row["nominal_both_lower95_positive"]=pd.NA
    return row


def main():
    configmod=local_module("scirep_revision_config_20261111.py","_scirep_rev_config_for_inference")
    core=local_module("scirep_revision_cv_core_20261111.py","_scirep_rev_core_for_inference")
    config={x:getattr(configmod,x) for x in dir(configmod) if x.isupper()}
    root=Path(config["OUTPUT_ROOT"])
    root.mkdir(parents=True,exist_ok=True)
    audit=[]
    completed=[]
    for dataset,inputfile in config["INPUT_FILES"].items():
        source=core.load_source(Path(inputfile))
        source_sha=core.sha256_path(Path(inputfile))
        indiv=[]
        joints=[]
        for event in source.columns:
            series=source[event]
            actual=series.iloc[48:].to_numpy(float)
            a=analyze_method(core,root,dataset,"prophet",event,actual,source_sha)
            b=analyze_method(core,root,dataset,"arima_sarima",event,actual,source_sha)
            indiv.extend([a,b]); joints.append(combine_pair(a,b))
        ia=pd.DataFrame(indiv)
        joint=pd.DataFrame(joints)
        core.atomic_csv(ia,root/f"INFERENCE_BY_METHOD_{dataset}.csv")
        n_tot=len(source.columns)
        n_notrun=int((joint["status_prophet"].eq("NOT_RUN") | joint["status_arima_sarima"].eq("NOT_RUN")).sum())
        n_audit=int((joint["status_prophet"].eq("AUDIT_ERROR") | joint["status_arima_sarima"].eq("AUDIT_ERROR")).sum())
        n_joint=int(joint.joint_status.eq("VALID_IUT").sum())
        n_ineligible=n_tot-n_joint
        progress={"dataset":dataset,"n_expected_ae":n_tot,"n_joint_testable":n_joint,
                  "n_joint_not_testable":n_ineligible,"n_not_run_either":n_notrun,
                  "n_audit_error_either":n_audit,
                  "initial_all_zero_AEs":int((source.iloc[:48,:].sum(axis=0)==0).sum()),
                  "bh_calculated":False}
        # Never mislabel BH on a pilot subset as the final family.
        if n_notrun==0 and n_audit==0:
            valid=joint.joint_status.eq("VALID_IUT")
            joint["q_BH_dataset_IUT"]=np.nan
            joint["fdr_joint_q_lt_005"]=pd.NA
            joint.loc[valid,"q_BH_dataset_IUT"]=bh_qvalues(joint.loc[valid,"p_IUT_joint"].to_numpy(float))
            joint.loc[valid,"fdr_joint_q_lt_005"]=(joint.loc[valid,"q_BH_dataset_IUT"]<config["FDR_Q_THRESHOLD"]).to_numpy()
            progress["bh_calculated"]=True
            progress["n_fdr_joint_positive"]=int(joint["fdr_joint_q_lt_005"].fillna(False).astype(bool).sum())
            progress["n_nominal_joint_positive"]=int(joint["nominal_both_lower95_positive"].fillna(False).astype(bool).sum())
            core.atomic_csv(joint,root/f"IUT_BH_RESULTS_{dataset}.csv")
            completed.append(joint)
            print(f"BH COMPLETE {dataset}: {n_joint}/{n_tot} testable; FDR findings: {progress['n_fdr_joint_positive']}")
        else:
            # Diagnostic table NOT labelled IUT_BH_RESULTS (no misleading q-values).
            core.atomic_csv(joint,root/f"QC_JOINT_PARTIAL_{dataset}.csv")
            print(f"BH NOT RUN for {dataset}: {n_notrun} not run, {n_audit} audit errors")
        audit.append(progress)
    core.atomic_csv(pd.DataFrame(audit),root/"QC_PROGRESS_ALL_DATASETS.csv")
    if len(completed)==3:
        core.atomic_csv(pd.concat(completed,ignore_index=True),root/"IUT_BH_RESULTS_ALL_DATASETS.csv")
        print("ALL THREE DATASETS COMPLETE; final joint results saved.")
    else:
        print("No final combined table until all 3 datasets finish.")
    return pd.DataFrame(audit)

if __name__ == "__main__":
    main()
