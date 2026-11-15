# -*- coding: utf-8 -*-
"""Fast regression tests without Prophet fit or full SARIMA grid.
Run with: python test_revision_logic_20261111.py
"""
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
core=runpy.run_path(str(HERE/"scirep_revision_cv_core_20261111.py"))
inf=runpy.run_path(str(HERE/"summarize_IUT_BH_REVISED_20261111.py"))


def test_math():
    bh=inf["bh_qvalues"]([.01,.04,.02,.2])
    assert np.allclose(bh, [.04,.05333333333333334,.04,.2]),bh
    d=inf["t_ci_11"](np.array([.01,.02,.03,.04,.05,.06,.07,.08,.09,.1,.11]))
    assert d["valid"] and d["ci95_low"]>0 and d["p_direction_adjusted_two_sided"]<.05
    x=inf["t_ci_11"](-np.array([.01,.02,.03,.04,.05,.06,.07,.08,.09,.1,.11]))
    assert x["valid"] and x["p_direction_adjusted_two_sided"]==1.0
    assert not inf["t_ci_11"](np.zeros(11))["valid"]
    a={"dataset":"overall","event":"A","status":"INFERENCE_READY", "p_direction_adjusted_two_sided":.001,"ci95_low":.5}
    b={"dataset":"overall","event":"A","status":"INFERENCE_READY", "p_direction_adjusted_two_sided":.01,"ci95_low":.1}
    z=inf["combine_pair"](a,b)
    assert z["p_IUT_joint"]==.01 and z["nominal_both_lower95_positive"]
    b["p_direction_adjusted_two_sided"]=1.0
    assert inf["combine_pair"](a,b)["p_IUT_joint"]==1.0


def make_series(initial_zero=False):
    t=pd.date_range("2005-01-01",periods=180,freq="MS")
    data=np.arange(180,dtype=float)%6
    if initial_zero:
        data[:60]=0
    return pd.Series(data,index=t,name="試験用AE")


def mock_fit(train,test):
    # Deliberately nontrivial loss differences and no nested model fitting.
    a=np.full(12,float(train.iloc[-1])+1)
    s=np.full(12,float(train.iloc[-1])+.5)
    return {"baseline":a,"seasonal":s},{"baseline":{},"seasonal":{}}


def test_checkpoints():
    original=core["fit_prophet_pair"]
    # function globals as returned from runpy are mutable shared by functions
    fn=core["forecast_origin"]
    fn.__globals__["fit_prophet_pair"]=mock_fit
    try:
        with tempfile.TemporaryDirectory() as temp:
            s=make_series(False)
            out=Path(temp)/"event"
            r=core["run_event"](s,"overall","prophet",out,"fingerprint-test",False)
            assert r["status"]=="INFERENCE_READY" and r["n_origin_fitted"]==11
            paired=pd.read_csv(out/"revised_cv_monthly_paired.csv")
            assert len(paired)==132 and paired["ds"].nunique()==132
            r2=core["run_event"](s,"overall","prophet",out,"fingerprint-test",False)
            assert r2["status"]=="INFERENCE_READY"
            s2=make_series(True)
            r3=core["run_event"](s2,"overall","prophet",Path(temp)/"zero_event","new-test",False)
            assert r3["status"]=="INFERENCE_INELIGIBLE_ZERO_HISTORY"
            assert r3["n_origin_zero_history"]>=1
            z=pd.read_csv(Path(temp)/"zero_event"/"revised_cv_monthly_paired.csv")
            assert (z[z.origin_year==2009].d_abs_error==0).all()
    finally:
        fn.__globals__["fit_prophet_pair"]=original


if __name__=="__main__":
    test_math()
    test_checkpoints()
    print("PASS: BH, direction-aware IUT, year-wise t, zero-history policy, durable checkpoints")
