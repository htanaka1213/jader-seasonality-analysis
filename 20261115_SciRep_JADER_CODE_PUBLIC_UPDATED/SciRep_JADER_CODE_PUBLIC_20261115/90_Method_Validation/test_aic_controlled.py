# -*- coding: utf-8 -*-
"""Small reproducibility test: structural burn, genuine seasonal grid, and same-forecast window.
Requires statsmodels/scipy, but no user JADER file.
"""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("_test_controlled_core",HERE/"arima_11x12_controlled_core.py")
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

def main():
    assert len(core.ARIMA_ORDERS)==4
    assert len(core.SEASONAL_ORDERS)==3
    assert len(core.SARIMA_ORDERS)*len(core.SEASONAL_ORDERS)==12
    assert all(s[1]==0 and (s[0] or s[2]) for s in core.SEASONAL_ORDERS)
    idx=pd.date_range("2005-01-01",periods=180,freq="MS")
    rng=np.random.default_rng(987)
    y=pd.Series(rng.poisson(4,size=180).astype(float),index=idx)
    train,test=core.split_aligned_year(y,2009)
    assert len(train)==48 and len(test)==12
    a=core.original_aic_search(train,'ARIMA')
    s=core.original_aic_search(train,'SARIMA')
    for result in (a,s):
        assert result['fit'] is not None
        assert result['selected_loglikelihood_burn']==1
        assert result['selected_nobs_effective']==47
        assert result['candidate_burn_min']==result['candidate_burn_max']==1
    rows,detail=core.run_one_year(y,"overall","test_toy",2009,"fp_test")
    assert detail['complete']
    assert len(rows)==12 and rows['ds'].is_unique
    assert np.allclose(rows['d_abs_error'],rows['ae_arima']-rows['ae_sarima'])
    assert detail['arima_candidate_burn_min']==detail['sarima_candidate_burn_max']==1
    print('PASS: 4 vs 12 candidate models, common 47 effective months, 12 paired predictions')

if __name__=='__main__':main()
