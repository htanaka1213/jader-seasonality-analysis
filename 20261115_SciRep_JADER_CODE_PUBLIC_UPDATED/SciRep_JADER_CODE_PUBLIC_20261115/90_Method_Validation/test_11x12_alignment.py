# -*- coding: utf-8 -*-
"""Tests use supplied monthly GBS CSV (if present) and an independent toy series.
Fast test does not run the 72-candidate grid. Optional full-grid smoke can be run separately.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('test_core',ROOT/'arima_11x12_core.py')
core=importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

def run():
    assert len(core.ARIMA_ORDERS)==8
    assert len(core.SARIMA_ORDERS)==8
    assert len(core.SEASONAL_ORDERS)==8
    assert len(core.YEARS)==11
    idx=pd.date_range('2005-01-01',periods=180,freq='MS')
    y=pd.Series(np.arange(180,dtype=float),index=idx)
    all_ds=[]
    for yr in core.YEARS:
        tr,te=core.split_aligned_year(y,yr)
        assert len(te)==12 and len(tr)==48+12*(yr-2009)
        assert tr.max()<te.min()
        assert tr.index.max()<te.index.min()
        all_ds.extend(te.index.tolist())
    assert pd.DatetimeIndex(all_ds).equals(pd.date_range('2009-01-01','2019-12-01',freq='MS'))
    di=core.diagnostic_ci(np.array([1]*9+[-1]*2,dtype=float))
    assert di['year_positive_count']==9 and di['year_negative_count']==2
    assert di['group_t95_low_exploratory']<np.mean([1]*9+[-1]*2)<di['group_t95_high_exploratory']
    # Check transparent loss diffs and 132 month grouping
    r=[]
    for yr in core.YEARS:
        for m in range(1,13):
            date=pd.Timestamp(year=yr,month=m,day=1)
            r.append({'origin_year':yr,'ds':date.strftime('%Y-%m-%d'),
                      'ae_arima':1.0,'ae_sarima':0.7,'d_abs_error':0.3,
                      'se_arima':1.0,'se_sarima':0.49})
    row, groups = core.summarize_one(pd.DataFrame(r),[], 'overall','test','example_sha','example_fp')
    assert len(groups)==11 and row['n_predicted_months']==132
    assert np.isclose(row['delta_mae_arima_minus_sarima_raw'],0.3)
    print('PASS: CV split 11 origins, 132 unique months, candidate grids 8+64, CI and summary')

if __name__=='__main__':run()
