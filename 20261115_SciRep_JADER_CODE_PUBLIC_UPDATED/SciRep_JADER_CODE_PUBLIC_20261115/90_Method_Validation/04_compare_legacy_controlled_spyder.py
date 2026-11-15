# -*- coding: utf-8 -*-
"""Compare original-spec (legacy) and fixed-differencing sensitivity 11x12 models.
No model refitting. Keeps one row per (dataset, event)."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("_scirep_legacy_core_compare_v2",HERE/"arima_11x12_core.py")
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
BASE=(HERE.parent / "outputs" / "historical_pilot")
ROOTS={"legacy":BASE/"ARIMA_SARIMA_11x12_PILOT",
       "controlled":BASE/"ARIMA_SARIMA_11x12_CONTROLLED_PILOT"}
COLS=["dataset","event","n_origins","mae_arima_raw","mae_sarima_raw",
      "delta_mae_arima_minus_sarima_raw","group_t95_low_exploratory",
      "group_t95_high_exploratory","group_t_p_two_sided_exploratory",
      "year_positive_count","year_negative_count"]

def main():
    summaries={}
    for label,root in ROOTS.items():
        p=root/"PILOT_SUMMARY_11x12.csv"
        if not p.is_file():
            print('SKIP missing:',p)
            continue
        df=pd.read_csv(p,encoding='utf-8-sig')
        df=df[df.status.eq('OK')].copy()
        summaries[label]=df[COLS].rename(columns={c:f'{label}_{c}' for c in COLS if c not in ('dataset','event')})
    if len(summaries)!=2:
        print('Both legacy and controlled summary CSVs required to compare.')
        return
    a,b=summaries['legacy'],summaries['controlled']
    out=a.merge(b,on=['dataset','event'],how='outer',indicator=True,validate='one_to_one')
    both=out['_merge'].eq('both')
    out['both_positive_delta']=np.where(both,(out['legacy_delta_mae_arima_minus_sarima_raw']>0)&(out['controlled_delta_mae_arima_minus_sarima_raw']>0),False)
    out['legacy_t_CI_lower_positive']=out['legacy_group_t95_low_exploratory']>0
    out['controlled_t_CI_lower_positive']=out['controlled_group_t95_low_exploratory']>0
    out['classification_changed_t_exploratory']=out['legacy_t_CI_lower_positive']!=out['controlled_t_CI_lower_positive']
    out['delta_mae_change_controlled_minus_legacy']=out['controlled_delta_mae_arima_minus_sarima_raw']-out['legacy_delta_mae_arima_minus_sarima_raw']
    dest=BASE/"ARIMA_11x12_AIC_AUDIT"/"LEGACY_VS_CONTROLLED_11x12.csv"
    core.write_csv_atomic(out,dest)
    print('Saved:',dest)
    print('Fully paired cases:',int(both.sum()),' / ',len(out))
    print('IMPORTANT: t-interval classifications are exploratory, not FDR-ready.')

if __name__=='__main__':main()
