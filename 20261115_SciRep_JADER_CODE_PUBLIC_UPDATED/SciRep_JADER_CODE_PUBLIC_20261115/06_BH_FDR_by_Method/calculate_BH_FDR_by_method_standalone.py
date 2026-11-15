# -*- coding: utf-8 -*-
"""Supplementary Benjamini-Hochberg FDR analysis, separately by dataset and method.

Place this file in the desired output folder. Inputs are the three
INFERENCE_BY_METHOD_{overall,non_vaccine,vaccine}.csv files, located either
next to this script, under a sibling input/ directory, or selected with INPUT_DIR.
No forecasting or sensitivity refitting is performed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
INPUT_DIR = None  # optional: Path('/path/to/inference_csv_folder')
OUTPUT_DIR = HERE
ALPHA = 0.05
DATASETS = ('overall', 'non_vaccine', 'vaccine')
ENGINES = ('prophet', 'arima_sarima')


def find_input(dataset):
    filename = f'INFERENCE_BY_METHOD_{dataset}.csv'
    candidates = ([Path(INPUT_DIR) / filename] if INPUT_DIR is not None else
                  [HERE / filename, HERE / 'input' / filename,
                   HERE.parent / filename, HERE.parent / 'input' / filename])
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f'{filename} not found. Set INPUT_DIR to the folder containing the three inference CSVs. Tried: {candidates}')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    summary = []
    for ds in DATASETS:
        source = find_input(ds)
        df = pd.read_csv(source, encoding='utf-8-sig')
        required = {'dataset', 'event', 'engine', 'status', 'ci95_low', 'delta_mae',
                    'p_direction_adjusted_two_sided'}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f'{source}: missing columns {sorted(missing)}')
        if not df['dataset'].eq(ds).all():
            raise ValueError(f'{source}: dataset labels do not match {ds}')
        if df.duplicated(['dataset', 'event', 'engine']).any():
            raise ValueError(f'{source}: duplicate AE/method rows')
        for engine in ENGINES:
            sub = df.loc[df.engine.eq(engine)].copy()
            if sub.empty:
                raise ValueError(f'No {engine} rows in {source}')
            eligible = sub.status.eq('INFERENCE_READY')
            p = pd.to_numeric(sub.loc[eligible, 'p_direction_adjusted_two_sided'], errors='coerce')
            if p.isna().any() or (~p.between(0, 1)).any():
                raise ValueError(f'{ds}/{engine}: missing or invalid eligible p values')
            # The source uses the TWO-SIDED t-test p value if Delta MAE > 0,
            # otherwise p=1. This is NOT the conventional one-sided p/2.
            # Reusing the original p values preserves the published 5/10 result.
            # We use the *existing* directional p values, not a fresh test.
            q = multipletests(p.to_numpy(), alpha=ALPHA, method='fdr_bh')[1]
            sub['bh_test_family'] = f'{ds}__{engine}'
            sub['bh_family_n'] = int(eligible.sum())
            sub['q_BH_by_method'] = np.nan
            sub.loc[eligible, 'q_BH_by_method'] = q
            sub['candidate_ci95_positive'] = eligible & (pd.to_numeric(sub.ci95_low, errors='coerce') > 0)
            sub['bh_significant_positive'] = eligible & (sub.q_BH_by_method < ALPHA) & (pd.to_numeric(sub.delta_mae, errors='coerce') > 0)
            sub['candidate_survives_BH'] = sub.candidate_ci95_positive & sub.bh_significant_positive
            summary.append({'dataset': ds, 'engine': engine, 'n_total_rows': len(sub),
                            'n_tested': int(eligible.sum()),
                            'n_CI95_candidates': int(sub.candidate_ci95_positive.sum()),
                            'n_BH_significant_positive': int(sub.bh_significant_positive.sum()),
                            'n_CI95_candidates_surviving_BH': int(sub.candidate_survives_BH.sum()),
                            'min_q_BH': float(np.min(q)), 'alpha': ALPHA,
                            'source_file': source.name})
            results.append(sub)
    all_rows = pd.concat(results, ignore_index=True)
    all_rows.to_csv(OUTPUT_DIR / 'BH_FDR_BY_METHOD_ALL_AEs.csv', index=False, encoding='utf-8-sig')
    s = pd.DataFrame(summary)
    s.to_csv(OUTPUT_DIR / 'BH_FDR_BY_METHOD_SUMMARY.csv', index=False, encoding='utf-8-sig')
    all_rows.loc[all_rows.candidate_survives_BH].to_csv(
        OUTPUT_DIR / 'BH_FDR_BY_METHOD_SURVIVING_CANDIDATES.csv', index=False, encoding='utf-8-sig')
    print(s[['dataset', 'engine', 'n_tested', 'n_CI95_candidates', 'n_CI95_candidates_surviving_BH']].to_string(index=False))
    expected = {('overall', 'prophet'): (720, 7, 0), ('overall', 'arima_sarima'): (720, 9, 0),
                ('non_vaccine', 'prophet'): (692, 4, 0), ('non_vaccine', 'arima_sarima'): (692, 7, 0),
                ('vaccine', 'prophet'): (47, 10, 5), ('vaccine', 'arima_sarima'): (47, 2, 0)}
    for row in summary:
        key = (row['dataset'], row['engine'])
        observed = (row['n_tested'], row['n_CI95_candidates'], row['n_CI95_candidates_surviving_BH'])
        if observed != expected[key]:
            raise RuntimeError(f'Unexpected counts for {key}: {observed} != {expected[key]}')
    print('PASS: All six method-specific BH summaries match manuscript v12.')
    print('Output folder:', OUTPUT_DIR)

if __name__ == '__main__':
    main()
