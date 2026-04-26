# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:19:54 2026
pandas         : 2.2.2
numpy          : 1.26.4
matplotlib     : 3.9.2
scipy          : 1.13.1
statsmodels    : 0.14.2
scikit-learn   : 1.5.1
prophet        : 1.3.0
Python         : 3.12.3
"""

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
import numpy as np
import datetime
import logging
import warnings
import os
import matplotlib.pyplot as plt
import scipy
import statsmodels
import sklearn
import prophet

print("\n=== Library versions ===")
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("matplotlib:", plt.matplotlib.__version__)
print("scipy:", scipy.__version__)
print("statsmodels:", statsmodels.__version__)
print("scikit-learn:", sklearn.__version__)
print("prophet:", prophet.__version__)

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

def bootstrap_metrics_diff_ci_from_performance(
    performance_seasonality: pd.DataFrame,
    performance_no_seasonality: pd.DataFrame,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42
):

    merge_keys = [c for c in ['horizon'] if c in performance_seasonality.columns and c in performance_no_seasonality.columns]

    if not merge_keys:
        perf_s = performance_seasonality.reset_index(drop=True).copy()
        perf_n = performance_no_seasonality.reset_index(drop=True).copy()
        min_len = min(len(perf_s), len(perf_n))
        perf_s = perf_s.iloc[:min_len].copy()
        perf_n = perf_n.iloc[:min_len].copy()

        merged = pd.DataFrame({
            'mae_seasonality': perf_s['mae'].to_numpy(dtype=float),
            'mae_no_seasonality': perf_n['mae'].to_numpy(dtype=float),
            'rmse_seasonality': perf_s['rmse'].to_numpy(dtype=float),
            'rmse_no_seasonality': perf_n['rmse'].to_numpy(dtype=float),
        })

        if 'mape' in perf_s.columns and 'mape' in perf_n.columns:
            merged['mape_seasonality'] = perf_s['mape'].to_numpy(dtype=float)
            merged['mape_no_seasonality'] = perf_n['mape'].to_numpy(dtype=float)

    else:
        merged = performance_seasonality.merge(
            performance_no_seasonality,
            on=merge_keys,
            suffixes=('_seasonality', '_no_seasonality')
        )

    if merged.empty:
        return {
            'mae_seasonality_pm': None,
            'mae_no_seasonality_pm': None,
            'delta_mae_pm': None,
            'delta_mae_ci_lower': None,
            'delta_mae_ci_upper': None,

            'rmse_seasonality_pm': None,
            'rmse_no_seasonality_pm': None,
            'delta_rmse_pm': None,
            'delta_rmse_ci_lower': None,
            'delta_rmse_ci_upper': None,

            'mape_seasonality_pm': None,
            'mape_no_seasonality_pm': None,
            'delta_mape_pm': None,
            'delta_mape_ci_lower': None,
            'delta_mape_ci_upper': None,

            'n_rows_boot': 0
        }

    rng = np.random.default_rng(random_state)
    n = len(merged)

    mae_seasonality_pm = merged['mae_seasonality'].mean()
    mae_no_seasonality_pm = merged['mae_no_seasonality'].mean()
    delta_mae_pm = mae_no_seasonality_pm - mae_seasonality_pm

    rmse_seasonality_pm = merged['rmse_seasonality'].mean()
    rmse_no_seasonality_pm = merged['rmse_no_seasonality'].mean()
    delta_rmse_pm = rmse_no_seasonality_pm - rmse_seasonality_pm

    has_mape = ('mape_seasonality' in merged.columns) and ('mape_no_seasonality' in merged.columns)

    if has_mape:
        mape_valid = merged[['mape_seasonality', 'mape_no_seasonality']].dropna().copy()
        if len(mape_valid) > 0:
            mape_seasonality_pm = mape_valid['mape_seasonality'].mean()
            mape_no_seasonality_pm = mape_valid['mape_no_seasonality'].mean()
            delta_mape_pm = mape_no_seasonality_pm - mape_seasonality_pm
        else:
            mape_seasonality_pm = None
            mape_no_seasonality_pm = None
            delta_mape_pm = None
    else:
        mape_seasonality_pm = None
        mape_no_seasonality_pm = None
        delta_mape_pm = None

    boot_mae_diff = np.empty(n_boot)
    boot_rmse_diff = np.empty(n_boot)

    if has_mape and len(mape_valid) > 0:
        boot_mape_diff = np.empty(n_boot)
    else:
        boot_mape_diff = None

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = merged.iloc[idx]

        boot_mae_diff[i] = sample['mae_no_seasonality'].mean() - sample['mae_seasonality'].mean()
        boot_rmse_diff[i] = sample['rmse_no_seasonality'].mean() - sample['rmse_seasonality'].mean()

        if boot_mape_diff is not None:
            idx_m = rng.integers(0, len(mape_valid), size=len(mape_valid))
            sample_m = mape_valid.iloc[idx_m]
            boot_mape_diff[i] = sample_m['mape_no_seasonality'].mean() - sample_m['mape_seasonality'].mean()

    mae_ci_lower = np.quantile(boot_mae_diff, alpha / 2)
    mae_ci_upper = np.quantile(boot_mae_diff, 1 - alpha / 2)

    rmse_ci_lower = np.quantile(boot_rmse_diff, alpha / 2)
    rmse_ci_upper = np.quantile(boot_rmse_diff, 1 - alpha / 2)

    if boot_mape_diff is not None:
        mape_ci_lower = np.quantile(boot_mape_diff, alpha / 2)
        mape_ci_upper = np.quantile(boot_mape_diff, 1 - alpha / 2)
    else:
        mape_ci_lower = None
        mape_ci_upper = None

    return {
        'mae_seasonality_pm': mae_seasonality_pm,
        'mae_no_seasonality_pm': mae_no_seasonality_pm,
        'delta_mae_pm': delta_mae_pm,
        'delta_mae_ci_lower': mae_ci_lower,
        'delta_mae_ci_upper': mae_ci_upper,

        'rmse_seasonality_pm': rmse_seasonality_pm,
        'rmse_no_seasonality_pm': rmse_no_seasonality_pm,
        'delta_rmse_pm': delta_rmse_pm,
        'delta_rmse_ci_lower': rmse_ci_lower,
        'delta_rmse_ci_upper': rmse_ci_upper,

        'mape_seasonality_pm': mape_seasonality_pm,
        'mape_no_seasonality_pm': mape_no_seasonality_pm,
        'delta_mape_pm': delta_mape_pm,
        'delta_mape_ci_lower': mape_ci_lower,
        'delta_mape_ci_upper': mape_ci_upper,

        'n_rows_boot': n
    }

dt_now = datetime.datetime.now()
today = dt_now.strftime('%Y%m%d')
print(today)

save_dir = r""
os.makedirs(save_dir, exist_ok=True)

data = pd.read_csv(r"")

data['年月'] = pd.to_datetime(data['年月'], format='%Y%m')
data.set_index('年月', inplace=True)

data_15y_100 = data.loc[:, data.sum(axis=0) >= 100]

df = data_15y_100.copy()

print(df.head())

adverse_events = df.columns
print("number of adverse_events", len(adverse_events))

results = []

for event in adverse_events:
    print(f"Processing: {event}")

    try:
        df_prophet = df.reset_index()[['年月', event]].rename(columns={'年月': 'ds', event: 'y'})
        df_prophet.dropna(subset=['y'], inplace=True)
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        counts_n = int(df_prophet['y'].sum())

        model_seasonality = Prophet(yearly_seasonality=True)
        model_seasonality.fit(df_prophet)

        model_no_seasonality = Prophet(yearly_seasonality=False)
        model_no_seasonality.fit(df_prophet)

        try:
            cv_seasonality = cross_validation(model_seasonality, initial='1095 days', period='365 days', horizon='365 days')
            cv_no_seasonality = cross_validation(model_no_seasonality, initial='1095 days', period='365 days', horizon='365 days')

            performance_seasonality = performance_metrics(cv_seasonality)
            performance_no_seasonality = performance_metrics(cv_no_seasonality)

            mae_seasonality = performance_seasonality['mae'].mean()
            rmse_seasonality = performance_seasonality['rmse'].mean()
            mae_no_seasonality = performance_no_seasonality['mae'].mean()
            rmse_no_seasonality = performance_no_seasonality['rmse'].mean()

            try:
                mape_seasonality = performance_seasonality['mape'].mean()
            except Exception:
                mape_seasonality = None

            try:
                mape_no_seasonality = performance_no_seasonality['mape'].mean()
            except Exception:
                mape_no_seasonality = None

            bootstrap_result = bootstrap_metrics_diff_ci_from_performance(
                performance_seasonality=performance_seasonality,
                performance_no_seasonality=performance_no_seasonality,
                n_boot=2000,
                alpha=0.05,
                random_state=42
            )

            mae_seasonality_boot = bootstrap_result['mae_seasonality_pm']
            mae_no_seasonality_boot = bootstrap_result['mae_no_seasonality_pm']
            delta_mae_boot = bootstrap_result['delta_mae_pm']
            delta_mae_ci_lower = bootstrap_result['delta_mae_ci_lower']
            delta_mae_ci_upper = bootstrap_result['delta_mae_ci_upper']

            rmse_seasonality_boot = bootstrap_result['rmse_seasonality_pm']
            rmse_no_seasonality_boot = bootstrap_result['rmse_no_seasonality_pm']
            delta_rmse_boot = bootstrap_result['delta_rmse_pm']
            delta_rmse_ci_lower = bootstrap_result['delta_rmse_ci_lower']
            delta_rmse_ci_upper = bootstrap_result['delta_rmse_ci_upper']

            mape_seasonality_boot = bootstrap_result['mape_seasonality_pm']
            mape_no_seasonality_boot = bootstrap_result['mape_no_seasonality_pm']
            delta_mape_boot = bootstrap_result['delta_mape_pm']
            delta_mape_ci_lower = bootstrap_result['delta_mape_ci_lower']
            delta_mape_ci_upper = bootstrap_result['delta_mape_ci_upper']

            n_pairs_boot = bootstrap_result['n_rows_boot']

        except Exception as e:
            print(f"CV/performance failed for {event}: {e}")

            mae_seasonality = None
            rmse_seasonality = None
            mae_no_seasonality = None
            rmse_no_seasonality = None

            mape_seasonality = None
            mape_no_seasonality = None

            mae_seasonality_boot = None
            mae_no_seasonality_boot = None
            delta_mae_boot = None
            delta_mae_ci_lower = None
            delta_mae_ci_upper = None

            rmse_seasonality_boot = None
            rmse_no_seasonality_boot = None
            delta_rmse_boot = None
            delta_rmse_ci_lower = None
            delta_rmse_ci_upper = None

            mape_seasonality_boot = None
            mape_no_seasonality_boot = None
            delta_mape_boot = None
            delta_mape_ci_lower = None
            delta_mape_ci_upper = None

            n_pairs_boot = None

        results.append([
            event, counts_n,

            mae_seasonality, rmse_seasonality, mape_seasonality,
            mae_no_seasonality, rmse_no_seasonality, mape_no_seasonality,

            mae_seasonality_boot, mae_no_seasonality_boot,
            delta_mae_boot, delta_mae_ci_lower, delta_mae_ci_upper,

            rmse_seasonality_boot, rmse_no_seasonality_boot,
            delta_rmse_boot, delta_rmse_ci_lower, delta_rmse_ci_upper,

            mape_seasonality_boot, mape_no_seasonality_boot,
            delta_mape_boot, delta_mape_ci_lower, delta_mape_ci_upper,

            n_pairs_boot
        ])

        try:
            forecast_seasonality = model_seasonality.predict(
                model_seasonality.make_future_dataframe(periods=12, freq='MS')
            )
            fig_components = model_seasonality.plot_components(forecast_seasonality)
            fig_components.savefig(f"{save_dir}/{event}_components.png")
            plt.close(fig_components)

            plt.figure(figsize=(8, 6))
            yhat_plot = forecast_seasonality['yhat'].clip(lower=0)
            yhat_lower_plot = forecast_seasonality['yhat_lower'].clip(lower=0)
            yhat_upper_plot = forecast_seasonality['yhat_upper'].clip(lower=0)

            plt.fill_between(
                forecast_seasonality['ds'],
                yhat_lower_plot,
                yhat_upper_plot,
                color='lightgray',
                alpha=0.3,
                label='95% prediction interval'
            )
            plt.plot(forecast_seasonality['ds'], yhat_plot, color='blue', label='Forecast')
            plt.scatter(df_prophet['ds'], df_prophet['y'], color='black', s=5, label='Observed reports')

            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Reports (n)', fontsize=12)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)

            ymax = max(float(df_prophet['y'].max()), float(yhat_upper_plot.max()))
            plt.ylim(0, ymax * 1.05)

            plt.tight_layout()
            plt.savefig(f"{save_dir}/{event}_forecast_interval.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Plot saving failed for {event}: {e}")

    except Exception as e:
        print(f"Error processing {event}: {e}")
        results.append([event] + [None] * 23)

results_df = pd.DataFrame(results, columns=[
    '有害事象', 'Counts(n)',

    'MAE_季節性あり', 'RMSE_季節性あり', 'MAPE_季節性あり',
    'MAE_季節性なし', 'RMSE_季節性なし', 'MAPE_季節性なし',

    'MAE_季節性あり_CV誤差平均', 'MAE_季節性なし_CV誤差平均',
    'ΔMAE_bootstrap', 'ΔMAE_95CI下限', 'ΔMAE_95CI上限',

    'RMSE_季節性あり_CV誤差平均', 'RMSE_季節性なし_CV誤差平均',
    'ΔRMSE_bootstrap', 'ΔRMSE_95CI下限', 'ΔRMSE_95CI上限',

    'MAPE_季節性あり_CV誤差平均', 'MAPE_季節性なし_CV誤差平均',
    'ΔMAPE_bootstrap', 'ΔMAPE_95CI下限', 'ΔMAPE_95CI上限',

    '比較予測点数_bootstrap'
])

save_path = os.path.join(save_dir, today + "prophet_non_vaccine.csv")
results_df.to_csv(save_path, index=False, encoding='utf-8-sig')
