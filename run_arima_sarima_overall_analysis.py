# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:08:08 2026
numpy          : 1.26.4
pandas         : 2.2.2
matplotlib     : 3.9.2
statsmodels    : 0.14.2
scikit-learn   : 1.5.1
"""

import datetime
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import matplotlib
matplotlib.rcParams['font.family'] = ['Arial', 'MS Gothic'] 

dt_now = datetime.datetime.now()
today = dt_now.strftime('%Y%m%d')
print(today)

save_dir = ""
os.makedirs(save_dir, exist_ok=True)

data = pd.read_csv(r"")

data['年月'] = pd.to_datetime(data['年月'], format='%Y%m')
data.set_index('年月', inplace=True)

data_15y_100 = data.loc[:, data.sum(axis=0) >= 100]

df = data_15y_100.copy()

print(df.head())

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if not np.any(y_true == 0) else np.nan
    return mae, rmse, mape

def bootstrap_fold_diff_ci(arima_values, sarima_values, n_boot=2000, alpha=0.05, random_state=42):

    arima_values = np.array(arima_values, dtype=float)
    sarima_values = np.array(sarima_values, dtype=float)

    valid_mask = (~np.isnan(arima_values)) & (~np.isnan(sarima_values))
    arima_values = arima_values[valid_mask]
    sarima_values = sarima_values[valid_mask]

    if len(arima_values) == 0:
        return {
            "delta_mean": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_folds_used": 0
        }

    delta_folds = arima_values - sarima_values
    delta_mean = np.mean(delta_folds)

    rng = np.random.default_rng(random_state)
    n = len(delta_folds)
    boot_means = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(delta_folds[idx])

    ci_lower = np.quantile(boot_means, alpha / 2)
    ci_upper = np.quantile(boot_means, 1 - alpha / 2)

    return {
        "delta_mean": delta_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_folds_used": n
    }


def grid_search_arima(series, p_range=(0,2), d_range=(0,2), q_range=(0,2)):

    p = range(p_range[0], p_range[1])
    d = range(d_range[0], d_range[1])
    q = range(q_range[0], q_range[1])
    pdq_combinations = list(itertools.product(p, d, q))

    best_aic = np.inf
    best_order = None
    best_model = None

    for order in pdq_combinations:
        try:
            model = SARIMAX(series, order=order, seasonal_order=(0,0,0,0),
                           enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = order
                best_model = model_fit
        except:
            continue

    return best_model, best_order, best_aic

def grid_search_sarima(series, p_range=(0,2), d_range=(0,2), q_range=(0,2),
                      P_range=(0,2), D_range=(0,2), Q_range=(0,2), s=12):

    p = range(p_range[0], p_range[1])
    d = range(d_range[0], d_range[1])
    q = range(q_range[0], q_range[1])
    P = range(P_range[0], P_range[1])
    D = range(D_range[0], D_range[1])
    Q = range(Q_range[0], Q_range[1])

    pdq_combinations = list(itertools.product(p, d, q))
    seasonal_pdq_combinations = list(itertools.product(P, D, Q))

    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None

    for order in pdq_combinations:
        for seasonal_order in seasonal_pdq_combinations:
            seasonal_order_full = seasonal_order + (s,)
            try:
                model = SARIMAX(series, order=order, seasonal_order=seasonal_order_full,
                               enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit(disp=False)
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
                    best_seasonal_order = seasonal_order_full
                    best_model = model_fit
            except:
                continue

    return best_model, best_order, best_seasonal_order, best_aic


adverse_events = df.columns

print("Number of adverse events:", len(adverse_events))

n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

results = []

for event in adverse_events:
    print(f"Processing: {event}")

    series = df[event].dropna()
    counts_n = int(series.sum())

    if not isinstance(series.index, pd.DatetimeIndex):
        print(f"  --> Skipping {event}: Index is not a DatetimeIndex")
        continue

    arima_metrics = {"MAE": [], "RMSE": [], "MAPE": []}
    sarima_metrics = {"MAE": [], "RMSE": [], "MAPE": []}

    arima_aic_list = []
    sarima_aic_list = []
    arima_bic_list = []
    sarima_bic_list = []
    arima_order_list = []
    sarima_order_list = []
    sarima_seasonal_order_list = []

    for fold, (train_index, test_index) in enumerate(tscv.split(series), 1):
        print(f"  Fold {fold}:")
        train, test = series.iloc[train_index], series.iloc[test_index]

        best_arima_model, best_arima_order, best_arima_aic = grid_search_arima(train)
        if best_arima_model is not None:
            arima_aic_list.append(best_arima_aic)
            arima_bic_list.append(best_arima_model.bic)
            arima_order_list.append(best_arima_order)

            try:
                arima_pred = best_arima_model.get_forecast(steps=len(test))
                arima_forecast = arima_pred.predicted_mean

                mae, rmse, mape = calculate_metrics(test, arima_forecast)
                arima_metrics["MAE"].append(mae)
                arima_metrics["RMSE"].append(rmse)
                arima_metrics["MAPE"].append(mape)
            except:
                arima_metrics["MAE"].append(np.nan)
                arima_metrics["RMSE"].append(np.nan)
                arima_metrics["MAPE"].append(np.nan)
        else:
            arima_aic_list.append(np.nan)
            arima_bic_list.append(np.nan)
            arima_order_list.append((np.nan, np.nan, np.nan))
            arima_metrics["MAE"].append(np.nan)
            arima_metrics["RMSE"].append(np.nan)
            arima_metrics["MAPE"].append(np.nan)

        best_sarima_model, best_sarima_order, best_sarima_seasonal_order, best_sarima_aic = grid_search_sarima(train)
        if best_sarima_model is not None:
            sarima_aic_list.append(best_sarima_aic)
            sarima_bic_list.append(best_sarima_model.bic)
            sarima_order_list.append(best_sarima_order)
            sarima_seasonal_order_list.append(best_sarima_seasonal_order)

            try:
                sarima_pred = best_sarima_model.get_forecast(steps=len(test))
                sarima_forecast = sarima_pred.predicted_mean

                mae, rmse, mape = calculate_metrics(test, sarima_forecast)
                sarima_metrics["MAE"].append(mae)
                sarima_metrics["RMSE"].append(rmse)
                sarima_metrics["MAPE"].append(mape)
            except:
                sarima_metrics["MAE"].append(np.nan)
                sarima_metrics["RMSE"].append(np.nan)
                sarima_metrics["MAPE"].append(np.nan)
        else:
            sarima_aic_list.append(np.nan)
            sarima_bic_list.append(np.nan)
            sarima_order_list.append((np.nan, np.nan, np.nan))
            sarima_seasonal_order_list.append((np.nan, np.nan, np.nan, 12))
            sarima_metrics["MAE"].append(np.nan)
            sarima_metrics["RMSE"].append(np.nan)
            sarima_metrics["MAPE"].append(np.nan)

    mae_ci_result = bootstrap_fold_diff_ci(arima_metrics["MAE"], sarima_metrics["MAE"], n_boot=2000, alpha=0.05, random_state=42)
    rmse_ci_result = bootstrap_fold_diff_ci(arima_metrics["RMSE"], sarima_metrics["RMSE"], n_boot=2000, alpha=0.05, random_state=42)
    mape_ci_result = bootstrap_fold_diff_ci(arima_metrics["MAPE"], sarima_metrics["MAPE"], n_boot=2000, alpha=0.05, random_state=42)

    result = {
        "有害事象名": event,
        "Counts(n)": counts_n,

        "ARIMA_平均_AIC": np.nanmean(arima_aic_list),
        "SARIMA_平均_AIC": np.nanmean(sarima_aic_list),
        "ARIMA_平均_BIC": np.nanmean(arima_bic_list),
        "SARIMA_平均_BIC": np.nanmean(sarima_bic_list),

        "ARIMA_平均_MAE": np.nanmean(arima_metrics["MAE"]),
        "SARIMA_平均_MAE": np.nanmean(sarima_metrics["MAE"]),
        "ΔMAE": mae_ci_result["delta_mean"],
        "ΔMAE_95CI下限": mae_ci_result["ci_lower"],
        "ΔMAE_95CI上限": mae_ci_result["ci_upper"],
        "MAE_CI使用fold数": mae_ci_result["n_folds_used"],

        "ARIMA_平均_RMSE": np.nanmean(arima_metrics["RMSE"]),
        "SARIMA_平均_RMSE": np.nanmean(sarima_metrics["RMSE"]),
        "ΔRMSE": rmse_ci_result["delta_mean"],
        "ΔRMSE_95CI下限": rmse_ci_result["ci_lower"],
        "ΔRMSE_95CI上限": rmse_ci_result["ci_upper"],
        "RMSE_CI使用fold数": rmse_ci_result["n_folds_used"],

        "ARIMA_平均_MAPE": np.nanmean(arima_metrics["MAPE"]),
        "SARIMA_平均_MAPE": np.nanmean(sarima_metrics["MAPE"]),
        "ΔMAPE": mape_ci_result["delta_mean"],
        "ΔMAPE_95CI下限": mape_ci_result["ci_lower"],
        "ΔMAPE_95CI上限": mape_ci_result["ci_upper"],
        "MAPE_CI使用fold数": mape_ci_result["n_folds_used"],

        "ARIMA_最適_p,d,q": arima_order_list,
        "SARIMA_最適_p,d,q,P,D,Q,12": sarima_seasonal_order_list
    }
    results.append(result)

    try:
        # ARIMA
        best_arima_model, best_arima_order, best_arima_aic = grid_search_arima(series)
        arima_forecast = best_arima_model.get_forecast(steps=12).predicted_mean

        # SARIMA
        best_sarima_model, best_sarima_order, best_sarima_seasonal_order, best_sarima_aic = grid_search_sarima(series)
        sarima_forecast = best_sarima_model.get_forecast(steps=12).predicted_mean

        fig, ax = plt.subplots(figsize=(12, 6))

        series.plot(ax=ax, label="Observed", color="black")

        arima_forecast.plot(ax=ax, label="ARIMA Forecast", color="blue")

        sarima_forecast.plot(ax=ax, label="SARIMA Forecast", color="red")

        ax.set_title(f"{event} : ARIMA vs. SARIMA Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        plt.legend()

        plot_filename = f"{event}_ARIMA_SARIMA_forecast.png"
        plt.savefig(os.path.join(save_dir, plot_filename))
        plt.close()

        print(f"  --> Plot saved: {plot_filename}")
    except:
        print(f"  --> Plot not saved due to an error.: {event}")

    print("-"*50)

results_df = pd.DataFrame(results)

def format_ci(lower, upper):
    if pd.isna(lower) or pd.isna(upper):
        return np.nan   # 空欄にしたいなら "" でも可
    return f"({round(lower, 4):.4f}-{round(upper, 4):.4f})"

results_df["ΔMAE_95CI"] = results_df.apply(
    lambda row: format_ci(row["ΔMAE_95CI下限"], row["ΔMAE_95CI上限"]),
    axis=1
)

results_df["ΔRMSE_95CI"] = results_df.apply(
    lambda row: format_ci(row["ΔRMSE_95CI下限"], row["ΔRMSE_95CI上限"]),
    axis=1
)

results_df["ΔMAPE_95CI"] = results_df.apply(
    lambda row: format_ci(row["ΔMAPE_95CI下限"], row["ΔMAPE_95CI上限"]),
    axis=1
)

columns_order = [
    "有害事象名",
    "Counts(n)",

    "ARIMA_平均_AIC",
    "SARIMA_平均_AIC",
    "ARIMA_平均_BIC",
    "SARIMA_平均_BIC",

    "ARIMA_平均_MAE",
    "SARIMA_平均_MAE",
    "ΔMAE",
    "ΔMAE_95CI下限",
    "ΔMAE_95CI上限",
    "ΔMAE_95CI",
    "MAE_CI使用fold数",

    "ARIMA_平均_RMSE",
    "SARIMA_平均_RMSE",
    "ΔRMSE",
    "ΔRMSE_95CI下限",
    "ΔRMSE_95CI上限",
    "ΔRMSE_95CI",
    "RMSE_CI使用fold数",

    "ARIMA_平均_MAPE",
    "SARIMA_平均_MAPE",
    "ΔMAPE",
    "ΔMAPE_95CI下限",
    "ΔMAPE_95CI上限",
    "ΔMAPE_95CI",
    "MAPE_CI使用fold数",

    "ARIMA_最適_p,d,q",
    "SARIMA_最適_p,d,q,P,D,Q,12"
]

results_df = results_df[columns_order]


metrics_csv_path = os.path.join(save_dir, "ARIMA_SARIMA_overall.csv")
results_df.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')

print(f"All metrics saved to {metrics_csv_path}")
