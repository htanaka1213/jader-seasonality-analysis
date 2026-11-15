# -*- coding: utf-8 -*-
"""Descriptive peak calendar month and peak-to-mean ratio for JADER AEs.

For each AE and dataset, sum 2005-2019 counts over the 15 occurrences
of each calendar month.  Peak = month(s) with the largest summed count.
Ratio = peak-month total / (15-year total / 12) = 12 * peak / total.

The metric is descriptive only; it is NOT a test of seasonality or AE risk.

Usage (any one):
    # 1. Spyder: run this file; reads the co-located revision config, (uses the revised main-analysis config if necessary).
    # 2. Portable/repository: put the three monthly input CSVs in ./data/.
    # 3. Explicit directory from a terminal:
    python compute_peak_month_REVISED_20261113.py --data-dir ./data
    # 4. Change output location:
    python compute_peak_month_REVISED_20261113.py --data-dir ./data --output-dir ./outputs

Inputs are the same three monthly count CSVs as the revised main analysis.
Output: output_peak_month/PEAK_MONTH_ALL_AEs_YYYYMMDD.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT_BASENAMES = {
    "overall": "20250112_yearly_counts_filtered.csv",
    "non_vaccine": "20260411_yearly_counts_filtered_non_vaccine.csv",
    "vaccine": "20260411_yearly_counts_filtered_vaccine_only.csv",
}
CALENDAR_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
EXPECTED_DATES = pd.date_range("2005-01-01", "2019-12-01", freq="MS")
MIN_REPORTS_15Y = 100


def choose_inputs(data_dir: Path | None) -> tuple[dict[str, Path], str]:
    """Choose an explicit local data directory, a repo data/ directory,
    the main-analysis configuration, or the documented original paths.
    """
    if data_dir is not None:
        return ({key: data_dir / filename for key, filename in INPUT_BASENAMES.items()},
                f"explicit data directory: {data_dir}")

    local_data = BASE.parent / "data" / "monthly"
    if all((local_data / filename).is_file() for filename in INPUT_BASENAMES.values()):
        return ({key: local_data / filename for key, filename in INPUT_BASENAMES.items()},
                f"co-located data directory: {local_data}")

    config_file = BASE.parent / "02_Main_Analysis" / "scirep_revision_config_20261111.py"
    if config_file.is_file():
        spec = importlib.util.spec_from_file_location("peak_month_main_config", config_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load main-analysis configuration: {config_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        paths = {name: Path(module.INPUT_FILES[name]) for name in INPUT_BASENAMES}
        if int(module.MIN_REPORTS_15Y) != MIN_REPORTS_15Y:
            raise ValueError("MIN_REPORTS_15Y differs between scripts")
        return paths, f"main-analysis configuration: {config_file.name}"

    raise FileNotFoundError("Monthly inputs missing. Place CSVs in ../data/monthly or supply --data-dir.")


def read_monthly_counts(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV is missing: {path}. Use --data-dir or update main config.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "年月" not in df:
        raise ValueError(f"Missing '年月' column: {path}")
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate AE column names: {path}")

    dates = pd.to_datetime(df.pop("年月").astype(str), format="%Y%m", errors="raise")
    if not pd.DatetimeIndex(dates).equals(EXPECTED_DATES):
        raise ValueError(f"Expected exactly 180 monthly rows (2005-01 to 2019-12): {path}")

    counts = df.apply(pd.to_numeric, errors="raise")
    a = counts.to_numpy(dtype=float)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f"Non-finite, missing or negative monthly counts: {path}")
    if not np.allclose(a, np.rint(a), atol=1e-9, rtol=0.0):
        raise ValueError(f"Monthly AE counts must be whole numbers: {path}")

    counts.index = EXPECTED_DATES
    return counts.loc[:, counts.sum(axis=0) >= MIN_REPORTS_15Y].copy()


def calculate_peak_month(path_by_dataset: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ds, path in path_by_dataset.items():
        counts = read_monthly_counts(path)
        by_calendar_month = counts.groupby(counts.index.month).sum()  # 12 rows
        for event in counts.columns:
            monthly = by_calendar_month[event].to_numpy(dtype=np.int64)
            total = int(monthly.sum())
            peak = int(monthly.max())
            tied = np.flatnonzero(monthly == peak)
            rows.append({
                "dataset": ds,
                "event": event,
                "n_reports_2005_2019": total,
                "peak_month": "/".join(CALENDAR_MONTHS[i] for i in tied),
                "n_peak_month_tied": len(tied),
                "peak_month_reports": peak,
                "peak_to_mean_ratio": peak / (total / 12),
                **{f"reports_{CALENDAR_MONTHS[i]}": int(monthly[i]) for i in range(12)},
            })
        print(f"[INPUT] {ds}: {counts.shape[1]} eligible AEs; {path}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Directory containing the 3 input monthly CSVs")
    parser.add_argument("--output-dir", type=Path, help="Directory for output CSV")
    args = parser.parse_args()

    paths, source = choose_inputs(args.data_dir)
    print("[INPUT CONFIG]", source)
    results = calculate_peak_month(paths)
    out_dir = args.output_dir or BASE.parent / "outputs" / "peak_month"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"PEAK_MONTH_ALL_AEs_{dt.datetime.now():%Y%m%d}.csv"
    results.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"[RESULTS] {len(results)} AE-dataset pairs, {int((results.n_peak_month_tied > 1).sum())} with tied peaks")
    print("[SAVED]", output.resolve())


if __name__ == "__main__":
    main()
