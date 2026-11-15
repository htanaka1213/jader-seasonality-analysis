# -*- coding: utf-8 -*-
"""CONFIGURATION for the 2026-11-11 revision of the submitted JADER study.

The first run intentionally processes ONE event per dataset and engine.
After checking the outputs, set MAX_EVENTS_PER_DATASET = None and rerun.
Yearly checkpoints already finished under the same configuration are reused.
"""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "monthly"
INPUT_FILES = {
    "overall": DATA_ROOT / "20250112_yearly_counts_filtered.csv",
    "non_vaccine": DATA_ROOT / "20260411_yearly_counts_filtered_non_vaccine.csv",
    "vaccine": DATA_ROOT / "20260411_yearly_counts_filtered_vaccine_only.csv",
}
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "main"
# A small smoke test by default; change to None for ALL eligible adverse events.
MAX_EVENTS_PER_DATASET = None
# Optionally set exact Japanese AE names; None means every AE meeting >=100 total.
SELECTED_EVENTS = None
# Rerun previously failed/unresolved origin-years; normally leave False.
RETRY_UNRESOLVED_ORIGINS = False

# Locked implementation of the 11-origin comparison from the controlled AIC pilot.
# Compare ARIMA(p,1,q) (p,q in {0,1}) vs SARIMA(p,1,q)(P,0,Q)[12]
# (p,q,P,Q in {0,1}, P+Q>0), no trend, stationary/invertible constraints.
ARIMA_SPECIFICATION = "controlled_d1_D0_v2"

# Inference rules; two-sided 95% CI + positive effect, then IUT and BH within dataset.
MIN_REPORTS_15Y = 100
FDR_Q_THRESHOLD = 0.05

# Design decision: all-zero initial 48 months -> no primary inferential classification
# for either method. Zero forecasts are shown DESCRIPTIVELY only.
ZERO_HISTORY_POLICY = "both_zero_prediction_descriptive_no_pvalue"

# User-reviewed output namespace. Change OUTPUT_ROOT for a substantively new analysis.
REVISION_ID = "JADER_SciRep_REVISED_20261111_v1"
