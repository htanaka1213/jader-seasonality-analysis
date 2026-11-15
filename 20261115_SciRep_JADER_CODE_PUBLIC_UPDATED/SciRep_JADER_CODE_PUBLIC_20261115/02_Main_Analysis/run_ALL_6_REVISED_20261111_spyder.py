# -*- coding: utf-8 -*-
"""Spyder convenience entry point for all 3 datasets x 2 model families.

Default is ONE AE PER DATASET for a smoke test. Set MAX_EVENTS_PER_DATASET=None
in scirep_revision_config_20261111.py before a full-scale run.
"""
from pathlib import Path
import runpy
HERE=Path(__file__).resolve().parent
if __name__ == "__main__":
    for method in ("prophet", "arima_sarima"):
        for dataset in ("overall", "non_vaccine", "vaccine"):
            print("\n>>> RUN",method,dataset,flush=True)
            runpy.run_path(str(HERE / f"run_{method}_{dataset}_analysis_REVISED_20261111.py"),run_name="__main__")
    print("\n>>> FORECAST EXTRACTION COMPLETE. Run summarize_IUT_BH_REVISED_20261111.py separately.")
