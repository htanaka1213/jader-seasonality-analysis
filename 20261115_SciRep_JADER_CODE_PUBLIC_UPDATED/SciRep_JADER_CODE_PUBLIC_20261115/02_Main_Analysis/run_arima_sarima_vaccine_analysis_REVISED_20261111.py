# -*- coding: utf-8 -*-
"""Revision of GitHub_SourceCode/run_arima_sarima_vaccine_analysis.py.

Run in Spyder (F5). Configuration lives in scirep_revision_config_20261111.py.
"""
from pathlib import Path
import runpy
HERE = Path(__file__).resolve().parent
cfg = runpy.run_path(str(HERE / "scirep_revision_config_20261111.py"))
core = runpy.run_path(str(HERE / "scirep_revision_cv_core_20261111.py"))
if __name__ == "__main__":
    core["run_dataset_engine"](cfg, "vaccine", "arima_sarima")
