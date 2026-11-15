# -*- coding: utf-8 -*-
"""Regression test for CmdStan version fingerprint instability.
Uses stub Prophet/CmdStanPy modules; does not need an installed Prophet.
"""
import importlib
import runpy
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    fake_state = {"registered": False, "backend_initializations": 0}
    prophet = types.ModuleType("prophet")
    prophet.__version__ = "1.3.0"

    class FakeProphet:
        def __init__(self, yearly_seasonality=False):
            fake_state["registered"] = True
            fake_state["backend_initializations"] += 1

    prophet.Prophet = FakeProphet
    cmd = types.ModuleType("cmdstanpy")
    cmd.__version__ = "1.2.0"
    cmd.cmdstan_version = lambda: (2, 37) if fake_state["registered"] else None

    # Restore any original modules even when the test fails.
    prior = {k: sys.modules.get(k) for k in ("prophet", "cmdstanpy")}
    sys.modules["prophet"] = prophet
    sys.modules["cmdstanpy"] = cmd
    try:
        core = runpy.run_path(str(HERE / "scirep_revision_cv_core_20261111.py"))
        first = core["env_versions"]()
        second = core["env_versions"]()
        assert first["cmdstan"] == "(2, 37)", first
        assert second == first, (first, second)
        assert fake_state["backend_initializations"] == 2
        print("PASS: CmdStan fingerprint stable before and after backend initialization")
    finally:
        for key, old in prior.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old


if __name__ == "__main__":
    main()
