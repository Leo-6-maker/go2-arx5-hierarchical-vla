import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


def pytest_addoption(parser):
    parser.addoption("--run-mujoco", action="store_true", default=False,
                     help="Run engine checks only in an independently authorized MuJoCo environment")


def pytest_configure(config):
    config.addinivalue_line("markers", "mujoco: requires explicit --run-mujoco outside GitHub CI")
    if config.getoption("--run-mujoco") and os.environ.get("GITHUB_ACTIONS") == "true":
        raise pytest.UsageError("MuJoCo execution is not part of GitHub CPU contract CI")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-mujoco"):
        marker = pytest.mark.skip(reason="engine check requires separate authorization and --run-mujoco")
        for item in items:
            if "mujoco" in item.keywords:
                item.add_marker(marker)
