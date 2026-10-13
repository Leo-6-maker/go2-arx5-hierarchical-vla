"""The default contract suite never needs to construct a simulator."""
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_github_rejects_mujoco_opt_in_before_collection():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--run-mujoco", "--collect-only", "-q"],
        cwd=ROOT, env={**os.environ, "GITHUB_ACTIONS": "true"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 4
    assert "MuJoCo execution is not part of GitHub CPU contract CI" in result.stderr


def test_contract_collection_does_not_import_engine():
    result = subprocess.run(
        [sys.executable, "-c", "import runpy, sys; runpy.run_path('tests/task_data/test_contracts.py'); "
         "assert 'mujoco' not in sys.modules; assert 'go2arx5_vla.task_data.engine' not in sys.modules"],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
