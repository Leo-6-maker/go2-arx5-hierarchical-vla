"""CPU/static regression tests for the PI0117 Vanda H200 launcher."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45d_d1_vanda_h200_launcher as launcher  # noqa: E402


def test_launcher_has_no_top_level_model_imports() -> None:
    guard = launcher.static_guard()
    assert guard["status"] == "PASS"
    assert guard["top_level_forbidden_imports"] == []
    tree = ast.parse(Path(launcher.__file__).read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and getattr(node, "module", "") in {"torch", "transformers", "gr00t", "cuda"}
        for node in tree.body
    )


def test_absolute_launcher_from_foreign_cwd_empty_pythonpath(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "stage45d_d1_vanda_h200_launcher.py"), "--plan-only"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS"
    assert result["smoke_token"] == launcher.SMOKE_PASS_TOKEN
    assert result["full_token"] == launcher.FULL_PASS_TOKEN


def test_offline_child_environment_activates_pinned_nested_hf_resolution(tmp_path: Path) -> None:
    cosmos_root = tmp_path / "cosmos-cache"
    cosmos_root.mkdir()
    environment = launcher.child_environment(
        {"cosmos_root": str(cosmos_root)},
        tmp_path / "evidence",
        launcher.AUTHORITY_TOKEN,
    )
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["PYTHONPATH"] == ""
    assert environment["GROOT_HF_LOCAL_FIRST"] == "1"
    assert environment["GROOT_PATCH_MISTRAL"] == "1"
    model_link = (
        tmp_path
        / "evidence"
        / "runtime_namespace"
        / "huggingface"
        / "hub"
        / "models--nvidia--Cosmos-Reason2-2B"
    )
    assert model_link.is_symlink()
    assert model_link.resolve() == cosmos_root.resolve()


def test_child_json_parser_accepts_runtime_log_prefix_and_requires_trailing_object() -> None:
    result = launcher.parse_child_json(
        "[groot] local cache hit\\nprogress 100%\\n" + json.dumps({"status": "PASS", "calls": 3})
    )
    assert result == {"status": "PASS", "calls": 3}
    with pytest.raises(launcher.LauncherError, match="trailing JSON"):
        launcher.parse_child_json(json.dumps({"status": "PASS"}) + "unexpected log")


def test_wrong_head_fails_closed_before_gpu_observation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = False

    def forbidden_gpu() -> dict:
        nonlocal called
        called = True
        raise AssertionError("GPU observation must not run after wrong HEAD")

    monkeypatch.setattr(launcher, "gpu_observation", forbidden_gpu)
    with pytest.raises(launcher.LauncherError, match="HEAD"):
        launcher.execute(
            tmp_path / "paths.json",
            tmp_path / "evidence",
            "0" * 40,
            launcher.AUTHORITY_TOKEN,
            0,
            1,
        )
    assert called is False


def test_runtime_paths_reject_missing_or_extra_keys(tmp_path: Path) -> None:
    malformed = {key: f"/tmp/{key}" for key in launcher.PATH_KEYS}
    malformed.pop("full24_runtime_statistics")
    path = tmp_path / "paths.json"
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(launcher.LauncherError, match="key set"):
        launcher.read_paths(path)


@pytest.mark.parametrize("repair,attempt", [(True, 1), (0, True), (3, 1), (0, 7)])
def test_repair_and_attempt_are_strict_bounded_integers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repair: object, attempt: object
) -> None:
    monkeypatch.setattr(launcher, "checkout_observation", lambda _head: (_ for _ in ()).throw(AssertionError("must fail first")))
    with pytest.raises(launcher.LauncherError):
        launcher.execute(
            tmp_path / "paths.json",
            tmp_path / "evidence",
            "a" * 40,
            launcher.AUTHORITY_TOKEN,
            repair,  # type: ignore[arg-type]
            attempt,  # type: ignore[arg-type]
        )
