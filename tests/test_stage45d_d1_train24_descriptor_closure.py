"""CPU/static tests for the deterministic Stage45D TRAIN24 closure."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45d_d1_train24_descriptor_closure as closure  # noqa: E402


def _projection() -> dict:
    timestamps = [float(index) * 0.05 for index in range(closure.HORIZON)]
    action = [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5] for _ in timestamps]
    return {
        "path": "/accepted/episode_000000.parquet",
        "sha256": "a" * 64,
        "state_v1": action[0],
        "expert_physical_action": action,
        "source_timestamps": timestamps,
        "observation_timestamp": timestamps[0],
    }


def test_static_guard_has_no_model_side_top_level_imports() -> None:
    result = closure.static_guard()
    assert result["status"] == "PASS"
    assert result["top_level_forbidden_imports"] == []
    tree = ast.parse((ROOT / "scripts/stage45d_d1_train24_descriptor_closure.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert not imported.intersection({"torch", "transformers", "gr00t", "cuda"})


def test_ppm_round_trip_is_exact_and_cpu_only(tmp_path: Path) -> None:
    rgb = np.zeros(closure.RGB_SHAPE, dtype=np.uint8)
    rgb[0, 0] = [1, 2, 3]
    path = tmp_path / "frame.ppm"
    closure.write_ppm(path, rgb)
    assert np.array_equal(closure.read_ppm(path), rgb)


def test_descriptor_payload_is_materialized_and_target_excluded(tmp_path: Path) -> None:
    projection = _projection()
    body_video = tmp_path / "body.mp4"
    wrist_video = tmp_path / "wrist.mp4"
    body_video.write_bytes(b"body")
    wrist_video.write_bytes(b"wrist")
    body_frame = tmp_path / "body.ppm"
    wrist_frame = tmp_path / "wrist.ppm"
    payload = closure._descriptor_payload(
        index=0,
        projection=projection,
        body_frame=body_frame,
        wrist_frame=wrist_frame,
        body_video=body_video,
        wrist_video=wrist_video,
    )
    assert payload["schema"] == "SameSceneExpertSnapshotV1"
    assert payload["target_excluded_from_model_input"] is True
    assert payload["provenance"]["target_xyz_in_model_input"] is False
    assert payload["provenance"]["language"] == closure.LANGUAGE
    assert payload["provenance"]["new_scientific_data_generated"] is False
    assert payload["future_desired_eef_9d"] == [row[:9] for row in projection["expert_physical_action"]]
    assert payload["future_gripper"] == [[row[9]] for row in projection["expert_physical_action"]]


def test_source_projection_hashes_are_stable() -> None:
    projection = _projection()
    first = closure._source_projection_hashes(projection)
    second = closure._source_projection_hashes(projection)
    assert first == second
    assert set(first) == {"state_v1", "source_timestamps", "expert_physical_action", "language"}
