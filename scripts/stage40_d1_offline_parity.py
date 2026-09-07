#!/usr/bin/env python3
"""CPU-only parity check for the Stage40 StructuredActionV1 live decoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
D0_ROOT = Path("/scratch/e1192548/leon-storage/runs/stage40_d0_decode_20260907")
HISTORICAL_RUN = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage40_c_20260907_r1/stage40_live_604652.hopper-m-02"
)
STATS = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage38_structured_groot_offline_20260906/stage38d_train_a7f6880/"
    "experiment_cfg/structured_action_statistics.json"
)
OUT = Path("/scratch/e1192548/leon-storage/runs/stage40_d1_offline_parity_20260907")
SEEDS = (2223, 2228, 2233, 2238)
TS = np.arange(40, dtype=np.float64) * 0.05


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> int:
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from go2arx5_vla.safety.trajectory_gate import TrajectorySafetyGateV1
    from go2arx5_vla.stage40_structured_decode import decode_raw_structured_action
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    from go2arx5_vla.contracts.trajectory import TrajectoryChunkV1

    d0 = json.loads((D0_ROOT / "stage40_d0_decode_result.json").read_text())
    by_seed = {int(item["seed"]): item for item in d0["records"]}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        response_npz = HISTORICAL_RUN / "mailbox" / "responses" / f"stage40-s{seed}-q0.npz"
        expected_npz = D0_ROOT / f"seed_{seed}_derived_paths.npz"
        with np.load(response_npz, allow_pickle=False) as data:
            raw = np.asarray(data["raw_universal_action"])
            state = np.asarray(data["input_state"], dtype=np.float32)
        with np.load(expected_npz, allow_pickle=False) as data:
            expected = np.asarray(data["structured_physical"], dtype=np.float64)
            expected_ts = np.asarray(data["timestamps"], dtype=np.float64)
        if not np.array_equal(expected_ts, TS):
            raise AssertionError(f"seed {seed}: D0 timestamps changed")
        chunk, physical, details = decode_raw_structured_action(
            raw,
            state,
            stats_path=STATS,
            source_timestamps=TS,
            observation_timestamp=0.0,
            chunk_id=f"stage40-d1-parity-{seed}",
        )
        max_abs = float(np.max(np.abs(physical - expected)))
        if max_abs > 1e-12:
            raise AssertionError(f"seed {seed}: decoded parity max_abs={max_abs}")
        record = by_seed[seed]
        expected_detail = record["structured_action_decode"]
        expected_sha = expected_detail["physical_trajectory_sha256_float64"]
        if details["physical_sha256_float64"] != expected_sha:
            raise AssertionError(f"seed {seed}: physical SHA mismatch")
        gate = TrajectorySafetyGateV1().evaluate(
            chunk,
            now=0.0,
            achieved_position=state[:3],
            achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[3:9])),
        )
        gate_dict = gate.as_dict()
        expected_gate = expected_detail["safety_gate"]
        if gate_dict["decision"] != expected_gate["decision"]:
            raise AssertionError(f"seed {seed}: gate decision mismatch")
        rows.append(
            {
                "seed": seed,
                "raw_sha256": sha256_array(raw),
                "state_v1_sha256": sha256_array(state),
                "raw_shape": list(raw.shape),
                "physical_shape": list(physical.shape),
                "physical_sha256_float64": details["physical_sha256_float64"],
                "expected_d0_physical_sha256_float64": expected_sha,
                "max_abs_diff_vs_d0": max_abs,
                "stats_sha256": details["stats_sha256"],
                "mask_sha256_float64": details["mask_sha256_float64"],
                "anchor_position_error_m": float(np.linalg.norm(physical[0, :3] - state[:3])),
                "anchor_rotation_error_rad": float(
                    np.arccos(
                        np.clip(
                            abs(
                                float(
                                    np.dot(
                                        matrix_to_quat(rot6d_rows_to_matrix(state[3:9])),
                                        matrix_to_quat(rot6d_rows_to_matrix(physical[0, 3:9])),
                                    )
                                )
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                    * 2.0
                ),
                "timing_fractions": details["timing_fractions"],
                "timing_sum": float(sum(details["timing_fractions"])),
                "gate": gate_dict,
            }
        )
    result = {
        "schema": "STAGE40_D1_OFFLINE_STRUCTUREDACTION_PARITY_V1",
        "stage": "Stage40-D1",
        "mode": "CPU_ONLY_ZERO_MODEL_ZERO_WBC",
        "source_d0_result_sha256": hashlib.sha256(
            (D0_ROOT / "stage40_d0_decode_result.json").read_bytes()
        ).hexdigest(),
        "stats_sha256": "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206",
        "seeds": list(SEEDS),
        "rows": rows,
        "all_exact": all(item["max_abs_diff_vs_d0"] == 0.0 for item in rows),
        "status": "PASS",
        "no_model_calls": True,
        "no_wbc_calls": True,
        "gpu_submitted": False,
    }
    path = OUT / "stage40_d1_offline_parity.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(path), "status": result["status"], "all_exact": result["all_exact"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
