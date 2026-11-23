"""Read the complete registered predecessor panel, including unused and stopped cards."""

import sys
from pathlib import Path

import numpy as np
from identity import read, record, rows, sha

from go2arx5_vla.trajectory.so3 import geodesic_angle
from scripts.pi0292.eval_contract import matrix10


def main(root, out):
    b0 = read(root / "v14/final/B0_BACKEND_IDENTITY_AND_SOURCE_ACTUAL.json")
    hashes = b0["cards"][0]["binary_sha256"]
    old = read(root / "v13/run01/S0_ACTUAL.json")
    spec = read(
        next(Path(p) for p in old["inputs"] if p.endswith("/EXECUTION_SPEC.json"))
    )
    binding = spec["mobile_interface_qualification"]
    proof = Path(binding["path"])
    assert sha(proof) == binding["sha256"]
    batch = proof.parents[2]
    first = batch.with_name("mobile-wbc-sla-boundary-v1-m0-20261115-01")
    panels = [first, batch]
    summary = read(batch / "RESULT.private.json")
    expected = {
        (axis, mode)
        for axis in ("+x", "-x", "+y", "-y", "+yaw", "-yaw")
        for mode in ("WORLD_HOLD", "SMOOTH_5CM_TRANSLATION")
    }
    cards = {}
    for panel in panels:
        batch_results = {
            item["root"]: item["result"]
            for item in read(panel / "RESULT.private.json")["outcomes"]
        }
        for folder in sorted((panel / "runs").iterdir()):
            if not folder.is_dir():
                continue
            spec = read(folder / "EXECUTION_SPEC.json")
            condition = spec["mobile_condition"]
            key = (condition["axis"], condition["eef_mode"])
            assert key in expected and key not in cards
            f = folder / "RESULT.private.json"
            if not f.exists():
                assert folder.name not in batch_results
                if panel == first:
                    # These registered cards were carried into the second batch
                    # after the first card encountered a technical failure.
                    continue
                assert (
                    condition["axis"] == "-yaw" and "yaw" in summary["quarantined_axes"]
                )
                cards[key] = {
                    "condition": condition,
                    "status": "WITHHELD_YAW_QUARANTINE",
                    "ticks": 0,
                }
                continue
            result = read(f)
            assert result == batch_results[folder.name]
            n = folder / "native-01"
            loaded_file = max(n.glob("LOADED_EXECUTOR_*.private.json"))
            loaded = read(loaded_file)
            verified = {
                role: loaded[role]["executable_sha256"]
                == hashes[Path(spec[field]).name]
                for role, field in (
                    ("sim", "native_binary"),
                    ("controller", "controller_binary"),
                    ("solver", "solver_binary"),
                )
            }
            if not all(verified.values()):
                raise ValueError("Predecessor backend differs")
            samplefile = n / "MOBILE_SAMPLES.private.jsonl"
            task = (
                [r for r in rows(samplefile) if r["native_start_step"] >= 5703]
                if samplefile.exists()
                else []
            )
            metrics = {}
            if task:
                target = matrix10(
                    np.asarray([r["target_endpoint_EEF10"] for r in task])
                )
                achieved = np.asarray([r["actual_TCP"] for r in task])
                pos = np.linalg.norm(target[:, :3, 3] - achieved[:, :3, 3], axis=1)
                ang = geodesic_angle(target[:, :3, :3], achieved[:, :3, :3])
                vals = result["metrics"]
                for values, name, scale in (
                    (pos, "translation", 1000.0),
                    (ang, "rotation", 1.0),
                ):
                    rms = float(np.sqrt(np.mean(values**2)))
                    assert np.isclose(
                        rms,
                        vals[
                            "EEF_"
                            + name
                            + "_RMS_"
                            + ("m" if name == "translation" else "rad")
                        ],
                        rtol=1e-7,
                        atol=1e-10,
                    )
                    metrics[name] = {
                        "rms": rms * scale,
                        "p95": float(np.percentile(values, 95)) * scale,
                        "max": float(values.max()) * scale,
                    }
                assert len(task) == vals["task_atoms"]
                metrics["samples"] = len(task)
            status = (
                "COMPLETED"
                if result["completed"]
                else "SAFETY_STOP"
                if result["unsafe"]
                else "TECHNICAL_FAILURE"
            )
            cards[key] = {
                "condition": condition,
                "status": status,
                "ticks": result["actual_native_integrations"],
                "metrics": metrics,
                "position_units": "mm",
                "orientation_units": "rad",
                "loaded_binaries_match_B0": verified,
                "loaded_receipt": loaded_file.name,
                "result_sha256": sha(f),
                "samples_sha256": sha(samplefile) if samplefile.exists() else None,
            }
    assert set(cards) == expected
    ticks = sum(x["ticks"] for x in cards.values())
    assert (
        ticks
        == summary["native_integrations"]
        + read(batch / "EXECUTION_SPEC.json")["retained_prior_native_integrations"]
    )
    record(
        out / "HISTORICAL_TRACKING_ACTUAL.json",
        {
            "scope": "Prior M0: moving base, fixed EEF orientation; separate from the new B2 constant-base and publication-mode pairs",
            "batch_sha256": [sha(p / "RESULT.private.json") for p in panels],
            "registered": 12,
            "historical_ticks": ticks,
            "cases": list(cards.values()),
            "native_requests_this_run": 0,
            "new_native_ticks": 0,
        },
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
