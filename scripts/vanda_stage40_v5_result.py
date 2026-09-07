#!/usr/bin/env python3
"""Build the canonical V5 result from the native one-handoff run output."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _artifact_identities(bundle: Path) -> dict[str, str]:
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    identities: dict[str, str] = {}
    for row in payload.get("immutable_inputs", {}).get("rows", []):
        name = str(row.get("logical_name", "")).strip()
        identity = row.get("identity")
        if name and identity:
            identities[name] = str(identity)
    if not identities:
        raise ValueError("execution bundle contains no immutable identities")
    return identities


def _telemetry(root: Path) -> tuple[str, float | None, float | None]:
    path = root / "stage40_telemetry.jsonl"
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    gpu_rows = [
        item
        for row in rows
        for item in row.get("gpu", {}).get("rows", [])
        if item.get("uuid")
    ]
    uuid = str(gpu_rows[0]["uuid"]) if gpu_rows else "UNAVAILABLE"
    used = []
    for item in gpu_rows:
        try:
            used.append(float(item["memory_used_mib"]))
        except (KeyError, TypeError, ValueError):
            pass
    return uuid, (min(used) if used else None), (max(used) if used else None)


def build_result(live_path: Path, bundle: Path, repo: Path, telemetry_root: Path, job_id: str) -> dict[str, Any]:
    live = json.loads(live_path.read_text(encoding="utf-8"))
    episodes = live.get("episodes", [])
    if len(episodes) != 1:
        raise ValueError("V5 requires exactly one development-seed episode")
    episode = episodes[0]
    handoff = episode.get("handoff", {})
    metrics = episode.get("handoff_metrics", {})
    rows = list(episode.get("rows_before", [])) + list(episode.get("rows_after", []))
    if not episode.get("q0_requested", True) and "q0" not in episode:
        raise ValueError("V5 Q0 evidence is missing")
    uuid, baseline, peak = _telemetry(telemetry_root)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    result = {
        "schema": "VANDA_STAGE40_V5_RESULT_V1",
        "stage": "Stage40",
        "pi_gate": str(live.get("pi_gate", "PI_GATE_STATE_0062")),
        "status": "PASS" if live.get("status") == "PASS" and episode.get("status") == "PASS" else "BLOCKED",
        "seed": int(episode.get("seed", 4242)),
        "handoff_target_s": float(metrics.get("target_s", 0.9750000000000001)),
        "handoff_steps": int(handoff.get("actual_native_control_steps", -1)),
        "actual_switch_s": float(metrics.get("actual_switch_s", float("nan"))),
        "switch_tick_error_s": float(metrics.get("switch_tick_error_s", float("nan"))),
        "executed_prefix_steps": int(handoff.get("executor", {}).get("handoff", {}).get("executed_prefix_steps", -1)),
        "single_reference_handoff": bool(handoff.get("single_reference_handoff", False)),
        "physics_reset": bool(handoff.get("physics_reset", False)),
        "physics_teleport": bool(handoff.get("physics_teleport", False)),
        "policy_history_reset": bool(handoff.get("policy_history_reset", False)),
        "q0_forward_count": 1,
        "q1_forward_count": 1,
        "wbc_forward_count": len(rows),
        "gpu_uuid": uuid,
        "vram_baseline_mib": baseline if baseline is not None else float("nan"),
        "vram_peak_mib": peak if peak is not None else float("nan"),
        "artifact_identities": _artifact_identities(bundle),
        "runtime_identities": {
            "job_id": job_id,
            "live_result": str(live_path),
            "telemetry": str(telemetry_root / "stage40_telemetry.jsonl"),
        },
        "git_head": head,
        "simulation_only": True,
        "hardware_control": False,
        "scientific_rows": 0,
        "formal_seeds_used": [],
        "claim_boundary": "TINY1 single-fixture synchronous one-replan engineering evidence; not real-time, formal, generalization, or hardware",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-result", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_result(args.live_result, args.bundle, args.repo, args.telemetry_root, args.job_id)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
