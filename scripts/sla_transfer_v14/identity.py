"""Read sealed teacher receipts and bind candidate software to the actual launcher."""

import hashlib
import itertools
import json
import os
import time
from collections import Counter
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path, obj):
    payload = json.dumps(obj, indent=2, allow_nan=False)
    with path.open("x") as stream:
        stream.write(payload + "\n")


def card_receipt(card):
    d = read(card)
    n = card.parent / "native-01"
    commands = read(n / "PROCESS_COMMANDS.private.json")
    receiver = Path(next(c for c in commands if c["name"] == "planner")["argv"][2])
    assert (
        sha(receiver) == d["source_files"][str(receiver.relative_to(d["source_root"]))]
    )
    back = {
        Path(d[k]).name: sha(Path(d[k]))
        for k in ("native_binary", "controller_binary", "solver_binary")
    }
    loaded = read(n / "LOADED_EXECUTOR_01.private.json")
    for proc, key in (
        ("sim", "native_binary"),
        ("solver", "solver_binary"),
        ("controller", "controller_binary"),
    ):
        assert d[key] in next(c["argv"] for c in commands if c["name"] == proc)
        assert loaded[proc]["executable_sha256"] == back[Path(d[key]).name]
    modes, policy_clocks = [], []
    for event in rows(n / "ROS_CONSUMPTION.private.jsonl"):
        if event["topic_key"] == "mode":
            modes.append(event["message"]["data"])
        if event["topic_key"] == "policy":
            policy_clocks.append(event["message"]["init_observation"]["time"])
    policy_delta = [b - a for a, b in itertools.pairwise(policy_clocks) if b > a]
    installed = Counter()
    accepted = []
    stops = []
    for r in rows(n / "planner.jsonl"):
        if r["kind"] == "accept":
            raw = json.loads(r["raw"])
            assert hashlib.sha256(r["raw"].encode()).hexdigest() == r["ref_id"]
            assert len(raw["ee"]["waypoints"]) == 41 and raw["_pi0294"]["ttl_s"] == 0.4
            accepted.append(r["source_mpc_s"])
        elif (
            r["kind"] == "installed"
            and r.get("ref_id")
            and r.get("bound_start") is not None
        ):
            installed[(len(r["poses"][0]), len(r["inputs"][0]))] += 1
        elif r["kind"] == "lifecycle":
            stops.append([r.get("reason"), r.get("action")])
    samples = [
        r
        for r in rows(n / "MOBILE_SAMPLES.private.jsonl")
        if r["segment"] == "MOBILE_TASK"
    ]
    receiver_log = (n / "planner.log").read_text()
    solver_log = (n / "solver.log").read_text()
    assert (
        samples and "WBC" in modes and "/mobile_manipulator_mpc_target" in receiver_log
    )
    lease = Path(d["site_world_lease"]["path"])
    assert sha(lease) == d["site_world_lease"]["sha256"]
    return {
        "binary_sha256": back,
        "receiver_sha256": sha(receiver),
        "mode_observations": modes,
        "archived_owner": d["owner"],
        "expired_lease": read(lease)["deadline_epoch"] < time.time(),
        "physics_dt_s": d["physics_dt_s"],
        "profile": d["execution_clock_profile"],
        "planner_accept": len(accepted),
        "policy_echo_rows": len(policy_clocks),
        "distinct_increasing_policy_intervals": len(policy_delta),
        "median_positive_policy_init_interval_s": median(policy_delta)
        if policy_delta
        else None,
        "max_accept_gap_s": max(b - a for a, b in itertools.pairwise(accepted)),
        "installed_state_input_widths": {str(k): v for k, v in installed.items()},
        "lifecycle": stops,
        "mobile_consumption_rows": len(samples),
        "rows_with_exact_solver_echo": sum(
            bool(r["solver_exact_installed_targets"]) for r in samples
        ),
        "rows_with_arm_and_base_output": sum(
            r["arm_requests"] > 0 and r["base_commands"] > 0 for r in samples
        ),
        "selected_reference_status": sorted(
            {r["controller_selected_reference"] for r in samples}
        ),
        "solver_config_lines": [
            l.strip()
            for l in solver_log.splitlines()
            if "DesiredFrequency'" in l or "'timeHorizon'" in l
        ],
        "evidence_sha256": {
            k: sha(n / k)
            for k in (
                "planner.jsonl",
                "MOBILE_SAMPLES.private.jsonl",
                "PROCESS_COMMANDS.private.json",
                "ROS_CONSUMPTION.private.jsonl",
                "LOADED_EXECUTOR_01.private.json",
            )
        },
        "historical_lease_sha256": sha(lease),
    }


def main(root, output):
    old = root / "v13"
    idx = read(old / "ARTIFACT_INDEX.json")
    assert all(
        sha(old / path) == value for path, value in idx["per_file_sha256"].items()
    )
    s = read(old / "run01/S0_ACTUAL.json")
    assert all(sha(Path(path)) == h for path, h in s["inputs"].items())
    canonical = read(old / "run01/CANONICAL_ACTUAL.json")["rows"]
    for r in canonical:
        f = old / "run01/arrays" / (r["id"] + "_" + r["arm"] + ".npz")
        assert (
            sha(f) == r["candidate_sha256"] == sha(old / "reproduction/arrays" / f.name)
        )
        assert sha(old / "run01/canonical" / r["file"]) == r["sha256"]
    cards = [
        card_receipt(Path(p)) for p in s["inputs"] if p.endswith("/EXECUTION_SPEC.json")
    ]
    card = read(
        next(Path(p) for p in s["inputs"] if p.endswith("/EXECUTION_SPEC.json"))
    )
    source_root = Path(card["native_root"]) / "src"
    archive = read(Path(__file__).with_name("ARCHIVE_ABI.json"))
    for name, expected in archive.items():
        files = list(source_root.rglob(name))
        assert len(files) == 1 and sha(files[0]) == expected
    grip = read(old / "run01/S2_ACTUAL.json")["windows"]
    assert len(grip) == 16 and all(
        w["G0"] == "NOT_STUDENT_OWNED_RESET" and w["G1"] == "HANDOVER_BLOCKED"
        for w in grip
    )
    assert len(canonical) == 78 and len(cards) == 6 and len(s["windows"]) == 16
    record(
        output,
        {
            "status": "PRIVATE_LOCAL_VERIFIED",
            "previous_inputs": len(s["inputs"]),
            "target_windows": 16,
            "canonical_packages": 78,
            "cards": cards,
            "runner_uid": os.getuid(),
            "matched_archive_and_site_source": archive,
            "verified_student_gripper_owners": 0,
            "current_native_lease": None,
            "new_native_requests": 0,
            "new_ticks": 0,
            "gpu_forwards": 0,
            "stop_receipt": "NOT_APPLICABLE_NO_NEW_WORLD",
        },
    )


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]), Path(sys.argv[2]))
