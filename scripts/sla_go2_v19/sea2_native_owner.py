"""Bind the four SEA2 calibration cases to the existing standing site owner."""

import argparse
import os
import time
from contextlib import closing
from pathlib import Path

import numpy as np

from sea2_native_profile_preflight import CARDS, IDENTITIES, calibration_student, stop_predecessor
from sea2_native_score import score_case
from sea2_preflight import sealed_archive
from sea_reference import BASE, receiver_clock_limits
from reach_native import check_occupancy, configure, execute, pin, read, record, require, sha, sites
from reach_pair import environment

REPORT = BASE / "reports/SLA_GO2_V19/SEA2"
REQUEST = REPORT / "NATIVE_PROFILE_REQUEST.json"
CAMPAIGN = "SEA2-NATIVE-KNOWN-PHYSICAL-01"
ORDER = [c[0] for c in CARDS]


def inputs(root):
    request = read(REQUEST)
    evidence = read(pin(REPORT / "NATIVE_PROFILE_CPU.json", request["cpu_evidence_sha256"]))
    require(request["id"] == CAMPAIGN and request["allocation"] is None
            and [x["id"] for x in request["slots"]] == ORDER
            and request["physics_limits"] == {
                "native_starts": 4, "ticks_per_start": 20000, "total_ticks": 80000,
                "contact": False, "real_hardware": False, "model_forwards": 0, "optimizer_updates": 0,
            }
            and evidence["actual_native_starts"] == evidence["physics_ticks"] == 0,
            "calibration request differs from the CPU-verified science slots")
    by_layout = dict(zip(("G00", "G01"), sites(root), strict=True))
    versions = request["held_fixed"]["runtime_identities"]
    receiver = pin(root / "v18/receiver-preview02/student_grip.so", versions["outer_library"])
    sealed = sealed_archive(root)
    receipt = read(sealed(root / "v18/Z1_REACH_483/G00-H0/EXECUTION_SPEC.json"))
    pins = receipt["b2_code"]
    for filename, expected in pins.items():
        pin(BASE / "scripts/sla_transfer_v14" / filename, expected)

    for expected, card in zip(request["slots"], CARDS, strict=True):
        name, layout, xyz, euler = card
        _, base = by_layout[layout]
        require(
            expected == dict(id=name, layout=layout, xyz_m=list(xyz), xyz_euler_deg=list(euler),
                             frozen_scene_sha256=base["frozen_files"][base["model_path"]]),
            "slot geometry or intent differs from the registration",
        )
        for field in IDENTITIES[1:]:
            require(base["frozen_files"][base[field]] == versions[field], "different frozen executable")
        for path, digest in base["frozen_files"].items():
            pin(Path(path), digest)
        pin(Path(base["model_path"]), expected["frozen_scene_sha256"])
    limits = receiver_clock_limits(read(BASE / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")["limits"])
    return request, evidence, by_layout, receiver, pins, limits


def verify_allocation(allocation, binding):
    require(
        allocation.get("issued") is True
        and isinstance(allocation.get("authorization_id"), str)
        and bool(allocation["authorization_id"].strip())
        and allocation.get("candidate_sha256") == binding["candidate_sha256"]
        and allocation.get("cold_proof_sha256") == binding["cold_proof_sha256"]
        and allocation.get("id") == CAMPAIGN
        and allocation.get("slots") == ORDER
        and allocation.get("physics_limits") == binding["physics_limits"]
        and type(allocation.get("valid_until_epoch")) in (int, float)
        and np.isfinite(allocation["valid_until_epoch"])
        and allocation["valid_until_epoch"] > time.time(),
        "independent matching SEA2 allocation is missing or expired",
    )


def code_fingerprints():
    base = Path(__file__).parent
    local = [
        "sea2_native_owner.py", "sea2_native_score.py",
        "sea2_native_profile_preflight.py", "sea_reference.py", "sea2_preflight.py",
        "sea2_owner_cold.py",
    ]
    shared = [
        "sla_go2_v18/reach_native.py", "sla_go2_v18/standing_reach.py",
        "sla_go2_v18/reach_pair.py", "sla_transfer_v15/d1_matched_owner.py",
        "sla_transfer_feedback/native_known.py",
    ]
    return {
        **{n: sha(base / n) for n in local},
        **{n: sha(BASE / "scripts" / n) for n in shared},
    }


def binding(root, run_inputs):
    request, evidence, layouts, receiver, _, _ = run_inputs
    stop = stop_predecessor(root, sealed_archive(root))
    require(stop == evidence["predecessor_stop"], "STOP predecessor changed")
    return dict(
        profile=CAMPAIGN, candidate_sha256=sha(REQUEST), cpu_evidence_sha256=request["cpu_evidence_sha256"],
        physics_limits=request["physics_limits"], slots=ORDER, receiver_sha256=sha(receiver),
        code=code_fingerprints(), world_domains={k: v[1]["ROS_DOMAIN_ID"] for k,v in layouts.items()},
        stopped_at=stop["stop_tick"],
    )


def run_slot(dest, card, run_inputs, allocation):
    from d1_matched_owner import with_case_lease
    from scripts.pi0292.durable import OwnerLock

    name, layout, _, _ = card
    _, _, sites_by_layout, receiver, pins, limits = run_inputs
    profile, base = sites_by_layout[layout]
    case = dest / name
    os.sched_setaffinity(0, base["cpus"])
    with closing(OwnerLock(Path(base["mobile_world_lock"]))):
        check_occupancy(base)
        require(not any((p / "STOP.json").exists() for p in (dest, Path(base["campaign_root"]))),
                "operator STOP")
        require(time.time() + base["wall_limit_s"] <= allocation["valid_until_epoch"],
                "science allocation expires before this lease")
        grip = calibration_student(card, limits)
        fields = dict(
            scope=CAMPAIGN, max_ticks=20000, campaign_root=str(dest), outer_library=str(receiver),
            frozen_files={**base["frozen_files"], str(receiver): sha(receiver)},
            native_environment=environment(base, case),
        )
        return with_case_lease(
            case, base, pins, sha(dest / "REGISTRATION.json"), fields,
            lambda spec, sp: execute(
                base, sp, spec, profile, student=grip,
                scorer=lambda path, result, spec: score_case(path, result, spec, card, limits),
            ),
        )


def execute_panel(root, prepared, cold, allocation):
    _, _, layouts, _, _, _ = prepared
    base = layouts["G00"][1]
    scratch = root / "v19/sea2-native-owner-scratch"
    for part in ("tmp", "cache", "logs"):
        (scratch / part).mkdir(parents=True, exist_ok=True, mode=0o700)
    configure(dict(base, native_environment=environment(base, scratch)))
    dest = root / "v19" / CAMPAIGN
    dest.mkdir(mode=0o700)
    record(dest / "REGISTRATION.json", read(REQUEST))
    record(dest / "ALLOCATION.json", allocation)
    record(dest / "OWNER_COLD.json", cold)
    results = []
    try:
        for card in CARDS:
            verdict = run_slot(dest, card, prepared, allocation)
            lease = read(dest / card[0] / "control/LEASE.json")
            require(lease["status"] == "CLOSED", "live lease after calibration card")
            results.append(dict(slot=card[0], **verdict))
            if not verdict.get("completed"):
                break
    finally:
        requested = [slot for slot in ORDER if (dest / slot / "NATIVE_REQUEST.json").is_file()]
        results_by_slot = {x["slot"]: x for x in results}
        record(dest / "PANEL_CLOSE.json", dict(
            results=results, requested=requested, unstarted=[x for x in ORDER if x not in requested],
            unresolved_requested=[x for x in requested if x not in results_by_slot],
            observed_integrations=sum(x["ticks"] for x in results), frozen_forwards=0, updates=0,
        ))


def main(args):
    os.umask(0o077)
    root = args.root.resolve()
    if args.cold:
        from sea2_owner_cold import cold_check

        record(args.cold, cold_check(root, inputs(root)))
        return

    require(args.preflight and args.allocation_sha256, "execute needs cold proof and exact allocation hash")
    cold = read(args.preflight)
    allocation = read(pin(args.execute, args.allocation_sha256))
    verify_allocation(allocation, dict(
        candidate_sha256=sha(REQUEST), cold_proof_sha256=sha(args.preflight),
        physics_limits=read(REQUEST)["physics_limits"],
    ))
    prepared = inputs(root)
    require(cold.get("status") == "CPU_OWNER_BINDING_VERIFIED"
            and cold.get("binding") == binding(root, prepared)
            and cold.get("new_native_starts") == cold.get("physics_ticks") == 0,
            "owner inputs changed since verified cold binding")
    execute_panel(root, prepared, cold, allocation)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cold", type=Path)
    mode.add_argument("--execute", type=Path, help="independent scientific allocation file")
    cli.add_argument("--allocation-sha256")
    cli.add_argument("--preflight", type=Path, help="exact cold acceptance result")
    main(cli.parse_args())
