"""Owner for the six conditionally approved no-contact Reach cards in #488."""

import argparse
import fcntl
import os
from contextlib import closing
from pathlib import Path

import numpy as np

from reach_native import (
    BASE,
    check_occupancy,
    execute,
    pin,
    read,
    record,
    require,
    sha,
    sites,
)
from reach_forward import site
from reach_pair import (
    REG as OLD_REG,
    environment,
    paired_audit,
    preflight as old_preflight,
    setup,
)
from reach_adapter_student import AdapterStudent, governor
from reach_intent_replay import compile_path, frozen_checks, standing
from reach_student import Student
from reach_policy_input import sensor_input
from task_space_intent import ReachEpisode

ORDER = (
    "G00-RAW",
    "G00-DIRECT",
    "G00-GEOMETRY",
    "G01-GEOMETRY",
    "G01-DIRECT",
    "G01-RAW",
)
EVIDENCE = BASE / "reports/SLA_GO2_V18/Z1_ATTRIBUTION"


def verify_registration(r):
    require(
        r["issue"] == 488
        and r["allocation"] == "CONDITIONALLY_APPROVED"
        and tuple(r["slots"]) == ORDER
        and r["max_ticks_per_card"] == 20000
        and r["max_total_ticks"] == 120000
        and r["max_forwards_per_model_card"] == 2
        and r["max_forwards"] == 8
        and r["updates"] == r["contact"] == 0,
        "different science allocation",
    )


def preflight(root, layouts):
    from scripts.pi0292.eval_contract import matrix10

    old = read(OLD_REG)
    prev = old_preflight(root, layouts, old)
    fit = read(EVIDENCE / "DIRECT_FIT.json")
    require(
        len(fit["rows"]) == 19 and fit["integrations"] == 0,
        "missing independent Corelab fit sweep",
    )
    latest = read(EVIDENCE / "WRIST_MASK_CUSTODY.json")
    require(
        latest["close"]["returned"] == 2 and latest["leases_closed"], "I4 is incomplete"
    )
    semantic = read(EVIDENCE / "INTERFACE_AUDIT.json")
    require(
        len(semantic["rows"]) == 19 and semantic["integrations"] == 0,
        "I2/I3 incomplete",
    )
    horizon = np.arange(221) * 0.05
    limits = read(BASE / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")[
        "limits"
    ]
    episodes = {}
    tests = []
    for case in standing(root):
        atom = case["at"]
        r = matrix10(np.asarray(atom["target_endpoint_EEF10"]))
        p = matrix10(np.asarray(atom["installed_EEF10"]))
        actual = np.asarray(atom["actual_TCP"])
        panel = "Z1_REACH_483" if case["id"].startswith("G00") else "Z1_REACH_483_G01"
        archived = root / "v18" / panel / case["id"].rsplit("-", 1)[0]
        _, sensor = sensor_input(
            root,
            archived,
            archived / case["episode"],
            case["tick"],
            with_visual_goal=True,
        )
        obj = sensor["red_rgb"]
        require(
            obj == case["obj"] or all(obj[k] == case["obj"][k] for k in obj),
            "live-format red estimate changed",
        )
        require(
            np.allclose(sensor["actual_tcp"], actual, rtol=0, atol=1e-8),
            "live actual TCP mismatch",
        )
        episode = case["episode"]
        first = case["entry"]["tick"] == case["tick"]
        if first:
            episodes[episode] = ReachEpisode(episode, actual, r, obj)
        controller = episodes[episode]
        for treat in ("DIRECT", "GEOMETRY"):
            row = dict(id=case["id"], treatment=treat)
            try:
                candidate = controller.plan(
                    episode,
                    actual,
                    r,
                    obj,
                    method=treat,
                    model_world=case["curve"].knots if treat == "DIRECT" else None,
                )
            except ValueError as exc:
                row["refused"] = str(exc)
                require(not first, "entry proposal refused")
            else:
                require(first, "old deviating successor accepted")
                compiled = compile_path(candidate, None, r, p, limits)
                poses, _, _ = compiled.at(horizon)
                governor(controller, poses)
                row["sweep"] = frozen_checks(root, case, candidate, compiled, r, p)
                require(
                    row["sweep"]["rejection"] is None, "initial scene sweep refused"
                )
            tests.append(row)
    require(len(tests) == 6 and len(episodes) == 2, "full fixture denominator")
    _, info = site(root)
    files = (
        "reach_adapter_panel.py",
        "reach_adapter_student.py",
        "reach_forward.py",
        "reach_policy_input.py",
        "reach_student.py",
        "reach_pair.py",
        "standing_reach.py",
        "reach_native.py",
        "task_space_intent.py",
        "reach_intent_replay.py",
    )
    return dict(
        closed_initial_stop=prev["qualification_sha256"],
        old=prev,
        preflight=tests,
        code={k: sha(Path(__file__).with_name(k)) for k in files},
        gpu=info["gpu"],
        static_fit_sha256=sha(EVIDENCE / "DIRECT_FIT.json"),
        mask_close_sha256=sha(EVIDENCE / "WRIST_MASK_CUSTODY.json"),
        integration_ticks=0,
        new_forwards=0,
    )


def run_slot(root, dest, slot, layouts, pins, receiver):
    from d1_matched_owner import with_case_lease
    from scripts.pi0292.durable import OwnerLock

    profile, base = layouts[int(slot.startswith("G01"))]
    treatment = slot.rsplit("-", 1)[1]
    student = Student(root) if treatment == "RAW" else AdapterStudent(root, treatment)
    os.sched_setaffinity(0, base["cpus"])
    case = dest / slot
    with closing(OwnerLock(Path(base["mobile_world_lock"]))):
        check_occupancy(base)
        require(
            not any(
                (p / "STOP.json").exists() for p in (dest, Path(base["campaign_root"]))
            ),
            "operator STOP",
        )
        return with_case_lease(
            case,
            base,
            pins,
            sha(dest / "REGISTRATION.json"),
            {
                "scope": "#488_PAIRED_NO_CONTACT_REACH",
                "max_ticks": 20000,
                "campaign_root": str(dest),
                "outer_library": str(receiver),
                "frozen_files": {**base["frozen_files"], str(receiver): sha(receiver)},
                "native_environment": environment(base, case),
            },
            lambda spec, sp: execute(
                base, sp, spec, profile, student=student, scorer=paired_audit
            ),
        )


def main(args):
    root = args.root.resolve()
    os.umask(0o077)
    require(
        os.environ.get("REACH_SHARED_GPU") == "1", "expected approved shared GPU mode"
    )
    layout = sites(root)
    setup(root, layout)
    dest = root / "v18/Z1_ADAPTER_488"
    require(not dest.exists(), "preallocated samples already opened")
    approval = read(args.allocation)
    verify_registration(approval)
    proof = preflight(root, layout)
    if args.preflight:
        record(args.preflight, proof)
        return
    require(read(args.execute) == proof, "CPU or frozen code preflight changed")
    old = read(OLD_REG)
    pins = read(root / "v18/Q_E1_482_H0/G00-H0/EXECUTION_SPEC.json")["b2_code"]
    receiver = pin(
        root / "v18/receiver-preview02/student_grip.so", old["receiver_sha256"]
    )
    dest.mkdir(mode=0o700)
    record(dest / "REGISTRATION.json", approval)
    record(
        dest / "QUALIFICATION.json",
        dict(
            ready_for_488=True,
            registration_sha256=sha(dest / "REGISTRATION.json"),
            proof=proof,
        ),
    )
    results = []
    try:
        for slot in ORDER:
            cases_model = slot.endswith(("-RAW", "-DIRECT"))
            if cases_model:
                lock, _ = site(root)
                with lock.open("rb") as lease:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.environ["REACH_CARD_GPU_FD"] = str(lease.fileno())
                    try:
                        value = run_slot(root, dest, slot, layout, pins, receiver)
                    finally:
                        del os.environ["REACH_CARD_GPU_FD"]
            else:
                value = run_slot(root, dest, slot, layout, pins, receiver)
            c = dest / slot
            require(
                read(c / "control/LEASE.json")["status"] == "CLOSED",
                "world still leased",
            )
            entered = sum(
                1
                for fp in c.glob("query-*/events.jsonl")
                if "FORWARD_ENTERED" in fp.read_text()
            )
            results.append(dict(slot=slot, forwards=entered, **value))
            if not value.get("completed") or not value.get("outcome"):
                break
    finally:
        started = [s for s in ORDER if (dest / s / "NATIVE_REQUEST.json").exists()]
        record(
            dest / "CLOSE.json",
            dict(
                results=results,
                started=started,
                unopened=[s for s in ORDER if s not in started],
                integrations=sum(r.get("ticks", 0) for r in results),
                forwards=sum(r.get("forwards", 0) for r in results),
                updates=0,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--allocation", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", type=Path)
    mode.add_argument("--preflight", type=Path)
    main(parser.parse_args())
