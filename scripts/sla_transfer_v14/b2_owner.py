"""Site entry point for the two explicitly approved B2 cards. Never replay a native request."""

import argparse
import hashlib
import json
import os
import socket
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from identity import read, sha

T0 = "T0_STATIC_HOLD"
T1 = "T1_CLOCK_FAULT_STOP"
CODE = (
    "b2_owner.py",
    "b2_native.py",
    "b2_receiver.py",
    "clock_guard.py",
    "identity.py",
)


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def pinned(path, pin):
    require(sha(path) == pin, "content mismatch: " + str(path))


def baseline(root):
    prior = read(root / "v13/run01/S0_ACTUAL.json")
    teacher = next(
        Path(k) for k in prior["inputs"] if k.endswith("/EXECUTION_SPEC.json")
    )
    pinned(teacher, prior["inputs"][str(teacher)])
    bound = read(teacher)["mobile_interface_qualification"]
    proof = Path(bound["path"])
    pinned(proof, bound["sha256"])
    spec_path = proof.parent / "EXECUTION_SPEC.json"
    pinned(spec_path, read(proof.parent / "control/LEASE.json")["spec_sha256"])
    spec = read(spec_path)
    require(read(proof)["completed"], "unqualified reference backend")
    return spec


def configure(spec):
    overlay = spec["source_root"]
    parent = spec["trusted_source_parent"]["source_root"]
    paths = [overlay, overlay + "/src", parent, parent + "/src"]
    if sys.executable != spec["python"] or os.environ.get("B2_CONFIGURED") != "1":
        env = {
            **os.environ,
            **spec["native_environment"],
            "CUDA_VISIBLE_DEVICES": "",
            "B2_CONFIGURED": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        env["PYTHONPATH"] = os.pathsep.join([*paths, env["PYTHONPATH"]])
        os.execve(spec["python"], [spec["python"], "-B", *sys.argv], env)
    import go2arx5_vla.world_model
    import scripts.wm02

    go2arx5_vla.world_model.__path__.insert(0, overlay + "/src/go2arx5_vla/world_model")
    scripts.wm02.__path__.insert(0, overlay + "/scripts/wm02")


def verify(spec, registration):
    from b2_native import TechnicalAdapter
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile

    from scripts.pi0294 import native_receiver

    src = Path(spec["source_root"])
    parent = spec["trusted_source_parent"]
    require(
        socket.gethostname() == "corelab-4080s" and os.getuid() == 2009,
        "wrong execution site",
    )
    for key in ("mobile_guard_qualification", "source_seal"):
        pinned(Path(spec[key]["path"]), spec[key]["sha256"])
    pinned(Path(parent["spec_path"]), parent["spec_sha256"])
    for base, files in (
        (src, spec["source_files"]),
        (Path(parent["source_root"]), read(Path(parent["spec_path"]))["source_files"]),
    ):
        for name, pin in files.items():
            pinned(base / name, pin)
    for name, pin in spec["frozen_files"].items():
        pinned(Path(name), pin)
    qual = read(Path(spec["mobile_guard_qualification"]["path"]))
    require(
        qual["passed"] and qual["original_joint_and_collision_guards_retained"],
        "mobile safety guard",
    )
    for filename, key in (
        ("mobile_safety_binding", "binding_sha256"),
        ("model_path", "model_sha256"),
        ("mobile_safety_library", "library_sha256"),
    ):
        pinned(Path(spec[filename]), qual[key])
    require(
        qual["library_sha256"] == spec["mobile_safety_library_sha256"],
        "inconsistent guard library",
    )
    for module, relative in (
        (mobile, "scripts/wm02/mobile_boundary_runner_v1.py"),
        (native_receiver, "scripts/pi0294/native_receiver.py"),
        (
            sys.modules[TechnicalAdapter.__base__.__module__],
            "src/go2arx5_vla/world_model/target_sequential.py",
        ),
    ):
        pinned(Path(module.__file__), spec["source_files"][relative])
    cards = registration["cards"]
    require(
        [(c["id"], c["total_tick_cap"]) for c in cards] == [(T0, 6203), (T1, 6703)]
        and registration["max_requests"] == 2
        and registration["total_tick_cap"] == 12906,
        "registration",
    )
    return {name: sha(Path(__file__).with_name(name)) for name in CODE}


def dry_receiver(spec):
    import b2_receiver
    from clock_replay import issue
    from ocs2_msgs.msg import MpcObservation
    from shadow import OfflineManager, observe

    from scripts.pi0292.durable import write

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        env = {
            "B2_CARD": T1,
            "PI0294_EPISODE": out.name,
            "PI0294_PLANNER_TRACE": str(out / "planner.jsonl"),
            "B2_RECEIVER_PIN": spec["source_files"][
                "scripts/pi0294/native_receiver.py"
            ],
        }

        def test_lifecycle(ctx, manager):
            class OfflineGuard(manager, OfflineManager):
                def create_timer(self, *args, **kwargs):
                    pass

            node = OfflineGuard()
            observe(node, 10.0)
            issue(node, out.name, 10.0, 0)
            before = len(node.target_pub)
            write(out / "B2_REWIND_REQUEST.json", {"episode": out.name, "step": 6678})
            node.check_fault_request()
            node.on_observation(MpcObservation())
            node.publish()
            terminal = read(out / "GUARD_TERMINAL.json")
            require(
                terminal["reference_invalidated"]
                and terminal["fault"]["reason"] == "rewind"
                and terminal["hold_request_published"]
                and len(node._r4_mode) == 1
                and len(node.target_pub) == before,
                "clock fault callback preflight",
            )

        with (
            patch.dict(os.environ, env),
            patch.object(b2_receiver.receiver_lifecycle, "run", test_lifecycle),
        ):
            b2_receiver.main()


def check_occupancy(spec):
    now = time.time()
    for row in read(Path(spec["registry"]))["entries"]:
        if row.get("ROS_DOMAIN_ID") != spec["ROS_DOMAIN_ID"] or not row.get(
            "local_lease"
        ):
            continue
        path = Path(row["local_lease"])
        if path.exists():
            lease = read(path)
            require(
                lease.get("status") != "ACTIVE"
                or lease.get("deadline_epoch", float("inf")) < now,
                "existing active owner: " + row.get("role", ""),
            )
    binaries = {
        Path(spec[k]).name[:15]
        for k in ("native_binary", "solver_binary", "controller_binary")
    }
    for proc in Path("/proc").glob("[0-9]*/comm"):
        try:
            name = proc.read_text().strip()
        except (FileNotFoundError, ProcessLookupError):
            continue
        require(name not in binaries, "native process already active: " + name)


def run(root, group, base, registration, pins):
    from b2_native import run_card

    from scripts.pi0292.durable import OwnerLock, write

    world_path = Path(base["mobile_world_lock"])
    registry = Path(base["registry"])
    require(not group.exists(), "prior B2 campaign exists; no automatic replay")
    os.sched_setaffinity(0, base["cpus"])
    with closing(OwnerLock(world_path)):
        check_occupancy(base)
        group.mkdir(mode=0o700, parents=True, exist_ok=False)
        write(
            group / "OWNER.json",
            {"pid": os.getpid(), "domain": base["ROS_DOMAIN_ID"], "epoch": time.time()},
        )
        for card in registration["cards"]:
            check_occupancy(base)
            folder = group / card["id"]
            for sub in ("control", "tmp", "logs", "cache", "data"):
                (folder / sub).mkdir(parents=True, mode=0o700)
            require(
                not any(
                    (p / "STOP.json").exists()
                    for p in (group, Path(base["campaign_root"]), folder)
                ),
                "operator stop",
            )
            spec = dict(
                base,
                role="SLA-V14-B2-" + card["id"],
                owner=group.name,
                registered_root=str(folder),
                mobile_group=str(group),
                scope="SLA_V14_B2_CONTACT_FREE_MUJOCO",
                b2_card=card["id"],
                b2_code=pins,
                max_ticks=card["total_tick_cap"],
                deadline_epoch=time.time() + card["wall_limit_s"],
                stage_identity_sha256=hashlib.sha256(
                    json.dumps([card, pins, group.name], sort_keys=True).encode()
                ).hexdigest(),
                mobile_condition={
                    "reference": "static_measured_TCP",
                    "card": card["id"],
                },
            )
            for key in ("logs", "cache", "data", "tmp"):
                spec[key] = str(folder / key)
            spec_path = folder / "EXECUTION_SPEC.json"
            write(spec_path, spec)
            lease_path = folder / "control/LEASE.json"
            lease = {
                "status": "ACTIVE",
                "spec_sha256": sha(spec_path),
                "owner": spec["owner"],
                "run_uuid": folder.name,
                "deadline_epoch": spec["deadline_epoch"],
            }
            with closing(OwnerLock(world_path.parent / "registration.lock")):
                check_occupancy(base)
                reg = read(registry)
                require(
                    not any(r.get("path") == str(folder) for r in reg["entries"]),
                    "duplicate registered path",
                )
                reg["entries"].append(
                    {
                        "path": str(folder),
                        "run_uuid": folder.name,
                        "role": spec["role"],
                        "source_sha": spec["source_sha"],
                        "owner": spec["owner"],
                        "scope": spec["scope"],
                        "executor_node": spec["host"],
                        "local_lease": str(lease_path),
                        "ROS_DOMAIN_ID": spec["ROS_DOMAIN_ID"],
                        "corelab_world_owner": str(world_path),
                    }
                )
                write(registry, reg)
                write(lease_path, lease)
            try:
                result = run_card(spec, spec_path)
                print(
                    json.dumps(
                        {
                            k: result.get(k)
                            for k in (
                                "card",
                                "completed",
                                "journal_integrations",
                                "failure",
                            )
                        }
                    ),
                    flush=True,
                )
            finally:
                write(
                    lease_path, dict(lease, status="CLOSED", closed_epoch=time.time())
                )
            if not result["completed"]:
                return 1
    return 0


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("mode", choices=("preflight", "execute"))
    cli.add_argument("root", type=Path)
    args = cli.parse_args()
    root = args.root.resolve()
    base = baseline(root)
    configure(base)
    registration_path = (
        root
        / "v14/qualification_release/technical_registration/B2_TECHNICAL_REGISTRATION.json"
    )
    pinned(
        registration_path,
        "3e07e809b912e040d2a713436e540c7d8cadf3b76b96afb4b209b737b7605643",
    )
    registration = read(registration_path)
    pins = verify(base, registration)
    dry_receiver(base)
    if args.mode == "preflight":
        print("CPU preflight passed", flush=True)
        return 0
    group = root / "v14/b2-technical-20261123-01"
    return run(root, group, base, registration, pins)


if __name__ == "__main__":
    raise SystemExit(main())
