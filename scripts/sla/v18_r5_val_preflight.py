"""Prepare and verify the R5 VAL producer without registering or dispatching a run."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, scope_identity, validate_frozen_assets
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import write
from scripts.sla.v18_r5_val_scope import (
    AUTHORITY, EVIDENCE, OWNER, PLAN, SCOPE, SLOT_IDS, TICK_CAP,
    check_world_lease, make_panel, plan, validate_allocation, validate_member,
)
from scripts.sla.v18_r5_val_teacher import prepare_case, storage_and_memory
from scripts.wm02.multitask_stage_cards_v6 import panel_cards, pin, require_unstarted_cases

OVERLAY = (
    "scripts/sla/v18_r5_val_preflight.py",
    "scripts/sla/v18_r5_val_scope.py",
    "scripts/sla/v18_r5_val_teacher.py",
    "scripts/wm02/v6_bootstrap.py",
    "src/go2arx5_vla/world_model/multitask_scope_v6.py",
)


def require_rejected(operation, reason):
    try:
        operation()
    except PermissionError as exc:
        if str(exc) == reason:
            return
        raise
    raise AssertionError("Cold check unexpectedly admitted: " + reason)



def check_lease_protocol(output):
    lock = output / "preflight-only.lock"
    lock.touch(mode=0o600, exist_ok=False)
    inode = lock.stat()
    now = time.time()
    receipt = output / "PREVENT_DISPATCH_TEST_LEASE.private.json"
    write(receipt, dict(kind="V18_R5_VAL_EXCLUSIVE_WORLD_LEASE",
                       uid=os.getuid(), owner_pid=os.getppid(), condition=SLOT_IDS[0], lock=str(lock),
                       lock_device=inode.st_dev, lock_inode=inode.st_ino,
                       deadline_epoch=now + 120, acquired_epoch=now))
    # The isolated lock is outside the canonical site world. Test only the
    # descriptor protocol, with no registered role, allocation or adapter.
    case = dict(site_world_lease=pin(receipt), panel_entry_id=SLOT_IDS[0], mobile_world_lock=str(lock),
                allocation_deadline_epoch=now + 120, deadline_epoch=now + 60)
    with lock.open("rb") as owned, lock.open("rb") as other:
        require_rejected(lambda: check_world_lease(case, owned.fileno()),
                         "R5_VAL_WORLD_IS_NOT_EXCLUSIVE")
        fcntl.flock(owned, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_world_lease(case, owned.fileno())
        require_rejected(lambda: check_world_lease(case, other.fileno()),
                         "R5_VAL_WORLD_FD_IS_NOT_OWNER")

def prepare(paths):
    source = Path(__file__).resolve().parents[2]
    output = paths.output.resolve()
    proposed_root = paths.future_root.resolve()
    output.mkdir(mode=0o700)
    if proposed_root.exists() or len(paths.source_commit) != 40:
        raise ValueError("Use a full source commit and a fresh proposed site root")

    evidence = {key: {"path": str(getattr(paths, key).resolve()), "sha256": sha}
                for key, sha in EVIDENCE.items()}
    legacy = read_pin(evidence["legacy_spec"])
    source_files = dict(legacy["source_files"])
    overlay = {name: digest(source / name) for name in OVERLAY}
    source_files.update(overlay)
    for name, expected in source_files.items():
        if digest(source / name) != expected:
            raise ValueError("R5_VAL_SOURCE_MEMBER_CHANGED:" + name)
    seal = hashlib.sha256(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    spec = copy.deepcopy(legacy)
    spec.update(evidence)
    spec.update(plan_commit=PLAN, goal_plan_sha=PLAN, scope=SCOPE, role=SCOPE, registry_scope=SCOPE,
                owner=OWNER, registered_root=str(proposed_root), mobile_group=str(proposed_root),
                campaign_root=str(proposed_root), collection_root=str(proposed_root),
                source_root=str(source), source_sha=paths.source_commit, source_files=source_files,
                source_seal=seal, sim_only_project_authorization=None,
                cases=[], registered_panel=None, resident_server_spec=None, sim_only_role="EXPERT",
                new_physical_requests_budget=len(SLOT_IDS), max_ticks=TICK_CAP,
                deadline_epoch=None, campaign_deadline_epoch=None, allocation_deadline_epoch=None)
    for key in ("tmp", "cache", "logs", "data"):
        spec[key] = str(proposed_root / key)
    env = spec["native_environment"]
    env.update(PYTHONPATH=env["PYTHONPATH"].replace(legacy["source_root"], str(source)),
               TMPDIR=spec["tmp"], XDG_CACHE_HOME=spec["cache"],
               ROS_LOG_DIR=str(proposed_root / "logs" / "ros"))
    population, rows, templates = plan(spec)
    panel = make_panel(population, rows, templates, panel_cards)
    panel_path = output / "PANEL.private.json"
    write(panel_path, panel)
    spec["registered_panel"] = pin(panel_path)
    require_unstarted_cases(spec["registry"], set(SLOT_IDS))
    require_rejected(lambda: validate_allocation(spec), "R5_VAL_SCIENTIFIC_ALLOCATION_REQUIRED")

    inherited = dict(spec, sim_only_project_authorization=legacy["sim_only_project_authorization"],
                     deadline_epoch=time.time() + 600)
    require_rejected(lambda: validate_allocation(inherited), "R5_VAL_SCOPED_SCIENTIFIC_ALLOCATION_REQUIRED")
    expected_scope = (SCOPE, PLAN, (), AUTHORITY)
    if scope_identity(spec) != expected_scope or scope_identity(legacy)[:2] != (legacy["scope"], legacy["plan_commit"]):
        raise AssertionError("The new scope and the archived scope must resolve independently")

    frozen = {}
    observed = []
    for row, item in zip(rows, panel, strict=True):
        _, case = prepare_case(spec, row, templates[row["id"]], item)
        validate_member(case, {"registered_root": str(proposed_root)})
        validate_frozen_assets(case)
        for path, expected in case["frozen_files"].items():
            if path not in frozen:
                frozen[path] = digest(path)
            if frozen[path] != expected:
                raise ValueError("R5_VAL_FROZEN_EXECUTOR_CHANGED:" + path)
        require_rejected(
            lambda: validate_member(dict(case, collection_split="TEST"),
                                    {"registered_root": str(proposed_root)}),
            "R5_VAL_TEACHER_ONLY",
        )
        require_rejected(
            lambda: validate_member(dict(case, sim_only_role="STUDENT"),
                                    {"registered_root": str(proposed_root)}),
            "R5_VAL_TEACHER_ONLY",
        )
        require_rejected(
            lambda: validate_member(dict(case, mobile_condition=dict(case["mobile_condition"], seed=0)),
                                    {"registered_root": str(proposed_root)}),
            "R5_VAL_FROZEN_TEACHER_AND_EXACT_SCENE_REQUIRED",
        )
        observed.append(dict(id=row["id"], max_ticks=case["max_ticks"],
                             seed=row["seed"], reset=row["reset"], sha256=item["scene"]["sha256"]))

    draft = output / "DRAFT_EXECUTION_SPEC.private.json"
    write(draft, spec)
    bootstrap = source / "scripts" / "wm02" / "v6_bootstrap.py"
    command = [spec["python"], "-B", str(bootstrap), "--r5-val-teacher", str(draft)]
    process = subprocess.run(command, cwd=source, env=dict(os.environ, **env),
                             capture_output=True, text=True, timeout=180, check=False)
    if process.returncode == 0 or "PermissionError: R5_VAL_SCIENTIFIC_ALLOCATION_REQUIRED" not in process.stderr:
        raise AssertionError("Unallocated production bootstrap did not fail closed: " + process.stderr)
    check_lease_protocol(output)
    if proposed_root.exists():
        raise AssertionError("A cold check cannot create a native run root")

    result = dict(status="COLD_VALIDATED_AWAITING_SCIENTIFIC_ALLOCATION", source_commit=paths.source_commit,
                  source_seal=seal, original_source_members=len(legacy["source_files"]),
                  source_members=len(source_files), overlay_sha256=overlay, frozen_files_checked=len(frozen),
                  inputs_sha256=EVIDENCE, panel_sha256=spec["registered_panel"]["sha256"], ordered_slots=observed,
                  registered_request_cap=len(SLOT_IDS), registered_tick_cap=TICK_CAP,
                  legacy_allocation_rejected=True, unallocated_bootstrap_rejected=True,
                  TEST_rejected=True, STUDENT_rejected=True, seed_mutation_rejected=True,
                  isolated_flock_protocol_verified=True,
                  capacity=storage_and_memory(proposed_root.parent),
                  site_allocation_present=False, site_world_leases=0, native_requests=0, integration_ticks=0,
                  native_dispatch_qualified=False, model_forwards=0, optimizer_updates=0)
    write(output / "COLD_RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in EVIDENCE:
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--future-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2))
