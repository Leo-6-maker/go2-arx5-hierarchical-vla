"""Run at most the four held-out Teacher cards under a separately allocated scope."""

import json
import os
import shutil
from pathlib import Path
import subprocess
import time
import traceback

from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close, dirs
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.core_hopper_collection import reason
from scripts.sla.v18_r5_val_scope import (
    EVIDENCE, PLAN, SCOPE, SLOT_IDS, TICK_CAP, make_panel, plan, validate_allocation,
)
from scripts.wm02.multitask_stage_cards_v6 import case_card, panel_cards, pin, require_unstarted_cases
from scripts.wm02.multitask_teacher_campaign_v6 import teacher_row
from scripts.wm02.sim_only_campaign_v51 import fatal_control_violation

CASE_RESERVATION_S = 2160
CASE_CLOSE_MARGIN_S = 60
MIN_AVAILABLE_RAM = 13 * 1024**3
MIN_FREE_DISK = (50 + 8) * 1024**3


def storage_and_memory(path):
    # Four historic captures plus normalizer export peaked near 1 GiB each.
    # Keep 8 GiB for this panel, on top of the historical 50 GiB reserve.
    disk = shutil.disk_usage(path)
    available = next(line for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    ram = int(available.split()[1]) * 1024
    return dict(disk_free_bytes=disk.free, ram_available_bytes=ram,
                disk_min_bytes=MIN_FREE_DISK, ram_min_bytes=MIN_AVAILABLE_RAM,
                passed=disk.free >= MIN_FREE_DISK and ram >= MIN_AVAILABLE_RAM)


def prepare_case(spec, row, template, item):
    folder, case = case_card(spec, template, item, spec["registered_panel"])
    case["frozen_files"].pop(template["model_path"])
    case.update({key: spec[key] for key in EVIDENCE})
    case.update(registry_scope=SCOPE, collection_split="VAL", expert_training_eligible=False,
                resident_server_spec=None)
    return folder, case


def dispatch(spec, row, template, item, world, acquired):
    root = Path(spec["registered_root"])
    folder, case = prepare_case(spec, row, template, item)
    end = min(spec["deadline_epoch"], acquired + CASE_RESERVATION_S)
    if end - time.time() < 1980 or (root / "STOP.json").exists():
        raise RuntimeError("R5_VAL_CASE_WINDOW_CLOSED_BEFORE_REGISTRATION")
    fd = world.handle.fileno()
    inode = os.fstat(fd)
    lease_path = root / "control" / ("world-" + row["id"] + ".private.json")
    write(lease_path, dict(kind="V18_R5_VAL_EXCLUSIVE_WORLD_LEASE",
          owner_pid=os.getpid(), uid=os.getuid(), condition=row["id"], acquired_epoch=acquired,
          deadline_epoch=end, lock=spec["mobile_world_lock"], lock_device=inode.st_dev,
          lock_inode=inode.st_ino))
    case.update(site_world_lease=pin(lease_path), allocation_deadline_epoch=end,
                deadline_epoch=end - CASE_CLOSE_MARGIN_S)
    validate(case)
    dirs(folder)
    register(spec, [(folder, case)])
    path = folder / "EXECUTION_SPEC.json"
    source = Path(spec["source_root"])
    command = [case["python"], "-B", str(source / "scripts/wm02/v6_bootstrap.py"),
               "--case", str(path), "--world-lock-fd", str(fd)]
    env = dict(os.environ, **case["native_environment"])
    with (folder / "logs" / "case.log").open("xb") as stream:
        child = subprocess.Popen(command, cwd=source, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, pass_fds=(fd,))
        try:
            write(root / "CURRENT_CASE.private.json",
                  dict(id=row["id"], pid=child.pid, spec=pin(path), epoch=time.time()))
        finally:
            child.wait()

    terminal = folder / "RESULT.private.json"
    if not terminal.is_file():
        raise RuntimeError("R5_VAL_TERMINAL_REQUIRES_RECONCILIATION:" + row["id"])
    return dict(spec=pin(path), result=pin(terminal), exit=child.returncode)


def run(spec):
    root = Path(spec["registered_root"])
    owner = OwnerLock(root / "control" / "producer.lock")
    try:
        if (root / "SUPERVISOR.private.json").exists() or (root / "RESULT.private.json").exists():
            raise PermissionError("R5_VAL_EXISTING_BATCH_REQUIRES_RECONCILIATION")
        return run_owned(spec, root)
    finally:
        owner.close()


def run_owned(spec, root):
    code, failure = 1, None
    current, records = None, []
    try:
        authority = validate_allocation(spec)
        legacy = read_pin(spec["legacy_spec"])
        if ((spec["scope"], spec["plan_commit"], spec["role"]) != (SCOPE, PLAN, SCOPE)
                or spec["campaign_root"] != str(root) or spec["mobile_group"] != str(root)
                or spec["physics"] is not True or spec["real_hardware"] is not False
                or spec["new_physical_requests_budget"] != len(SLOT_IDS) or spec["max_ticks"] != TICK_CAP
                or spec["new_model_generations_budget"] != 0 or spec["new_optimizer_updates_budget"] != 0
                or spec["new_training_visits_budget"] != 0 or spec["resident_server_spec"] is not None
                or spec["automatic_retry"] is not False
                or spec["admission_python"] != legacy["admission_python"]
                or spec["C20_statistics_path"] != legacy["C20_statistics_path"]):
            raise PermissionError("R5_VAL_FINITE_TEACHER_CAMPAIGN_REQUIRED")

        population, rows, templates = plan(spec)
        panel = make_panel(population, rows, templates, panel_cards)
        if panel != read_pin(spec["registered_panel"]):
            raise PermissionError("R5_VAL_PARENT_PANEL_MISMATCH")
        require_unstarted_cases(spec["registry"], set(SLOT_IDS))
        journal = Journal(root / "control" / "journal", root.name)
        write(root / "SUPERVISOR.private.json",
              dict(pid=os.getpid(), source_sha=spec["source_sha"], epoch=time.time()))

        for row, item in zip(rows, panel, strict=True):
            slot = row["id"]
            if (root / "STOP.json").exists():
                raise RuntimeError("R5_VAL_SITE_STOP_REQUESTED")
            if time.time() + 1980 >= spec["deadline_epoch"] or not storage_and_memory(root)["passed"]:
                raise RuntimeError("R5_VAL_NO_FINITE_SITE_CAPACITY")
            # A fresh world lock for every case. The lock survives in the child
            # until the original executor has shut down and produced its result.
            world = OwnerLock(Path(spec["mobile_world_lock"]))
            current = slot
            try:
                acquired = time.time()
                receipt = journal.once(slot, dict(slot=slot, allocation=spec["sim_only_project_authorization"],
                                                 panel=spec["registered_panel"], source=spec["source_sha"]),
                                       lambda: dispatch(spec, row, templates[slot], item, world, acquired))
            finally:
                world.close()

            case, native = read_pin(receipt["spec"]), read_pin(receipt["result"])
            lease = read(Path(case["registered_root"]) / "control" / "LEASE.json")
            if (lease["status"] != "CLOSED" or lease["spec_sha256"] != receipt["spec"]["sha256"]
                    or native["source_sha"] != spec["source_sha"]):
                raise RuntimeError("R5_VAL_TERMINAL_IDENTITY_OR_CLOSURE_UNRECONCILED:" + slot)
            outcome = teacher_row(case, native)
            record = dict(id=slot, split="VAL", result=receipt["result"], requests=native["physical_requests"],
                          ticks=native["actual_native_integrations"], shutdown_clean=native["shutdown_clean"],
                          hard_STOP=outcome["hard_STOP"], post_STOP_ticks=outcome["integrations_after_STOP"],
                          technical_failure=outcome["technical_failure"], first_failure=reason(outcome, native),
                          task_and_mobile_success=outcome["task_and_mobile_success"], score=outcome["score"],
                          admitted=False)
            records.append(record)
            write(root / "RECORDS.private.json", records)
            current = None
            if (record["requests"] != 1 or not 0 <= record["ticks"] <= row["native_max_ticks"]
                    or sum(r["ticks"] for r in records) > authority["native_tick_cap"]
                    or native["shutdown_clean"] is not True
                    or (record["hard_STOP"] and record["post_STOP_ticks"] != 0)
                    or fatal_control_violation(outcome) or outcome["technical_failure"] or receipt["exit"]):
                raise RuntimeError("R5_VAL_CONTROL_OR_TECHNICAL_TERMINAL:" + slot)

            if outcome["completed"] and outcome["task_and_mobile_success"] and not outcome["hard_STOP"]:
                request = Path(case["registered_root"]) / "ADMISSION_INPUT.private.json"
                write(request, dict(case=receipt["spec"], native=receipt["result"], condition=row,
                                    statistics=spec["C20_statistics_path"]))
                result = subprocess.check_output(
                    [spec["admission_python"], "-B", "-m", "scripts.sla.core_hopper_admission",
                     str(request), digest(request)], cwd=spec["source_root"],
                    env=dict(os.environ, **case["native_environment"]), text=True,
                    timeout=max(1.0, spec["deadline_epoch"] - time.time()),
                )
                verdict = json.loads(result)
                if verdict["admitted"]:
                    if verdict["episode"]["split"] != "VAL":
                        raise RuntimeError("R5_VAL_ADMISSION_SPLIT_MISMATCH:" + slot)
                    record.update(admitted=True, episode=verdict["episode"], first_failure=None)
                else:
                    record["first_failure"] = "ADMISSION: " + verdict["first_failure"]
                write(root / "RECORDS.private.json", records)
        code = 0
    except Exception:
        failure = traceback.format_exc()
    finally:
        paid = {row["id"] for row in records}
        write(root / "RESULT.private.json",
              dict(exit_code=code, failure=failure, unreconciled=current, records=records,
                   unstarted=[slot for slot in SLOT_IDS if slot not in paid and slot != current],
                   verified_requests=sum(row["requests"] for row in records),
                   verified_ticks=sum(row["ticks"] for row in records),
                   admitted_VAL=sum(row["admitted"] for row in records),
                   allocated_requests=len(SLOT_IDS), allocated_ticks=TICK_CAP,
                   source_sha=spec["source_sha"]))
        close(spec, code)
    return code
