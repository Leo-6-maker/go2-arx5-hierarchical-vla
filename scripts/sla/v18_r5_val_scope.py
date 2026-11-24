"""Frozen native Teacher qualification for the four independently held-out VAL slots."""

import fcntl
import os
from pathlib import Path
import time

from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

SCOPE = "SLA_GO2_V18_R5_VAL_TEACHER"
PLAN = "c94c6319712b7baa101da1960892efee4e0d94eb"
AUTHORITY = "V18_R5_VAL_TEACHER_EXPLICIT_ALLOCATION"
OWNER = "SLA_V18_R5_VAL_TEACHER_UID2009"
LEGACY_SPEC = "89f058d5cd3a69cae8f68eb6a63f63a040f20151d07daf0d88e6a0ec5c3f8ffe"
SLOTS = (
    ("V17-VAL-F1-HOLD-red", "CHV1-TRAIN-F1-HOLD-red-00", 35703,
     "a77bb4a1b66b9a8527d98cc0ce7b19e3a2ae383a71c9a4c9dcf117087175901c"),
    ("V17-VAL-F1-HOLD-blue", "CHV1-TRAIN-F1-HOLD-blue-00", 35703,
     "ac394c7ab6a94b55d000782f9f666be378824350d7cbb7d6a7f17a304af8d68e"),
    ("V17-VAL-F1-PLACE-red-right", "CHV1-TRAIN-F1-PLACE-red-right-00", 50703,
     "672fc40baeff9248c6b07f3f23c7e21c7e279574a16e7157c62f3d25290f9f97"),
    ("V17-VAL-F2-HOLD-blue", "CHV1-TRAIN-F2-HOLD-blue-00", 35703,
     "cfd5ba0c42ebc6b8bd000b0b2ad655e8bd2b440acc37af1d14430bd62ed06707"),
)
SLOT_IDS = tuple(slot for slot, _, _, _ in SLOTS)
TICK_CAP = sum(cap for _, _, cap, _ in SLOTS)
EVIDENCE = {
    "compiled_population": "e88c3aac0e54e8c81e8645734bb0e4672e42ba4f8152ece9f078b4c33ad4300b",
    "candidate_roster": "8ac642e4fa69bb73a9fc427a3c56ce895729e17a299cf77bc37fa500436b741e",
    "guard_preflight": "59190ba5d2ce9233431afc100ebf0e0bf537234e5c82c5452be88c933d023a7f",
    "legacy_spec": LEGACY_SPEC,
}
FROZEN_FIELDS = (
    "accepted_boundary_version", "compiled_observer_result", "compiled_outer_result",
    "controller_binary", "execution_clock_profile", "global_task_budget_s",
    "grasp_goal_raise_m", "mobile_guard_qualification", "mobile_interface_qualification",
    "mobile_safety_binding", "mobile_safety_library", "mobile_safety_library_sha256",
    "mobile_stage", "mobile_wbc_boundary_v1", "native_binary", "observer_library",
    "outer_library", "phase", "policy_from_world", "python", "solver_binary",
    "stage_budget_s", "startup_pause_stride_ticks", "student_goal_offset_m",
    "trace_library", "xvfb_run",
)
SITE_FIELDS = ("cpus", "host", "mobile_world_lock", "registry", "ROS_DOMAIN_ID", "uid", "world_control")


def plan(spec):
    """Read preflight inputs and verify membership without constructing a native adapter."""
    if spec["owner"] != OWNER or spec["registry_scope"] != SCOPE:
        raise PermissionError("R5_VAL_OWNER_SCOPE_REQUIRED")
    for key, sha in EVIDENCE.items():
        if spec[key]["sha256"] != sha:
            raise PermissionError("R5_VAL_PINNED_EVIDENCE_REQUIRED:" + key)

    legacy = read_pin(spec["legacy_spec"])
    population = read_pin(spec["compiled_population"])
    roster = read_pin(spec["candidate_roster"])
    preflight = read_pin(spec["guard_preflight"])
    if (any(spec[key] != legacy[key] for key in SITE_FIELDS)
            or Path(spec["source_root"]).resolve() == Path(legacy["source_root"]).resolve()
            or spec["trusted_source_parent"] != legacy["trusted_source_parent"]):
        raise PermissionError("R5_VAL_NATIVE_SITE_AND_INDEPENDENT_SOURCE_REQUIRED")
    if (tuple(row["slot"] for row in roster["rows"]) != SLOT_IDS
            or len(population["conditions"]["VAL"]) != 8
            or len(population["conditions"]["TEST"]) != 12
            or preflight["status"] != "REFERENCE_EFFORT_AND_CONTACT_CLASSIFIER_REPLAY_PASS"
            or tuple(row["slot"] for row in preflight["slots"]) != SLOT_IDS):
        raise PermissionError("R5_VAL_FIXED_POPULATION_AND_CPU_GUARD_EVIDENCE_REQUIRED")

    clusters = {row["id"]: row for row in population["clusters"]}
    val = {row["id"]: row for row in population["conditions"]["VAL"]}
    templates, rows = {}, []
    for (slot, archive, cap, template_hash), entry, guard in zip(SLOTS, roster["rows"], preflight["slots"]):
        row = val[slot]
        cluster = clusters[row["cluster"]]
        template_path = Path(legacy["registered_root"]) / "cases" / archive / "EXECUTION_SPEC.json"
        template = read_pin({"path": str(template_path), "sha256": template_hash})
        if (entry["split"] != "VAL" or cluster["split"] != "VAL"
                or entry["historical_positive"]["id"] != archive
                or entry["prospective_max_ticks"] != cap
                or not all(row[key] == entry[key] for key in ("seed", "reset", "family"))
                or row["instruction"] != entry["task"]
                or cluster["scene"]["sha256"] != entry["geometry_sha256"]
                or guard["scene_sha256"] != entry["geometry_sha256"]
                or guard["guard_sha256"] != template["mobile_safety_library_sha256"]
                or guard["matched"] is not True or guard["integration_ticks"] != 0 or guard["world_starts"] != 0):
            raise PermissionError("R5_VAL_CONDITION_OR_GUARD_BINDING:" + slot)

        templates[slot] = template
        rows.append(dict(row, noise_seed=row["seed"], split="VAL", native_max_ticks=cap,
                         condition_id=slot, scene_family=row["cluster"], scene_seed=row["seed"]))
    return population, rows, templates


def validate_allocation(spec):
    """Require a separate allocation recorded by the site authority."""
    authorization = spec.get("sim_only_project_authorization")
    if not authorization:
        raise PermissionError("R5_VAL_SCIENTIFIC_ALLOCATION_REQUIRED")
    authority = read_pin(authorization)
    if (authority.get("source") != AUTHORITY or authority.get("status") != "APPROVED"
            or authority.get("scope") != SCOPE or authority.get("plan_commit") != PLAN
            or authority.get("source_sha") != spec["source_sha"]
            or authority.get("source_seal") != spec["source_seal"]
            or authority.get("registered_root") != spec["campaign_root"]
            or authority.get("ordered_slots") != list(SLOT_IDS)
            or authority.get("evidence_sha256") != EVIDENCE
            or authority.get("registered_panel_sha256") != spec["registered_panel"]["sha256"]
            or authority.get("native_request_cap") != len(SLOT_IDS)
            or authority.get("native_tick_cap") != TICK_CAP
            or authority.get("hardware_allowed") is not False
            or authority.get("model_forwards") != 0 or authority.get("optimizer_updates") != 0
            or authority.get("automatic_retry") is not False
            or not time.time() < spec["deadline_epoch"] <= authority["deadline_epoch"]):
        raise PermissionError("R5_VAL_SCOPED_SCIENTIFIC_ALLOCATION_REQUIRED")
    return authority


def validate_member(case, authority):
    from scripts.wm02.multitask_stage_cards_v6 import panel_cards

    if (case["scope"] != SCOPE or case["plan_commit"] != PLAN
            or case["collection_split"] != "VAL" or case["sim_only_role"] != "EXPERT"
            or case["expert_training_eligible"] is not False or case["resident_server_spec"] is not None
            or case["automatic_retry"] is not False or case["model_loads"] != 0
            or case["real_hardware"] is not False or case["wall_limit_s"] != 1800
            or case["mobile_group"] != authority["registered_root"]):
        raise PermissionError("R5_VAL_TEACHER_ONLY")

    population, rows, templates = plan(case)
    slot = case["panel_entry_id"]
    panel = make_panel(population, rows, templates, panel_cards)
    item = next((item for item in panel if item["id"] == slot), None)
    if item is None or read_pin(case["registered_panel"]) != panel:
        raise PermissionError("R5_VAL_EXACT_ORDERED_PANEL_REQUIRED")

    template = templates[slot]
    old_scene = template["model_path"]
    expected_files = {k: v for k, v in template["frozen_files"].items() if k != old_scene}
    expected_files[item["scene"]["path"]] = item["scene"]["sha256"]
    if (any(case[key] != template[key] for key in FROZEN_FIELDS)
            or case["frozen_files"] != expected_files
            or case["mobile_condition"] != item["condition"]
            or case["task_instruction_id"] != item["instruction"]
            or case["multitask_geometry"] != item["geometry"]
            or case["model_path"] != item["scene"]["path"]
            or Path(case["registered_root"]) != Path(case["campaign_root"]) / "cases" / slot):
        raise PermissionError("R5_VAL_FROZEN_TEACHER_AND_EXACT_SCENE_REQUIRED")


def make_panel(population, rows, templates, panel_cards):
    # Use the original card constructor. Use the matching historical cap/profile
    # for each task even when another task belongs to the same family.
    family_templates = {row["family"]: templates[row["id"]] for row in rows}
    panel = []
    for row in rows:
        by_family = dict(family_templates, **{row["family"]: templates[row["id"]]})
        panel.extend(panel_cards(population, [(row, "H0")], by_family, role="EXPERT"))
    return panel


def check_world_lease(case, fd):
    lease = read_pin(case["site_world_lease"])
    held = os.fstat(fd)
    canonical = os.stat(case["mobile_world_lock"])
    if (lease["kind"] != "V18_R5_VAL_EXCLUSIVE_WORLD_LEASE" or lease["uid"] != os.getuid()
            or lease["owner_pid"] != os.getppid() or lease["condition"] != case["panel_entry_id"]
            or lease["lock"] != case["mobile_world_lock"]
            or (held.st_dev, held.st_ino) != (canonical.st_dev, canonical.st_ino)
            or (held.st_dev, held.st_ino) != (lease["lock_device"], lease["lock_inode"])
            or lease["deadline_epoch"] != case["allocation_deadline_epoch"]
            or not lease["acquired_epoch"] <= time.time() < case["deadline_epoch"] <= lease["deadline_epoch"]
            or lease["deadline_epoch"] - lease["acquired_epoch"] > 2160):
        raise PermissionError("R5_VAL_LIVE_FINITE_INHERITED_WORLD_LEASE_REQUIRED")
    # A separate open must observe a lock already held. The inherited
    # descriptor must be able to reapply that same exclusive lock.
    with Path(case["mobile_world_lock"]).open("rb") as other:
        try:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise PermissionError("R5_VAL_WORLD_IS_NOT_EXCLUSIVE")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise PermissionError("R5_VAL_WORLD_FD_IS_NOT_OWNER") from exc
