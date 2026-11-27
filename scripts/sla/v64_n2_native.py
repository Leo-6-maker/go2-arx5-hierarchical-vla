"""Run a fixed N2 short pair through the sealed native case runner."""

from contextlib import ExitStack, closing
from functools import partial
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.sla.v64_n2_roles import acquire_role_locks, check_descriptors, launch_pair, role_environment


PLAN = "4fe2c8cf1b69c168b2e3079e061bb7c702089cd3"
SCOPE = "SLA_CORE_MULTIENV_V1_N2_ENGINEERING"
AUTHORITY = "CURRENT_USER_SLA_CORE_MULTIENV_DATA_V1"
V64 = "531f826b36c1195db8c253ca71497f1aeb569c80"
ROSTER = "5896bbde4474aeccb11cf4653bbd48daef447a4eb1919f6ffd361b584df993e3"
ROLE_PROOF = "3f19f601a598898874888ded85086deecfc023de3e203b2d65bde68c893e9ed7"
PROFILE = "12135d6832dc6e833b52e15eb8ba196bed865b382480736d3294eb813cb94c06"
LIBRARY = "bf7bbeeb3760302c2d6b9300985f85d5bea140628c7e8e2f0ffa15cf2fe884f2"
FILESYSTEM_PROOF = "4b61bd7be960f26fec87b0afbec32a426a500eb50eef88317ab17a77931236b5"
PLANNER_PROOF = "45b904f757f2778573cd4a6656b4cc4c6ecbf8583fa5a41ae1584a620ae03cfd"
PLANNER_UST = "3dda3aff8ce04c7a1d5f1c5f20bd725d63e686a2042ee4834d8c25cde288f27b"
TICKS = 10703

TRANSPORT_ENV = (
    "ROS_DOMAIN_ID", "ROS_AUTOMATIC_DISCOVERY_RANGE", "ROS_STATIC_PEERS",
    "RMW_IMPLEMENTATION", "RMW_FASTRTPS_USE_QOS_FROM_XML", "FASTDDS_DEFAULT_PROFILES_FILE",
    "FASTRTPS_DEFAULT_PROFILES_FILE", "LTTNG_HOME", "LTTNG_UST_REGISTER_TIMEOUT", "SLA_MULTIWORLD_UST_REALM",
    "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME",
)
REMOVED_ENV = ("ROS_LOCALHOST_ONLY", "ROS_DISCOVERY_SERVER")


def binding(ref):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    from go2arx5_vla.world_model.storage import digest

    bound = read_pin(ref)
    checks = {"roster": ROSTER, "roles": ROLE_PROOF, "profile": PROFILE, "library": LIBRARY,
              "filesystem_proof": FILESYSTEM_PROOF, "planner_proof": PLANNER_PROOF, "planner_ust": PLANNER_UST}
    for key, expected in checks.items():
        evidence = bound[key]
        if evidence["sha256"] != expected or digest(evidence["path"]) != expected:
            raise PermissionError(f"N2 input differs from its verified receipt: {key}")
    roster = read_pin(bound["roster"])
    roles = read_pin(bound["roles"])
    planner = read_pin(bound["planner_proof"])
    pair = [row for row in roster["cases"] if row["pair"] == bound["pair"]]
    slots = {slot["role"]: slot for slot in roster["candidate_worlds"]}
    if (bound["v64"] != V64 or roles["native_requests"] or roles["new_integrations"] or roles["training_updates"]
            or planner["native_requests"] or planner["native_integrations"]
            or planner["library_sha256"] != PLANNER_UST
            or planner["groups"]["early"] != dict(starts=8, ready=8, clean_exits=8, private_pages=8, fastdds_shared=0)
            or len(pair) != 2 or len(slots) != 2 or {row["world_role"] for row in pair} != set(slots)
            or {proof["role"] for proof in roles["roles"]} != set(slots)
            or any(row["slot"]["max_native_ticks"] != TICKS or row["slot"]["max_native_requests"] != 1
                   for row in pair)
            or any(proof["domain"] != slots[proof["role"]]["ROS_DOMAIN_ID"]
                   or proof["cpus"] != slots[proof["role"]]["cpus"]
                   or not proof["checked_swapped_descriptors"] or proof["received_actual_frames"] != 16
                   for proof in roles["roles"])):
        raise PermissionError("N2 requires a complete fixed pair and both live role proofs")
    return bound, roster, pair, slots


def short_card(row):
    return dict(row, recording=row["archived_comparator"])


def member(spec):
    bound, _, pair, slots = binding(spec["v64_n2"])
    row, = (row for row in pair if row["slot"]["id"] == spec["panel_entry_id"])
    return bound, row, slots[row["world_role"]]


def transport_environment(spec, environment):
    result = dict(environment)
    for key in TRANSPORT_ENV:
        result[key] = spec["native_environment"][key]
    for key in REMOVED_ENV:
        result.pop(key, None)
    return result


def role_root(spec, slot):
    return Path(spec["v64_n2"]["path"]).parent / "roles" / slot["role"]


def native_role_environment(spec, slot, folder, bound):
    paths = role_root(spec, slot)
    return role_environment(spec["native_environment"], slot, folder, bound["profile"]["path"],
                            paths / "ust", bound["library"]["path"]) | {
        "XDG_RUNTIME_DIR": str(paths / "runtime"),
        "XDG_CONFIG_HOME": str(paths / "config"),
    }


def validate_member(case, authority):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    from scripts.sla.core_hopper_expert import OWNER

    bound, row, slot = member(case)
    original = read_pin(bound["original_authority"])
    template = read_pin(row["reference"])
    folder = Path(case["registered_root"])
    env = case["native_environment"]
    expected_env = native_role_environment(case, slot, folder, bound)
    exact_fields = (
        "mobile_guard_qualification", "mobile_safety_binding", "frozen_files", "task_instruction_id",
        "multitask_geometry", "policy_from_world", "model_path", "d2_probe_motion", "carry_repair",
        "collection_contract", "language", "sim_only_role", "untouched_legacy_fixed_wait_s",
    )
    if (case["owner"] != OWNER or authority["owner"] != OWNER or case["registry_scope"] != SCOPE
            or authority["v64_n2"] != case["v64_n2"] or authority["native_request_cap"] != 2
            or authority["native_tick_cap"] != 2 * TICKS or authority["tolerance"] != original["tolerance"]
            or authority["goal_start"] != original["goal_start"]
            or authority["deadline_epoch"] > original["goal_start"] + 7 * 86400
            or case["mobile_world_lock"] != slot["world_lock"]
            or case["ROS_DOMAIN_ID"] != slot["ROS_DOMAIN_ID"] or case["cpus"] != slot["cpus"]
            or case["site_world_spec"]["fixture"] != row["reference"] or case["max_ticks"] != TICKS
            or case["mobile_condition"] != dict(template["mobile_condition"], collection_episode=row["slot"]["id"])
            or case["collection_split"] != "ENGINEERING" or case["expert_training_eligible"] is not False
            or case["resident_server_spec"] is not None or case.get("d2_isolated_live_control") is not None
            or not all(case.get(key) == template.get(key) for key in exact_fields)
            or any(env.get(key) != expected_env[key] for key in (*TRANSPORT_ENV, "LD_PRELOAD"))
            or any(key in env for key in REMOVED_ENV)
            or env.get("LIBGL_ALWAYS_SOFTWARE") != "1"
            or env.get("__EGL_VENDOR_LIBRARY_FILENAMES") != "/usr/share/glvnd/egl_vendor.d/50_mesa.json"):
        raise PermissionError("N2 must preserve the registered references, safety and matching camera backend")


def check_case_locks(spec, world_fd, renderer_fd):
    from scripts.sla.core_hopper_expert import check_world_lease

    _, _, slot = member(spec)
    check_world_lease(spec, world_fd)
    check_descriptors(slot, (world_fd, renderer_fd))


def controller_argv(spec, output, argv, env):
    from scripts.sla.v64_n1_observation import wrap_controller

    _, row, _ = member(spec)
    return wrap_controller(spec, output, argv, env, row["slot"]["id"], detach=True)


def planner_environment(spec, environment):
    bound, _, _ = member(spec)
    # Initialize the same UST library before Python loads the ROS extension.
    preload = f"{bound['library']['path']}:{bound['planner_ust']['path']}"
    return dict(environment, LD_PRELOAD=preload)


def check_paused_transport(adapter):
    from go2arx5_vla.world_model.target_sequential import owned_simulator_pid
    from scripts.pi0292.durable import write
    from scripts.sla.v64_n1_observation import observer_child

    spec = adapter.spec
    realm = Path(spec["native_environment"]["SLA_MULTIWORLD_UST_REALM"])
    pages = [realm / f"{kind}.wait" for kind in ("user", "global")]
    processes = {"driver": os.getpid()}
    for name, process in adapter.processes:
        if name == "sim":
            processes[name] = owned_simulator_pid(process.pid, spec["native_binary"])
        elif name == "controller":
            processes[name] = observer_child(process.pid, spec["controller_binary"])
        else:
            processes[name] = process.pid
    evidence = {}
    for name, pid in processes.items():
        path = Path("/proc") / str(pid)
        maps = (path / "maps").read_text()
        evidence[name] = dict(pid=pid, private_pages_mapped=[str(page) in maps for page in pages],
                              private_page_inodes=[page.stat().st_ino for page in pages],
                              fastdds_shared_memory_mapped="/dev/shm/fastrtps_" in maps)
    names = adapter.node.get_node_names()
    write(adapter.output / "N2_PAUSED_TRANSPORT.private.json",
          dict(processes=evidence, domain=spec["ROS_DOMAIN_ID"], renderer="MATCHED_SOFTWARE",
               known_nodes=names, native_control_time_s=adapter.control()["sim_time_s"]))
    rejected = [name for name, detail in evidence.items() if not all(detail["private_pages_mapped"])
                or detail["fastdds_shared_memory_mapped"]]
    if rejected or adapter.node.get_name() not in names:
        raise PermissionError(f"N2 paused DDS or UST transport rejected: {rejected}")


def prepare_case(spec, row, slot, bound, lease, deadline):
    from scripts.sla.core_multienv_n1 import make_case

    role = dict(spec, mobile_world_lock=slot["world_lock"], ROS_DOMAIN_ID=slot["ROS_DOMAIN_ID"], cpus=slot["cpus"])
    folder, case = make_case(role, short_card(row), lease, deadline)
    case.update(scope=SCOPE, registry_scope=SCOPE, v64_n2=spec["v64_n2"])
    case["native_environment"] = native_role_environment(case, slot, folder, bound)
    return folder, case


def case_dirs(folder, runtime_root):
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs

    dirs(folder)
    runtime_root.mkdir(mode=0o700, parents=True)
    for name in ("config", "runtime", "ust"):
        child = runtime_root / name
        child.mkdir(mode=0o700)
        stat = child.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise PermissionError("N2 private runtime requires a filesystem honoring owner and mode")
    for name in ("user.wait", "global.wait"):
        fd = os.open(runtime_root / "ust" / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.ftruncate(fd, os.sysconf("SC_PAGE_SIZE"))
        finally:
            os.close(fd)


def require_capacity(root, spec):
    from scripts.sla.v64_n1_observation import require_writable

    capacity = require_writable(root)
    minimum = spec["multienv"]["estimates"]["floor_bytes"]
    reserve = 2 * spec["combined_inflight_reserve_bytes"]
    if capacity["free_bytes"] < minimum + reserve:
        raise OSError("The original storage floor plus both in-flight reserves is unavailable")
    return dict(capacity, production_floor_bytes=minimum, pair_reserve_bytes=reserve)


def run_member(spec, row, folder, case, descriptors):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    from scripts.pi0292.durable import read, write
    from scripts.pi0294.e1.journal import Journal
    from scripts.sla.core_multienv_n1 import measurement
    from scripts.sla.v64_n1_observation import shutdown_receipt
    from scripts.wm02.multitask_stage_cards_v6 import pin
    from scripts.wm02.sim_only_campaign_v51 import compact_case, fatal_control_violation

    root = Path(spec["registered_root"])
    identifier = row["slot"]["id"]
    path = folder / "EXECUTION_SPEC.json"
    args = ["taskset", "-c", ",".join(map(str, case["cpus"])), case["python"], "-B",
            str(Path(spec["source_root"]) / "scripts/wm02/v6_bootstrap.py"), "--case", str(path),
            "--world-lock-fd", str(descriptors[0]), "--renderer-lock-fd", str(descriptors[1])]

    def perform():
        env = dict(os.environ, **case["native_environment"])
        for key in REMOVED_ENV:
            env.pop(key, None)
        before = time.monotonic()
        with (folder / "logs/case.log").open("xb") as stream:
            child = subprocess.Popen(args, cwd=spec["source_root"], env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, pass_fds=descriptors)
            while child.poll() is None:
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    stop = folder / "STOP.json"
                    if not stop.exists() and shutil.disk_usage(folder).free < 2 * 1024**3:
                        write(stop, dict(reason="ACTUAL_LOCAL_TERMINAL_RESERVE", epoch=time.time()))
        terminal = folder / "RESULT.private.json"
        return dict(exit_code=child.returncode, pid=child.pid, start_monotonic=before,
                    end_monotonic=time.monotonic(), terminal=pin(terminal) if terminal.is_file() else None)

    receipt = Journal(root / "control/journal", root.name).once(identifier, dict(spec=pin(path)), perform)
    result = dict(id=identifier, native=receipt)
    if receipt["terminal"] is None:
        return dict(result, status="TERMINAL_UNKNOWN_RECONCILE")
    terminal = read_pin(receipt["terminal"])
    outcome = compact_case(case, terminal)
    result.update(physical_requests=terminal["physical_requests"], ticks=terminal["actual_native_integrations"],
                  outcome=outcome, wall_s=terminal["wall_s"])
    if (read(folder / "control/LEASE.json")["status"] != "CLOSED" or fatal_control_violation(outcome)
            or terminal["unsafe"] or terminal["technical_failure"] or not terminal["shutdown_clean"]
            or terminal["actual_native_integrations"] != TICKS or receipt["exit_code"] != 0):
        return dict(result, status="NATIVE_CONTROL_OR_TEARDOWN_FAILED")
    try:
        result["observer"] = shutdown_receipt(folder, detached=True)
        result["paused_transport"] = pin(folder / "native-01/N2_PAUSED_TRANSPORT.private.json")
        result["comparison"] = measurement(spec, short_card(row), case, terminal, receipt["terminal"])
    except (OSError, RuntimeError, ValueError) as exc:
        return dict(result, status="NATIVE_EVIDENCE_FAILED", failure=str(exc))
    return dict(result, status=result["comparison"]["status"])


def check_student_turn(spec):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

    handoff = Path(read_pin(spec["student_handoff_owner"])["registered_root"])
    if (handoff / "STUDENT_TURN_READY.private.json").exists():
        raise PermissionError("The original student fair-turn marker is set")


def run(spec):
    from go2arx5_vla.world_model.mfree_recovery_v2 import register
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    from go2arx5_vla.world_model.multitask_scope_v6 import validate
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.wm02.multitask_stage_cards_v6 import pin, require_unstarted_cases

    root = Path(spec["registered_root"])
    bound, roster, pair, slots = binding(spec["v64_n2"])
    result = dict(pair=bound["pair"], records={}, model_generations=0, training_admissions=0)
    code = 1
    try:
        with ExitStack() as stack:
            stack.enter_context(closing(OwnerLock(roster["original_global_lock"])))
            check_student_turn(spec)
            require_unstarted_cases(spec["registry"], {row["slot"]["id"] for row in pair})
            result["capacity"] = require_capacity(root, spec)
            owners = acquire_role_locks(stack, slots.values())
            now = time.time()
            end = min(spec["deadline_epoch"], now + 900)
            if end - now < 420:
                raise TimeoutError("The inherited finite native window cannot cover the matched pair")
            write(root / "SUPERVISOR.private.json", dict(pid=os.getpid(), epoch=now, source_sha=spec["source_sha"]))
            entries = []
            launches = []
            for row in pair:
                label = row["world_role"]
                slot = slots[label]
                identity = os.fstat(owners[label][0])
                lease = root / "control" / (label + ".json")
                write(lease, dict(kind="CORELAB_LIVE_SINGLE_WORLD_LEASE", owner_pid=os.getpid(), uid=os.getuid(),
                                  acquired_epoch=now, deadline_epoch=end, lock=slot["world_lock"],
                                  lock_inode=identity.st_ino, lock_device=identity.st_dev, condition=row["slot"]["id"]))
                folder, case = prepare_case(spec, row, slot, bound, pin(lease), end)
                validate(case)
                case_dirs(folder, role_root(case, slot))
                entries.append((folder, case))
                launches.append((label, partial(run_member, spec, row, folder, case)))
            register(spec, entries)
            result["records"] = launch_pair(launches, owners)
            receipt = [row["native"] for row in result["records"].values()]
            result["elapsed_s"] = max(row["end_monotonic"] for row in receipt) - min(
                row["start_monotonic"] for row in receipt)
            result["overlap_s"] = max(0, min(row["end_monotonic"] for row in receipt) - max(
                row["start_monotonic"] for row in receipt))
            code = int(any(row["status"] != "WITHIN_FROZEN_STATE_TOLERANCE"
                           for row in result["records"].values()))
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        write(root / "RESULT.private.json", dict(result, exit_code=code))
        close(spec, code)
    return code
