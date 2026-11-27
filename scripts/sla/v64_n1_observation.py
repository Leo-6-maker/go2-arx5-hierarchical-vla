"""Cold dispatch of the V6.4 reference observer in reserved PR425 interface slots."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time


SLOT = "MEV1-ENG-INTERFACE-00"
CLOSE_SLOT = "MEV1-ENG-INTERFACE-01"
DETACH_SLOT = "MEV1-ENG-INTERFACE-02"
FIRST_OBSERVER = "40af8145262bdbe3da44952b9ac69230d99d7bf2f186886c480b55cf28dbc22d"
FIRST_OBSERVER_NATIVE = "36467b6e43fb68e528e786c10e2660e461f514a0556ccda3e7f95dd8d8f16d18"
CLOSE_OBSERVER = "3bfe53258c1f7d9e86999175f7d0a8326ace11198b20a6a616bb458f5b084f0c"
CLOSE_OBSERVER_NATIVE = "0f57dc524954f9c1b47dc2db8640e0ed70e1dee8b28e9e7ef658764260b6f2bc"
CLOSE_DISARM = "93e2f4c82c7aa382b23573f2b89b18bcfc17c882df7951993a152914ce3fc7b4"
BASELINE = "MEV1-ENG-N1-SHORT-01"
V64 = "531f826b36c1195db8c253ca71497f1aeb569c80"
LIMIT = 8192
TICKS = 10703
RESERVE = 3 * 1024**3
FROZEN = "9e8ce46837d09bc39ae04d4b49f1ce4dd8c8c2d31bea39eac56c5806d108c1f3"



def closed_teardown(prior, slot, result_sha, native_sha):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

    if prior["sha256"] != result_sha:
        raise PermissionError("The followup must point to the expected closed observer")
    first = read_pin(prior)
    native = read_pin(first["native"]["terminal"])
    root = Path(prior["path"]).parent
    identifiers = [row["name"] for row in native["owned_process_exits"] if row["exit_code"] != 0]
    folder = root / "cases" / slot
    if (first["condition"] != slot or first["physical_requests"] != 1 or first["ticks"] != TICKS
            or first["native"]["terminal"]["sha256"] != native_sha
            or first["comparison"]["status"] != "WITHIN_FROZEN_STATE_TOLERANCE"
            or native["unsafe"] or native["technical_failure"] or native["shutdown_clean"]
            or identifiers != ["controller"]
            or any(json.loads(path.read_bytes())["status"] != "CLOSED" for path in
                   (root / "control/LEASE.json", folder / "control/LEASE.json"))):
        raise PermissionError("A technically isolated observer teardown failure is required")
    return folder


def slot_id(binding):
    if "prior_observer" not in binding:
        if "prior_close" in binding:
            raise PermissionError("Both original observer attempts must remain bound")
        return SLOT
    closed_teardown(binding["prior_observer"], SLOT, FIRST_OBSERVER, FIRST_OBSERVER_NATIVE)
    if "prior_close" in binding:
        folder = closed_teardown(binding["prior_close"], CLOSE_SLOT, CLOSE_OBSERVER, CLOSE_OBSERVER_NATIVE)
        if shutdown_receipt(folder)["sha256"] != CLOSE_DISARM:
            raise PermissionError("Require the actual failed teardown after hardware disarm")
        return DETACH_SLOT
    return CLOSE_SLOT


def shutdown_receipt(folder, *, detached=False):
    from go2arx5_vla.world_model.storage import digest

    trace = folder / "native-01/SELECTED_CONTROLLER.private.jsonl"
    path = trace.with_name(trace.name + ".shutdown")
    if not path.is_file():
        raise RuntimeError("Observer signal disarm was not recorded")
    record = json.loads(path.read_bytes())
    with trace.open() as stream:
        header = json.loads(next(stream))
        last_read_ns = max(row["monotonic_ns"] for row in map(json.loads, stream))
    if not (record["inferior_pid"] == header["inferior_pid"] and record["hardware_breakpoint_removed"]
            and record["signal_pending"] and record["monotonic_ns"] > last_read_ns):
        raise RuntimeError("Actual observer disarm did not occur before the inferior exited")
    receipt = dict(path=str(path), sha256=digest(path), actual_signal_disarm=True)
    if detached:
        exit_path = trace.with_name(trace.name + ".detached-exit")
        exit_record = json.loads(exit_path.read_bytes())
        if (exit_record["inferior_pid"] != header["inferior_pid"] or exit_record["native_exit"] != 0
                or exit_record["gdb_exit"] != 0 or not exit_record["protocol_complete"]):
            raise RuntimeError("Require actual successful debugger and controller exits")
        receipt["detached_native_exit"] = dict(path=str(exit_path), sha256=digest(exit_path))
    return receipt


def fixed_card(authority):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    from scripts.sla.core_multienv_geometry import load_population
    from scripts.sla.core_multienv_n1 import fixed_cards

    bound = read_pin(authority["v64_observation"])
    old = read_pin(bound["original_authority"])
    original = fixed_cards(old)[BASELINE]
    slots = load_population(Path(old["logical"]["path"]))["engineering"]
    slot, = (row for row in slots if row["id"] == slot_id(bound))
    if (bound["plan"] != V64 or bound["fixture"] != original["reference"]
            or bound["comparison"] != BASELINE or bound["slot"] != slot
            or slot["parallel_worlds"] != 1 or slot["max_native_ticks"] != TICKS
            or not slot["max_native_requests"] == authority["native_request_cap"] == 1
            or authority["native_tick_cap"] != TICKS or authority["tolerance"] != old["tolerance"]
            or authority["goal_start"] != old["goal_start"] or authority["cpus"] != old["cpus"]
            or authority["world_lock"] != old["world_lock"]):
        raise PermissionError("Use the fixed single-world observer allocation and original N1 comparator")
    return dict(original, slot=slot)


def controller_argv(spec, output, argv, env):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

    identifier = slot_id(read_pin(spec["v64_observation"]))
    return wrap_controller(spec, output, argv, env, identifier, detach=identifier == DETACH_SLOT)


def wrap_controller(spec, output, argv, env, identifier, *, detach):
    from go2arx5_vla.world_model.storage import digest

    controller = Path(argv[0])
    if (controller != Path(spec["controller_binary"]) or digest(controller) != FROZEN
            or spec["panel_entry_id"] != identifier or spec["max_ticks"] != TICKS):
        raise PermissionError("Only the frozen controller in this observer card can be traced")
    trace = output / "SELECTED_CONTROLLER.private.jsonl"
    observer = Path(spec["source_root"]) / "scripts/sla/v63_selected_reference_gdb.py"
    env = dict(env, SLA_SELECTED_REFERENCE_TRACE=str(trace), SLA_SELECTED_REFERENCE_TRACE_LIMIT=str(LIMIT))
    env.pop("SLA_SELECTED_REFERENCE_FIXTURE", None)
    env.pop("SLA_SELECTED_REFERENCE_FIXTURE_SOLUTION", None)
    env.pop("SLA_SELECTED_REFERENCE_DETACH", None)
    gdb = ["/usr/bin/gdb", "--nx", "--batch", "-ex", "set startup-with-shell off",
           "-ex", "set disable-randomization off", "-ex", "handle SIGINT nostop noprint pass",
           "-ex", "handle SIGTERM nostop noprint pass", "-ex", "starti", "-x", str(observer), "-ex", "continue"]
    if detach:
        env["SLA_SELECTED_REFERENCE_DETACH"] = "1"
        supervisor = observer.with_name("v64_observer_detached_child.py")
        gdb[3:3] = ["-ex", "set confirm off"]
        return [spec["python"], "-B", str(supervisor), str(trace), *gdb, "-ex", "detach", "--args", *argv], env
    gdb.insert(3, "--return-child-result")
    return [*gdb, "--args", *argv], env


def direct_children(pid):
    task = Path("/proc") / str(pid) / "task" / str(pid)
    try:
        return (task / "children").read_text().split()
    except FileNotFoundError:
        return []


def observer_child(wrapper, binary):
    proc = Path("/proc")
    debugger = Path("/usr/bin/gdb").resolve()
    children = direct_children(wrapper)
    candidates = set(children)
    for raw in children:
        child = proc / raw
        try:
            if (child.stat().st_uid == os.getuid() and os.getsid(int(raw)) == wrapper
                    and (child / "exe").resolve(strict=True) == debugger):
                candidates.update(direct_children(raw))
        except (FileNotFoundError, ProcessLookupError):
            continue
    candidates.update(direct_children(wrapper))
    owned = []
    for raw in candidates:
        child = proc / raw
        try:
            if (child.stat().st_uid == os.getuid() and (child / "exe").resolve(strict=True) == Path(binary).resolve()
                    and os.getsid(int(raw)) == wrapper):
                owned.append(int(raw))
        except (FileNotFoundError, ProcessLookupError):
            continue
    if len(owned) != 1:
        raise RuntimeError("Expected one exact controller child in the observer session")
    return owned[0]


def signal_observed(child, wrapper, binary, sig):
    fd = os.pidfd_open(child)
    try:
        if observer_child(wrapper, binary) != child:
            raise RuntimeError("Observed controller PID changed")
        signal.pidfd_send_signal(fd, sig)
    finally:
        os.close(fd)


def require_writable(root):
    mem = next(int(row.split()[1])*1024 for row in Path("/proc/meminfo").read_text().splitlines()
               if row.startswith("MemAvailable:"))
    free = shutil.disk_usage(root).free
    if free <= RESERVE or mem < 13*1024**3:
        raise OSError("The measured local trace, terminal reserve or memory is unavailable")
    with tempfile.TemporaryFile(dir=root) as stream:
        os.posix_fallocate(stream.fileno(), 0, 4*1024**2)
        os.fsync(stream.fileno())
    return dict(free_bytes=free, reserve_bytes=RESERVE, memory_bytes=mem, device=root.stat().st_dev)


def run(spec):
    from go2arx5_vla.world_model.mfree_recovery_v2 import register
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close, dirs
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate
    from scripts.pi0292.durable import OwnerLock, read, write
    from scripts.pi0294.e1.journal import Journal
    from scripts.sla.core_multienv_n1 import make_case, measurement
    from scripts.wm02.multitask_stage_cards_v6 import pin, require_unstarted_cases
    from scripts.wm02.sim_only_campaign_v51 import compact_case, fatal_control_violation

    root = Path(spec["registered_root"])
    authority = read_pin(spec["sim_only_project_authorization"])
    row = fixed_card(authority)
    bound = read_pin(authority["v64_observation"])
    identifier = row["slot"]["id"]
    result = dict(condition=identifier, physics_started=False, training_admissions=0, model_generations=0)
    code = 1
    try:
        handoff = Path(read_pin(spec["student_handoff_owner"])["registered_root"])
        if (handoff / "STUDENT_TURN_READY.private.json").exists():
            raise PermissionError("The original student fair-turn marker is set")
        with closing(OwnerLock(spec["mobile_world_lock"])) as world:
            require_unstarted_cases(spec["registry"], {identifier})
            result["capacity"] = require_writable(root)
            written = time.time()
            end = min(spec["deadline_epoch"], written+650)
            if end-written < 300:
                raise TimeoutError("The finite observation lease cannot cover the fixed workload")
            lease = root / "control/world.json"
            inode = os.fstat(world.handle.fileno())
            write(lease, dict(kind="CORELAB_LIVE_SINGLE_WORLD_LEASE", owner_pid=os.getpid(), uid=os.getuid(),
                              acquired_epoch=written, deadline_epoch=end, lock=spec["mobile_world_lock"],
                              lock_inode=inode.st_ino, lock_device=inode.st_dev, condition=identifier))
            write(root/"SUPERVISOR.private.json", dict(pid=os.getpid(), epoch=written, source_sha=spec["source_sha"]))
            folder, case = make_case(spec, row, pin(lease), end)
            case["v64_observation"] = authority["v64_observation"]
            validate(case)
            dirs(folder)
            register(spec, [(folder, case)])
            path = folder/"EXECUTION_SPEC.json"
            command = [case["python"], "-B", str(Path(spec["source_root"])/"scripts/wm02/v6_bootstrap.py"),
                       "--case", str(path), "--world-lock-fd", str(world.handle.fileno())]

            def perform():
                with (folder/"logs/case.log").open("xb") as stream:
                    child = subprocess.Popen(command, cwd=spec["source_root"], env=dict(os.environ, **case["native_environment"]),
                        stdout=stream, stderr=subprocess.STDOUT, pass_fds=(world.handle.fileno(),))
                    while child.poll() is None:
                        try:
                            child.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            stop = folder/"STOP.json"
                            if not stop.exists() and shutil.disk_usage(folder).free < 1024**3:
                                write(stop, dict(reason="ACTUAL_LOCAL_TERMINAL_RESERVE", epoch=time.time()))
                return dict(exit_code=child.returncode, terminal=pin(folder/"RESULT.private.json"), spec=pin(path))

            receipt = Journal(root/"control/journal", root.name).once(identifier, dict(binding=authority["v64_observation"]), perform)
            terminal = read_pin(receipt["terminal"])
            outcome = compact_case(case, terminal)
            result.update(physical_requests=terminal["physical_requests"], ticks=terminal["actual_native_integrations"],
                          physics_started=terminal["physical_requests"] == 1, native=receipt, outcome=outcome)
            if (read(folder/"control/LEASE.json")["status"] != "CLOSED" or fatal_control_violation(outcome)
                    or terminal["technical_failure"] or terminal["actual_native_integrations"] > TICKS):
                raise RuntimeError("Original control or terminal gate failed")
            result["comparison"] = measurement(spec, row, case, terminal, receipt["terminal"])
            baseline = read_pin(bound["baseline_terminal"])
            baseline_wall = baseline["wall_s"]
            result.update(native_wall_s=terminal["wall_s"], baseline_wall_s=baseline_wall,
                          overhead_relative=terminal["wall_s"]/baseline_wall-1)
            if not terminal["shutdown_clean"]:
                raise RuntimeError("Native observer did not satisfy the original clean shutdown gate")
            if identifier in (CLOSE_SLOT, DETACH_SLOT):
                result["observer_shutdown"] = shutdown_receipt(folder, detached=identifier == DETACH_SLOT)
            code = 0
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
    finally:
        group = next(line.strip().split("0::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines()
                     if line.startswith("0::"))
        cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
        result["cgroup_cpu"] = (cgroup/"cpu.stat").read_text()
        result["cgroup_memory_peak_bytes"] = int((cgroup/"memory.peak").read_text())
        write(root/"RESULT.private.json", dict(result, exit_code=code))
        close(spec, code)
    return code


def patched(path, edits):
    content = path.read_text()
    for old, new in edits:
        if content.count(old) != 1:
            raise ValueError(f"Exact original source seam is unavailable in {path.name}")
        content = content.replace(old, new)
    compile(content, str(path), "exec")
    path.unlink()
    path.write_text(content)


def stage(previous, root, overlay, source_sha, *, execute, prior_observer=None, prior_close=None):
    from go2arx5_vla.world_model.mfree_recovery_v2 import launch_service, register
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.durable import OwnerLock, read, write
    from scripts.sla.core_multienv_geometry import load_population
    from scripts.sla.core_multienv_n1 import fixed_cards, make_case
    from scripts.wm02.multitask_stage_cards_v6 import pin, require_unstarted_cases

    old = read(previous/"EXECUTION_SPEC.json")
    original = read_pin(old["sim_only_project_authorization"])
    baseline_root = previous.with_name(previous.name.removesuffix("-02")+"-01")/"cases"/BASELINE
    baseline = read(baseline_root/"RESULT.private.json")
    if (root.exists() or overlay.exists() or old["host"] != os.uname().nodename or old["uid"] != os.getuid()
            or read(previous/"control/LEASE.json")["status"] != "CLOSED"
            or read(baseline_root/"control/LEASE.json")["status"] != "CLOSED"
            or baseline["actual_native_integrations"] != TICKS or baseline["unsafe"]
            or not baseline["detail"]["interface_protocol_pass"]):
        raise PermissionError("Require the closed unchanged N1 comparison before a fresh observation")
    prior = {} if prior_observer is None else {"prior_observer": pin(prior_observer)}
    if prior_close is not None:
        prior["prior_close"] = pin(prior_close)
    identifier = slot_id(prior)
    population = load_population(Path(original["logical"]["path"]))["engineering"]
    slot, = (row for row in population if row["id"] == identifier)
    fixture = fixed_cards(original)[BASELINE]
    baseline_record, = (row for row in read(previous/"RESULT.private.json")["records"] if row["id"] == BASELINE)
    if baseline_record["measurement"]["status"] != "WITHIN_FROZEN_STATE_TOLERANCE":
        raise PermissionError("The matched comparator must have passed its original tolerance")
    require_unstarted_cases(old["registry"], {identifier})
    source = Path(old["source_root"])
    members = dict(old["source_files"])
    if any(digest(source/name) != sha for name,sha in members.items()):
        raise PermissionError("Original runtime source or seal changed")
    capacity = require_writable(root.parent)
    overlay.mkdir(mode=0o700)
    target = overlay/"source"
    shutil.copytree(source, target, copy_function=os.link, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    sl = Path("scripts/sla")
    wm = Path("src/go2arx5_vla/world_model")
    edits = {
        sl/"core_multienv_n1.py": [
            ("    row = fixed_cards(authority)[case['panel_entry_id']]",
             "    if 'v64_observation' in authority:\n        from scripts.sla.v64_n1_observation import fixed_card\n"
             "        row = fixed_card(authority)\n"
             "        if case.get('v64_observation') != authority['v64_observation'] or case['panel_entry_id'] != row['slot']['id']:\n"
             "            raise PermissionError('Exact registered observer case required')\n"
             "    else:\n        row = fixed_cards(authority)[case['panel_entry_id']]"),
            ("def run(spec):\n    root", "def run(spec):\n    if spec.get('v64_observation'):\n"
             "        from scripts.sla.v64_n1_observation import run as observer_run\n        return observer_run(spec)\n    root"),
        ],
        wm/"target_sequential.py": [
            ("    def launch(self,argv,name,env):\n        h=",
             "    def launch(self,argv,name,env):\n        if name == 'controller' and self.spec.get('v64_observation'):\n"
             "            from scripts.sla.v64_n1_observation import controller_argv\n"
             "            argv, env = controller_argv(self.spec, self.output, argv, env)\n        h="),
            ("                if name=='sim':\n                    try:child=owned_simulator_pid",
             "                observed = name == 'controller' and self.spec.get('v64_observation')\n"
             "                if observed:\n                    from scripts.sla.v64_n1_observation import observer_child, signal_observed\n"
             "                    try:child=observer_child(p.pid,self.spec['controller_binary'])\n"
             "                    except (RuntimeError, OSError) as exc:lookup_error=str(exc)\n"
             "                if name=='sim':\n                    try:child=owned_simulator_pid"),
            ("                        if child is not None:\n                            try:current=owned_simulator_pid",
             "                        if observed and child is not None:\n"
             "                            signal_observed(child,p.pid,self.spec['controller_binary'],sig)\n"
             "                        elif child is not None:\n                            try:current=owned_simulator_pid"),
        ],
        wm/"multitask_probe_d2.py": [
            ("    cancel = adapter.cancel_reference() if expected_stop is None else None",
             "    if adapter.spec.get('v64_observation') and expected_stop is None:\n"
             "        expected_stop = project_stop(adapter, adapter.spec['max_ticks'])\n"
             "    cancel = adapter.cancel_reference() if expected_stop is None else None")
        ],
    }
    for relative, changes in edits.items():
        patched(target/relative, changes)
        members[str(relative)] = digest(target/relative)
    for name in ("v64_n1_observation.py", "v63_selected_reference_gdb.py", "v64_selected_policy_match.py",
                 "v64_observer_detached_child.py"):
        relative = sl/name
        if (target/relative).exists():
            raise FileExistsError(relative)
        shutil.copy2(Path(__file__).with_name(name), target/relative)
        members[str(relative)] = digest(target/relative)
    binding = overlay/"BINDING.private.json"
    write(binding, dict(plan=V64, slot=slot, comparison=BASELINE, baseline_terminal=pin(baseline_root/"RESULT.private.json"),
                       fixture=fixture["reference"], original_authority=old["sim_only_project_authorization"], **prior))
    end = min(time.time()+1100, original["goal_start"]+7*86400)
    auth = overlay/"AUTHORITY.private.json"
    write(auth, dict(original, deadline_epoch=end, source_sha=source_sha, native_request_cap=1, native_tick_cap=TICKS,
                     v64_observation=pin(binding)))
    panel = overlay/"PANEL.private.json"
    case = read(baseline_root/"EXECUTION_SPEC.json")
    member, = (row for row in read_pin(case["registered_panel"]) if row["id"] == BASELINE)
    write(panel, [dict(member, id=identifier, condition=dict(member["condition"], collection_episode=identifier))])
    spec = dict(old, source_sha=source_sha, source_root=str(target), source_files=members,
                source_seal=hashlib.sha256(json.dumps(members,sort_keys=True).encode()).hexdigest(),
                registered_root=str(root), campaign_root=str(root), mobile_group=str(root),
                unattended_control=str(root/"control"), deadline_epoch=end, campaign_deadline_epoch=end,
                wall_limit_s=int(end-time.time()), sim_only_project_authorization=pin(auth),
                registered_panel=pin(panel), v64_observation=pin(binding), new_physical_requests_budget=1,
                cases=[str(root/"cases"/identifier/"EXECUTION_SPEC.json")])
    spec.pop("n1_archive_storage", None)
    spec.pop("retained_n1", None)
    spec.update({key: str(root/key) for key in ("logs","cache","tmp","data")})
    spec["native_environment"] = dict(old["native_environment"], TMPDIR=spec["tmp"], XDG_CACHE_HOME=spec["cache"],
        ROS_LOG_DIR=str(root/"logs/ros"), PYTHONPATH=os.pathsep.join(
            (str(target), str(target/"src"), old["native_environment"]["PYTHONPATH"])))
    # Verify the original validator from the new overlay, not the parent import.
    write(overlay/"PROSPECTIVE.private.json", spec)
    smoke = [old["python"], "-B", str(target/sl/"v64_n1_observation.py"), "preflight",
             str(overlay/"PROSPECTIVE.private.json")]
    subprocess.run(smoke, cwd=target, env=dict(os.environ, **spec["native_environment"]), check=True)
    write(overlay/"CAPACITY.private.json", capacity)
    if execute:
        with closing(OwnerLock(spec["mobile_world_lock"])):
            require_unstarted_cases(spec["registry"], {identifier})
            dirs(root)
            (root/"cases").mkdir()
            register(spec, [(root, spec)])
        print(json.dumps(launch_service(spec, "wm02-"+root.name, "--core-multienv-n1", bootstrap="v6_bootstrap.py")))


def preflight(path):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate
    from scripts.pi0292.durable import read
    from scripts.sla.core_multienv_n1 import make_case

    spec = read(path)
    row = fixed_card(read_pin(spec["sim_only_project_authorization"]))
    _, case = make_case(spec, row, None, min(spec["deadline_epoch"], time.time()+600))
    case["v64_observation"] = spec["v64_observation"]
    validate(case)
    print("Original native gates, fixed reference and one interface slot verified. No physical request.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("stage")
    for key in ("previous", "root", "overlay"):
        prep.add_argument(key, type=Path)
    prep.add_argument("source_sha")
    prep.add_argument("--execute", action="store_true")
    prep.add_argument("--prior-observer", type=Path)
    prep.add_argument("--prior-close", type=Path)
    check = sub.add_parser("preflight")
    check.add_argument("path", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"stage": stage, "preflight": preflight}[command](**args)
