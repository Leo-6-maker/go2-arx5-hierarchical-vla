"""Stage a sealed N2 overlay without altering the qualified single-world source."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.sla.v64_n1_observation import patched
from scripts.sla.v64_n2_native import (SCOPE, TICKS, V64, binding, check_student_turn,
                                        prepare_case, require_capacity)


def edits():
    wm = Path("src/go2arx5_vla/world_model")
    entry = Path("scripts/wm02")
    return {
        entry / "v6_bootstrap.py": [
            ("    core_hopper_expert = core_hopper_expert or core_multienv_expert",
             "    core_multienv_n2 = (spec['plan_commit'] == '4fe2c8cf1b69c168b2e3079e061bb7c702089cd3'\n"
             "                        and spec['scope'] == 'SLA_CORE_MULTIENV_V1_N2_ENGINEERING'\n"
             "                        and operation in ('--case', '--core-multienv-n2'))\n"
             "    core_hopper_expert = core_hopper_expert or core_multienv_expert"),
            ("or core_hopper_student or core_multienv_n1) and",
             "or core_hopper_student or core_multienv_n1 or core_multienv_n2) and"),
            ("    if operation == '--case' and (core_hopper_expert or core_multienv_n1):",
             "    if core_multienv_n2:\n"
             "        from scripts.sla.v64_n2_native import check_case_locks, run\n"
             "        if operation == '--core-multienv-n2':\n"
             "            raise SystemExit(run(spec))\n"
             "        check_case_locks(spec, int(sys.argv[sys.argv.index('--world-lock-fd')+1]),\n"
             "                         int(sys.argv[sys.argv.index('--renderer-lock-fd')+1]))\n"
             "    if operation == '--case' and (core_hopper_expert or core_multienv_n1):"),
        ],
        entry / "mobile_boundary_runner_v1.py": [
            ("    n1=spec.get('scope')=='SLA_CORE_MULTIENV_V1_N1_ENGINEERING'",
             "    n1=spec.get('scope') in ('SLA_CORE_MULTIENV_V1_N1_ENGINEERING', 'SLA_CORE_MULTIENV_V1_N2_ENGINEERING')"),
        ],
        wm / "multitask_scope_v6.py": [
            ("N1_SCOPE = 'SLA_CORE_MULTIENV_V1_N1_ENGINEERING'",
             "N1_SCOPE = 'SLA_CORE_MULTIENV_V1_N1_ENGINEERING'\n"
             "from scripts.sla.v64_n2_native import SCOPE as N2_SCOPE"),
            ("def scope_identity(spec):\n    if spec.get('scope') == N1_SCOPE:",
             "def scope_identity(spec):\n    if spec.get('scope') == N2_SCOPE:\n"
             "        from scripts.sla.v64_n2_native import PLAN, AUTHORITY\n"
             "        return N2_SCOPE, PLAN, (), AUTHORITY\n"
             "    if spec.get('scope') == N1_SCOPE:"),
            ("    n1_probe = scope == N1_SCOPE and interface",
             "    n1_probe = scope in (N1_SCOPE, N2_SCOPE) and interface"),
            ("            or env.get('ROS_LOCALHOST_ONLY')!='1' or env.get('ROS_AUTOMATIC_DISCOVERY_RANGE')!='LOCALHOST'",
             "            or (scope == N2_SCOPE and not interface)\n"
             "            or (scope != N2_SCOPE and (env.get('ROS_LOCALHOST_ONLY')!='1'\n"
             "                or env.get('ROS_AUTOMATIC_DISCOVERY_RANGE')!='LOCALHOST'))"),
            ("    if scope in adaptation.SCOPES and (authority",
             "    if scope == N2_SCOPE:\n"
             "        from scripts.sla.v64_n2_native import validate_member\n"
             "        validate_member(spec, authority)\n"
             "    if scope in adaptation.SCOPES and (authority"),
            ("    if adapter.spec.get('scope')==N1_SCOPE and adapter.spec['sim_only_role']=='INTERFACE':",
             "    if adapter.spec.get('scope') in (N1_SCOPE, N2_SCOPE) and adapter.spec['sim_only_role']=='INTERFACE':"),
        ],
        wm / "target_sequential.py": [
            ("    argv=[*prefix,'/usr/bin/env','LD_PRELOAD='+':'.join(spec[n]for n in order),",
             "    preloads = [spec[name] for name in order]\n"
             "    if spec.get('v64_n2'):\n        preloads.append(spec['native_environment']['LD_PRELOAD'])\n"
             "    argv=[*prefix,'/usr/bin/env','LD_PRELOAD='+':'.join(preloads),"),
            ("        if name == 'controller' and self.spec.get('v64_observation'):",
             "        if name == 'controller' and self.spec.get('v64_n2'):\n"
             "            from scripts.sla.v64_n2_native import controller_argv\n"
             "            argv, env = controller_argv(self.spec, self.output, argv, env)\n"
             "        elif name == 'controller' and self.spec.get('v64_observation'):"),
            ("        import resource\n        resource.setrlimit(resource.RLIMIT_CORE,(0,0))",
             "        if self.spec.get('v64_n2'):\n"
             "            from scripts.sla.v64_n2_native import REMOVED_ENV, transport_environment\n"
             "            os.environ.update(transport_environment(self.spec, os.environ))\n"
             "            for key in REMOVED_ENV:\n                os.environ.pop(key, None)\n"
             "        import resource\n        resource.setrlimit(resource.RLIMIT_CORE,(0,0))"),
            ("        self.native_env=env",
             "        if s.get('v64_n2'):\n"
             "            from scripts.sla.v64_n2_native import transport_environment\n"
             "            env = transport_environment(s, env)\n"
             "        self.native_env=env"),
            ("        await_native_startup(self)\n        self.publish(self.modes,'HOLD')",
             "        await_native_startup(self)\n"
             "        if s.get('v64_n2'):\n"
             "            from scripts.sla.v64_n2_native import check_paused_transport\n"
             "            check_paused_transport(self)\n"
             "        self.publish(self.modes,'HOLD')"),
            ("                observed = name == 'controller' and self.spec.get('v64_observation')",
             "                observed = name == 'controller' and (self.spec.get('v64_observation') or self.spec.get('v64_n2'))"),
        ],
        wm / "multitask_probe_d2.py": [
            ("    if adapter.spec.get('v64_observation') and expected_stop is None:",
             "    if (adapter.spec.get('v64_observation') or adapter.spec.get('v64_n2')) and expected_stop is None:"),
        ],
    }


def overlay_source(old, destination):
    from go2arx5_vla.world_model.storage import digest

    source = Path(old["source_root"])
    files = dict(old["source_files"])
    if any(digest(source / name) != sha for name, sha in files.items()):
        raise PermissionError("The closed N1 source no longer matches its seal")
    shutil.copytree(source, destination, copy_function=os.link,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative, replacements in edits().items():
        patched(destination / relative, replacements)
        files[str(relative)] = digest(destination / relative)
    for name in ("v64_n1_observation.py", "v64_n2_roles.py", "v64_n2_native.py", "v64_n2_native_stage.py"):
        relative = Path("scripts/sla") / name
        target = destination / relative
        if target.exists():
            target.unlink()
        shutil.copy2(Path(__file__).with_name(name), target)
        files[str(relative)] = digest(target)
    return files


def prospective(previous, n1, roster, roles, profile, library, filesystem_proof, root, overlay, commit, pair_number):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    from scripts.pi0292.durable import read, write
    from scripts.wm02.multitask_stage_cards_v6 import pin, require_unstarted_cases

    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("The full overlay Git revision is required")
    old = read(previous / "EXECUTION_SPEC.json")
    authority = read_pin(old["sim_only_project_authorization"])
    observer = read(previous / "RESULT.private.json")
    if (root.exists() or overlay.exists() or old["host"] != os.uname().nodename or old["uid"] != os.getuid()
            or read(previous / "control/LEASE.json")["status"] != "CLOSED"
            or observer["exit_code"] or observer["ticks"] != TICKS or not observer["observer_shutdown"].get("detached_native_exit")):
        raise PermissionError("Stage only after the native single-world observer closes successfully")
    overlay.mkdir(mode=0o700)
    binding_path = overlay / "BINDING.private.json"
    prior = read(n1 / "EXECUTION_SPEC.json")["sim_only_project_authorization"]
    original = read_pin(prior)
    write(binding_path, dict(v64=V64, pair=pair_number, roster=pin(roster), roles=pin(roles),
                             profile=pin(profile), library=pin(library), filesystem_proof=pin(filesystem_proof),
                             original_authority=prior))
    bound, fixed, rows, worlds = binding(pin(binding_path))
    if (original["logical"] != fixed["original_population"] or authority["tolerance"] != fixed["tolerances"]
            or authority["world_lock"] != fixed["original_global_lock"] or original["world_lock"] != authority["world_lock"]):
        raise PermissionError("The original world owner, tolerance and preregistered population must match")
    identifiers = {row["slot"]["id"] for row in rows}
    require_unstarted_cases(old["registry"], identifiers)
    check_student_turn(old)
    capacity = require_capacity(root.parent, old)
    source = overlay / "source"
    source_files = overlay_source(old, source)

    panel = overlay / "PANEL.private.json"
    members = []
    for row in rows:
        old_id = row["n1_comparator"]
        case = read(n1 / "cases" / old_id / "EXECUTION_SPEC.json")
        member, = (item for item in read_pin(case["registered_panel"]) if item["id"] == old_id)
        identifier = row["slot"]["id"]
        members.append(dict(member, id=identifier, condition=dict(member["condition"], collection_episode=identifier)))
    write(panel, members)
    end = min(time.time() + 1250, original["goal_start"] + 7 * 86400, old["allocation_deadline_epoch"])
    auth = overlay / "AUTHORITY.private.json"
    write(auth, dict(original, deadline_epoch=end, source_sha=commit, native_request_cap=2,
                     native_tick_cap=2 * TICKS, v64_n2=pin(binding_path)))
    spec = dict(old, source_sha=commit, source_root=str(source), source_files=source_files,
                source_seal=hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest(),
                registered_root=str(root), campaign_root=str(root), mobile_group=str(root),
                unattended_control=str(root / "control"), deadline_epoch=end, campaign_deadline_epoch=end,
                wall_limit_s=int(end - time.time()), sim_only_project_authorization=pin(auth),
                registered_panel=pin(panel), v64_n2=pin(binding_path), new_physical_requests_budget=2,
                scope=SCOPE, registry_scope=SCOPE, role="SLA_CORE_MULTIENV_N2_ENGINEERING_PARENT",
                cpus=sorted({cpu for slot in worlds.values() for cpu in slot["cpus"]}),
                cases=[str(root / "cases" / row["slot"]["id"] / "EXECUTION_SPEC.json") for row in rows])
    for key in ("n1_archive_storage", "retained_n1", "v64_observation"):
        spec.pop(key, None)
    spec.update({key: str(root / key) for key in ("logs", "cache", "tmp", "data")})
    spec["native_environment"] = dict(old["native_environment"], TMPDIR=spec["tmp"], XDG_CACHE_HOME=spec["cache"],
        ROS_LOG_DIR=str(root / "logs/ros"), PYTHONPATH=os.pathsep.join(
            (str(source), str(source / "src"), old["native_environment"]["PYTHONPATH"])))
    path = overlay / "PROSPECTIVE.private.json"
    write(path, spec)
    write(overlay / "CAPACITY.private.json", capacity)
    check = [spec["python"], "-B", "-m", "scripts.sla.v64_n2_native_stage", "verify", str(path)]
    subprocess.run(check, cwd=source, env=dict(os.environ, **spec["native_environment"]), check=True)
    return path


def verify(path):
    from go2arx5_vla.world_model.multitask_scope_v6 import validate
    from scripts.pi0292.durable import read

    spec = read(path)
    bound, _, rows, worlds = binding(spec["v64_n2"])
    validate_cases = []
    for row in rows:
        slot = worlds[row["world_role"]]
        _, case = prepare_case(spec, row, slot, bound, {}, spec["deadline_epoch"])
        validate(case)
        validate_cases.append(case)
    wrong = dict(validate_cases[0], native_environment=validate_cases[1]["native_environment"])
    try:
        validate(wrong)
    except PermissionError:
        pass
    else:
        raise RuntimeError("N2 accepted the other world's transport and trace paths")
    print(json.dumps(dict(status="SEALED_N2_PREFLIGHT_PASS", cases=[c["panel_entry_id"] for c in validate_cases],
                          wrong_role_environment_rejected=True, new_native_requests=0)))


def launch(path):
    from go2arx5_vla.world_model.mfree_recovery_v2 import launch_service, register
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs
    from scripts.pi0292.durable import OwnerLock, read
    from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases

    spec = read(path)
    _, roster, rows, _ = binding(spec["v64_n2"])
    root = Path(spec["registered_root"])
    with closing(OwnerLock(roster["original_global_lock"])):
        check_student_turn(spec)
        require_unstarted_cases(spec["registry"], {row["slot"]["id"] for row in rows})
        require_capacity(root.parent, spec)
        if spec["deadline_epoch"] - time.time() < 560:
            raise TimeoutError("The inherited finite dispatch window is too short")
        dirs(root)
        (root / "cases").mkdir()
        register(spec, [(root, spec)])
    print(json.dumps(launch_service(spec, "wm02-" + root.name, "--core-multienv-n2", bootstrap="v6_bootstrap.py")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for op in ("verify", "launch"):
        commands.add_parser(op).add_argument("path", type=Path)
    stage = commands.add_parser("prospective")
    for name in ("previous", "n1", "roster", "roles", "profile", "library", "filesystem_proof", "root", "overlay"):
        stage.add_argument(name, type=Path)
    stage.add_argument("commit")
    stage.add_argument("pair_number", type=int)
    args = vars(parser.parse_args())
    print_path = args.pop("operation")
    value = globals()[print_path](**args)
    if value is not None:
        print(value)
