"""Stage the fixed STABLE pair with the independently checked warmup simulator."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


STOCK = "35f0c4e09ca9ff7a0c1d19ce9af7f03c48a49f5a4405a7d0d4480c97d92e58e9"
ROLE_INPUTS = "c4d59d8e7cad412a6b88ae38c7761cf199e45f888c2a895639068745b58105da"
BUILD = "ac84febc5752f47fddfd364db5396bb4eb1c3363e45ca81a7d6487201e66b808"
N1 = "ea1ace48922c13a7e0da4c512d667dcb378fdcbd5f8c85228dfb0f7f6b8012f7"
PHASE_ENV = "WM02_WARMUP_PD_PHASE_V64"
SLOTS = {"MEV1-ENG-N2-SHORT-06", "MEV1-ENG-N2-SHORT-07"}


def phase_assets(bound):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

    inputs = bound["phase_inputs"]
    build = bound["phase_build"]
    if inputs["sha256"] != ROLE_INPUTS or build["sha256"] != BUILD:
        raise PermissionError("Require the fixed STABLE role inputs and original simulator build")
    proof = read_pin(inputs)
    receipt = read_pin(build)
    binary = Path(build["path"]).with_name("rl_sim_mujoco")
    if (receipt["new_binary_sha256"] != proof["candidate_simulator_sha256"]
            or proof["matched_n1_sha256"] != N1 or proof["original_prospective_sha256"] != STOCK
            or {row["id"] for row in proof["slots"]} != SLOTS
            or proof["original_request_cap"] != 2 or proof["original_tick_cap"] != 21406):
        raise PermissionError("Require the original paired slots and native N1 positive control")
    return proof, str(binary), receipt


def mask_for(proof, row, slot):
    matched, = (item for item in proof["slots"] if item["id"] == row["slot"]["id"])
    if (matched["family"] != row["family"] or matched["world"] != row["world_role"]
            or matched["cpu"] != slot["cpus"] or matched["gpu_index"] != slot["gpu_index"]
            or row["motion"] != "STABLE"):
        raise PermissionError("Warmup input belongs to a different world, layout or fixed reference")
    return matched["callback_mask"]


def bind_case(case, bound, row, slot):
    proof, binary, receipt = phase_assets(bound)
    files = case["frozen_files"]
    if files[case["native_binary"]] != receipt["original_binary_sha256"]:
        raise PermissionError("The candidate must descend from the registered simulator")
    case["native_binary"] = binary
    case["frozen_files"] = {**files, binary: receipt["new_binary_sha256"]}
    case["native_environment"][PHASE_ENV] = mask_for(proof, row, slot)


def validate_assets(case, template, bound, row, slot):
    from go2arx5_vla.world_model.storage import digest

    if "phase_inputs" not in bound:
        if PHASE_ENV in case["native_environment"] or case["frozen_files"] != template["frozen_files"]:
            raise PermissionError("The stock entry requires the unmodified original simulator and phase")
        return None
    proof, binary, receipt = phase_assets(bound)
    original = template["frozen_files"]
    expected = receipt["new_binary_sha256"]
    if (case["native_binary"] != binary or digest(binary) != expected
            or case["frozen_files"] != {**original, binary: expected}
            or original[template["native_binary"]] != receipt["original_binary_sha256"]
            or case["native_environment"].get(PHASE_ENV) != mask_for(proof, row, slot)):
        raise PermissionError("The simulator, exact role phase and original asset inventory must agree")
    return expected


def simulator_hash(case):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    from scripts.sla.v64_n2_native import member

    bound, row, slot = member(case)
    return validate_assets(case, read_pin(row["reference"]), bound, row, slot)


def install_overlay(stock, destination):
    from go2arx5_vla.world_model.storage import digest
    from scripts.sla.v64_n1_observation import patched

    source = Path(stock["source_root"])
    files = dict(stock["source_files"])
    if any(digest(source / name) != sha for name, sha in files.items()):
        raise PermissionError("The stock N2 source no longer matches its seal")
    shutil.copytree(source, destination, copy_function=os.link,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    edits = {
        "scripts/sla/v64_n2_native.py": [
            ('"mobile_guard_qualification", "mobile_safety_binding", "frozen_files",',
             '"mobile_guard_qualification", "mobile_safety_binding",'),
            ('    exact_fields = (\n',
             '    from scripts.sla.v64_n2_phase_stage import validate_assets\n'
             '    validate_assets(case, template, bound, row, slot)\n    exact_fields = (\n'),
            ('    return folder, case\n',
             '    if "phase_inputs" in bound:\n'
             '        from scripts.sla.v64_n2_phase_stage import bind_case\n'
             '        bind_case(case, bound, row, slot)\n    return folder, case\n'),
        ],
        "src/go2arx5_vla/world_model/multitask_scope_v6.py": [
            ('    for key,sha in BINARIES.items():\n',
             '    binaries = BINARIES\n'
             '    if spec.get("scope") == N2_SCOPE:\n'
             '        from scripts.sla.v64_n2_phase_stage import simulator_hash\n'
             '        candidate = simulator_hash(spec)\n'
             '        if candidate is not None:\n'
             '            binaries = {**BINARIES, "native_binary": candidate}\n'
             '    for key,sha in binaries.items():\n'),
        ],
    }
    for relative, changes in edits.items():
        patched(destination / relative, changes)
        files[relative] = digest(destination / relative)
    relative = "scripts/sla/v64_n2_phase_stage.py"
    shutil.copyfile(__file__, destination / relative)
    files[relative] = digest(destination / relative)
    return files


def stage(stock_path, inputs, build, n1_result, root, overlay, revision):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    from scripts.pi0292.durable import read, write
    from scripts.sla.v64_n2_native import binding, check_student_turn, require_capacity
    from scripts.wm02.multitask_stage_cards_v6 import pin, require_unstarted_cases

    stock_ref = pin(stock_path)
    if stock_ref["sha256"] != STOCK:
        raise PermissionError("Use the original fixed STABLE pair prospective")
    stock = read_pin(stock_ref)
    old_bound, _, rows, _ = binding(stock["v64_n2"])
    n1_ref = pin(n1_result)
    if (n1_ref["sha256"] != N1 or read(n1_result.parent / "control/LEASE.json")["status"] != "CLOSED"
            or stock["uid"] != os.getuid() or stock["host"] != os.uname().nodename or root.exists()
            or {row["slot"]["id"] for row in rows} != SLOTS
            or len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision)):
        raise PermissionError("Require the closed N1 trial, original host and two unused STABLE slots")
    require_unstarted_cases(stock["registry"], SLOTS)
    check_student_turn(stock)
    capacity = require_capacity(root.parent, stock)
    old_auth = read_pin(stock["sim_only_project_authorization"])
    now = time.time()
    end = min(now + 1250, stock["allocation_deadline_epoch"], old_auth["goal_start"] + 7 * 86400)
    if end - now < 700:
        raise TimeoutError("The inherited allocation cannot cover this pair")
    bound = dict(old_bound, phase_inputs=pin(inputs), phase_build=pin(build))
    phase_assets(bound)
    overlay.mkdir(mode=0o700)
    destination = overlay / "source"
    files = install_overlay(stock, destination)
    binding_path = overlay / "BINDING.private.json"
    write(binding_path, bound)
    binding_ref = pin(binding_path)
    authority = overlay / "AUTHORITY.private.json"
    write(authority, dict(old_auth, deadline_epoch=end, source_sha=revision, v64_n2=binding_ref))
    prospective = dict(stock, source_root=str(destination), source_files=files, source_sha=revision,
        source_seal=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        v64_n2=binding_ref, sim_only_project_authorization=pin(authority), registered_root=str(root),
        campaign_root=str(root), mobile_group=str(root), unattended_control=str(root / "control"),
        deadline_epoch=end, campaign_deadline_epoch=end, wall_limit_s=int(end - now),
        cases=[str(root / "cases" / row["slot"]["id"] / "EXECUTION_SPEC.json") for row in rows])
    prospective.update({key: str(root / key) for key in ("logs", "cache", "tmp", "data")})
    prospective["native_environment"] = dict(stock["native_environment"], TMPDIR=prospective["tmp"],
        XDG_CACHE_HOME=prospective["cache"], ROS_LOG_DIR=str(root / "logs/ros"),
        PYTHONPATH=os.pathsep.join((str(destination), str(destination / "src"),
                                  stock["native_environment"]["PYTHONPATH"])))
    path = overlay / "PROSPECTIVE.private.json"
    write(path, prospective)
    write(overlay / "CAPACITY.private.json", capacity)
    subprocess.run([stock["python"], "-B", "-m", "scripts.sla.v64_n2_phase_stage", "verify", str(path)],
        cwd=destination, env=dict(os.environ, **prospective["native_environment"]), check=True, timeout=60)
    print(path)


def verify(path):
    from go2arx5_vla.world_model.multitask_scope_v6 import validate
    from scripts.pi0292.durable import read
    from scripts.sla.v64_n2_native import binding, prepare_case
    from scripts.sla.v64_n2_native_stage import verify as stock_verify

    stock_verify(path)
    spec = read(path)
    bound, _, rows, slots = binding(spec["v64_n2"])
    first, second = (prepare_case(spec, row, slots[row["world_role"]], bound, {}, spec["deadline_epoch"])[1]
                     for row in rows)
    for case, other in ((first, second), (second, first)):
        swapped = dict(case, native_environment={**case["native_environment"],
                       PHASE_ENV: other["native_environment"][PHASE_ENV]})
        try:
            validate(swapped)
        except PermissionError:
            continue
        raise RuntimeError("Candidate accepted the opposing role's archived callback mask")
    print("Candidate pair passes; both swapped phase masks rejected; zero native requests.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="operation", required=True)
    prep = commands.add_parser("stage")
    for name in ("stock_path", "inputs", "build", "n1_result", "root", "overlay"):
        prep.add_argument(name, type=Path)
    prep.add_argument("revision")
    for name in ("verify", "launch"):
        commands.add_parser(name).add_argument("path", type=Path)
    args = vars(cli.parse_args())
    operation = args.pop("operation")
    if operation == "launch":
        from scripts.sla.v64_n2_native_stage import launch

        verify(args["path"])
        launch(args["path"])
    else:
        {"stage": stage, "verify": verify}[operation](**args)
