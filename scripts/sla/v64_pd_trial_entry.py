"""Stage the isolated F1 STABLE simulator timing experiment."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


SLOT = "MEV1-ENG-INTERFACE-03"
BASE = "MEV1-ENG-N1-SHORT-07"
PREFLIGHT = "1e581988997ff3d7a3bd5a563752080ef78d9940a19f26109e513db540734024"
OBSERVER = "a7630e674493356f2e06a36c40556250e04a742723e25a02a03141046c0a3a13"
PHASE_ENV = "WM02_WARMUP_PD_PHASE_V64"
TICKS = 10703


def decode(reference):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

    proof = read_pin(reference)
    if (reference["sha256"] != PREFLIGHT or proof["schema"] != "V64_PD_PHASE_TRIAL_PREFLIGHT"
            or proof["reference_slot"]["id"] != SLOT or not proof["slot_unstarted_at_readback"]
            or proof["native_integrations"] or proof["actual_native_requests"]):
        raise PermissionError("The exact closed CPU trial inputs are required")
    return proof


def card(authority, bound):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    from scripts.sla.core_multienv_n1 import fixed_cards
    from scripts.sla.v64_n1_observation import V64

    proof = decode(bound["phase_trial"])
    original = read_pin(bound["original_authority"])
    comparison = fixed_cards(original)[BASE]
    if (bound["plan"] != V64 or bound["comparison"] != BASE or bound["slot"] != proof["reference_slot"]
            or bound["fixture"] != comparison["reference"] or proof["reference"] != comparison["reference"]
            or proof["archived_state"] != comparison["recording"]
            or proof["original_tolerance"] != original["tolerance"]
            or authority["native_request_cap"] != 1 or authority["native_tick_cap"] != TICKS
            or any(authority[k] != original[k] for k in ("tolerance","goal_start","cpus","world_lock"))):
        raise PermissionError("Only the original F1 STABLE comparator and finite engineering world are allowed")
    return dict(comparison, slot=proof["reference_slot"])


def member(spec):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

    bound = read_pin(spec["v64_observation"])
    proof = decode(bound["phase_trial"])
    original = read_pin(proof["reference"])
    if (spec["panel_entry_id"] != SLOT or spec["scope"] != "SLA_CORE_MULTIENV_V1_N1_ENGINEERING"
            or spec["max_ticks"] != TICKS or spec["collection_split"] != "ENGINEERING"
            or spec["expert_training_eligible"] is not False
            or any(spec[key] for key in ("new_training_visits_budget","new_optimizer_updates_budget",
                                         "new_model_generations_budget"))):
        raise PermissionError("Only the isolated nontraining timing case may load the candidate")
    return proof, original


def bind_case(spec):
    proof, _ = member(spec)
    binref = proof["simulator_candidate"]
    spec["native_binary"] = binref["path"]
    spec["frozen_files"] = dict(spec["frozen_files"], **{binref["path"]:binref["sha256"]})
    spec["native_environment"] = dict(spec["native_environment"], **{PHASE_ENV:proof["optional_callback_mask"]})


def checked_binary(spec):
    from go2arx5_vla.world_model.storage import digest

    proof, original = member(spec)
    binref = proof["simulator_candidate"]
    if (spec["native_binary"] != binref["path"] or digest(binref["path"]) != binref["sha256"]
            or spec["native_environment"].get(PHASE_ENV) != proof["optional_callback_mask"]
            or spec["frozen_files"] != dict(original["frozen_files"], **{binref["path"]:binref["sha256"]})
            or any(spec[name] != proof["unchanged_binaries"][name]["path"] for name in proof["unchanged_binaries"])):
        raise PermissionError("The actual simulator, phase mask and untouched frozen assets must match")
    return binref["sha256"]


def frozen_files(spec, template):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

    binding = spec.get("v64_observation")
    if binding and "phase_trial" in read_pin(binding):
        checked_binary(spec)
        return template["frozen_files"]
    return spec["frozen_files"]


def edits():
    sl = Path("scripts/sla")
    wm = Path("src/go2arx5_vla/world_model")
    return {
        sl/"v64_n1_observation.py": [
            ("def slot_id(binding):\n", "def slot_id(binding):\n"
             "    if 'phase_trial' in binding:\n"
             "        from scripts.sla.v64_pd_trial_entry import SLOT, decode\n"
             "        decode(binding['phase_trial'])\n"
             "        return SLOT\n"),
            ('    old = read_pin(bound["original_authority"])\n',
             "    if 'phase_trial' in bound:\n"
             "        from scripts.sla.v64_pd_trial_entry import card\n"
             "        return card(authority,bound)\n"
             '    old = read_pin(bound["original_authority"])\n'),
            ('detach=identifier == DETACH_SLOT)', 'detach=identifier in (DETACH_SLOT,"MEV1-ENG-INTERFACE-03"))'),
            ('            validate(case)\n', "            if 'phase_trial' in bound:\n"
             "                from scripts.sla.v64_pd_trial_entry import bind_case\n"
             "                bind_case(case)\n"
             "            validate(case)\n"),
            ("            if identifier in (CLOSE_SLOT, DETACH_SLOT):\n"
             '                result["observer_shutdown"] = shutdown_receipt(folder, detached=identifier == DETACH_SLOT)',
             '            if identifier in (CLOSE_SLOT, DETACH_SLOT, "MEV1-ENG-INTERFACE-03"):\n'
             '                result["observer_shutdown"] = shutdown_receipt(folder, detached=identifier != CLOSE_SLOT)'),
        ],
        sl/"core_multienv_n1.py": [
            ("    if (case['owner'] != OWNER or authority['owner'] != OWNER",
             "    from scripts.sla.v64_pd_trial_entry import frozen_files\n"
             "    bound_frozen_files = frozen_files(case,template)\n"
             "    if (case['owner'] != OWNER or authority['owner'] != OWNER"),
            ("or not all(case.get(key) == template.get(key) for key in subset)",
             "or bound_frozen_files != template['frozen_files']\n"
             "            or not all(case.get(key) == template.get(key) for key in subset if key != 'frozen_files')"),
        ],
        wm/"multitask_scope_v6.py": [
            ("    for key,sha in BINARIES.items():\n",
             "    binary_hashes = BINARIES\n"
             "    ref = spec.get('v64_observation')\n"
             "    if ref and 'phase_trial' in read_pin(ref):\n"
             "        from scripts.sla.v64_pd_trial_entry import checked_binary\n"
             "        binary_hashes = {**BINARIES,'native_binary':checked_binary(spec)}\n"
             "    for key,sha in binary_hashes.items():\n"),
        ],
    }


def stage(previous, proof_path, root, overlay, revision):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.durable import read, write
    from scripts.sla.v64_n1_observation import V64, patched, require_writable
    from scripts.wm02.multitask_stage_cards_v6 import pin, require_unstarted_cases

    old = read(previous/"EXECUTION_SPEC.json")
    closed = read(previous/"control/LEASE.json")
    terminal = read_pin(dict(path=str(previous/"RESULT.private.json"),sha256=OBSERVER))
    if (closed["status"] != "CLOSED" or terminal["exit_code"] or not terminal["observer_shutdown"]["detached_native_exit"]
            or old["uid"] != os.getuid() or old["host"] != os.uname().nodename or root.exists()):
        raise PermissionError("Require the exact closed original observation and a new output root")
    proof_ref = pin(proof_path)
    proof = decode(proof_ref)
    prior = read_pin(old["v64_observation"])["original_authority"]
    original = read_pin(prior)
    require_unstarted_cases(old["registry"], {SLOT})
    require_writable(root.parent)
    end = min(time.time()+1000, old["allocation_deadline_epoch"], original["goal_start"]+7*86400)
    if end-time.time() < 700:
        raise TimeoutError("Original interface deadline is too short for this independent case")
    source = Path(old["source_root"])
    members = dict(old["source_files"])
    if any(digest(source/name) != sha for name,sha in members.items()):
        raise PermissionError("The actual observer source seal has changed")
    overlay.mkdir(mode=0o700)
    dest = overlay/"source"
    shutil.copytree(source,dest,copy_function=os.link,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for relative, replacements in edits().items():
        patched(dest/relative,replacements)
        members[str(relative)] = digest(dest/relative)
    relative = "scripts/sla/v64_pd_trial_entry.py"
    shutil.copyfile(__file__,dest/relative)
    members[relative] = digest(dest/relative)
    binding = overlay/"BINDING.private.json"
    write(binding,dict(plan=V64,slot=proof["reference_slot"],comparison=BASE,fixture=proof["reference"],
                       original_authority=prior,baseline_terminal=proof["matched_closed_negative"],phase_trial=proof_ref))
    auth = overlay/"AUTHORITY.private.json"
    write(auth,dict(original,deadline_epoch=end,source_sha=revision,native_request_cap=1,native_tick_cap=TICKS,
                    v64_observation=pin(binding)))
    panel = overlay/"PANEL.private.json"
    baseline = read(Path(proof["matched_closed_negative"]["path"]).parent/"EXECUTION_SPEC.json")
    row, = (r for r in read_pin(baseline["registered_panel"]) if r["id"] == BASE)
    write(panel,[dict(row,id=SLOT,condition=dict(row["condition"],collection_episode=SLOT))])
    prospect = dict(old, source_sha=revision,source_root=str(dest),source_files=members,
                    source_seal=hashlib.sha256(json.dumps(members,sort_keys=True).encode()).hexdigest(),
                    registered_root=str(root),campaign_root=str(root),mobile_group=str(root),
                    unattended_control=str(root/"control"),deadline_epoch=end,campaign_deadline_epoch=end,
                    wall_limit_s=int(end-time.time()),sim_only_project_authorization=pin(auth),
                    registered_panel=pin(panel),v64_observation=pin(binding),new_physical_requests_budget=1,
                    cases=[str(root/"cases"/SLOT/"EXECUTION_SPEC.json")])
    prospect.update({key:str(root/key) for key in ("logs","cache","tmp","data")})
    prospect["native_environment"] = dict(old["native_environment"],TMPDIR=prospect["tmp"],
        XDG_CACHE_HOME=prospect["cache"],ROS_LOG_DIR=str(root/"logs/ros"),
        PYTHONPATH=os.pathsep.join((str(dest),str(dest/"src"),old["native_environment"]["PYTHONPATH"])))
    output = overlay/"PROSPECTIVE.private.json"
    write(output,prospect)
    command = [prospect["python"],"-B","-m","scripts.sla.v64_pd_trial_entry","verify",str(output)]
    subprocess.run(command,cwd=dest,env=dict(os.environ,**prospect["native_environment"]),check=True,timeout=45)
    print(output)


def verify(path):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate
    from scripts.pi0292.durable import read
    from scripts.sla.core_multienv_n1 import make_case

    spec = read(path)
    authority = read_pin(spec["sim_only_project_authorization"])
    bound = read_pin(spec["v64_observation"])
    _, case = make_case(spec,card(authority,bound),{},spec["deadline_epoch"])
    case["v64_observation"] = authority["v64_observation"]
    bind_case(case)
    validate(case)
    invalid = dict(case,native_environment={**case["native_environment"],PHASE_ENV:"0"*100})
    try:
        validate(invalid)
    except PermissionError:
        pass
    else:
        raise RuntimeError("The actual trial entry admitted an altered warmup phase mask")
    print("Trial gate passed, changed phases rejected, zero native starts.")


def launch(path):
    from go2arx5_vla.world_model.mfree_recovery_v2 import register, launch_service
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    from scripts.pi0292.durable import OwnerLock, read
    from scripts.sla.v64_n1_observation import require_writable
    from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases

    spec = read(path)
    root = Path(spec["registered_root"])
    verify(path)
    with closing(OwnerLock(spec["mobile_world_lock"])):
        handoff = Path(read_pin(spec["student_handoff_owner"])["registered_root"])
        if (handoff/"STUDENT_TURN_READY.private.json").exists() or root.exists() or spec["deadline_epoch"]-time.time()<700:
            raise PermissionError("Site handoff, ownership or original finite deadline prevents launch")
        require_unstarted_cases(spec["registry"],{SLOT})
        require_writable(root.parent)
        dirs(root)
        (root/"cases").mkdir()
        register(spec,[(root,spec)])
    print(json.dumps(launch_service(spec,"wm02-"+root.name,"--core-multienv-n1",bootstrap="v6_bootstrap.py")))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    subs = cli.add_subparsers(dest="operation",required=True)
    prep = subs.add_parser("stage")
    for name in ("previous","proof_path","root","overlay"):
        prep.add_argument(name,type=Path)
    prep.add_argument("revision")
    for name in ("verify","launch"):
        subs.add_parser(name).add_argument("path",type=Path)
    args = vars(cli.parse_args())
    {"stage":stage,"verify":verify,"launch":launch}[args.pop("operation")](**args)
