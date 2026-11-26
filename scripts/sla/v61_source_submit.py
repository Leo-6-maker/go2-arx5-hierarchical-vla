"""Admit V6.1 SOURCE through the accounted single-H200 PBS route."""

import argparse
from contextlib import closing
import os
from pathlib import Path
import shlex
import subprocess
import time

from hopper_pbs import BalanceLoginExpired, PBS, PROJECT, native, project_grant, wall_seconds
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock
from scripts.sla.hopper_parent_custody import write_once
from scripts.sla.hopper_site import DEADLINE
from v61_source_bank import ENDPOINTS, PLAN
from v61_source_course import BANK_RECEIPT, R4, read_bank
from v61_source_worker import MODES, PROBE_STEPS, parent_gate, validated_boundary
from v62_source_admission import V62, provisional_gate, write_capacity
from v6_dual_source_submit import H200, SERVER, admit


MAX_WALL = {"probe": 3600, "250": 5400, "500": 5400, "1000": 7200, "5000": 21600}
PRIOR_NAMES = {"slav6ret1", "slav6ret2", "slav6rref", "slav6r5rel"}
V4_HISTORY = {"735204", "735374", "735409"}


def validated_config(config, mode):
    config = read(config)
    require(config["plan"] == PLAN and config["mode"] == mode, "Use the exact V6.1 capsule declaration")
    result = dict(plan=PLAN, stage=mode, start=config["start"],
                  end=PROBE_STEPS if mode == "probe" else int(mode),
                  bank_receipt_sha256=BANK_RECEIPT, r4_receipt_sha256=R4,
                  core_parent_release=config["core_parent_release"])
    if config.get("science_plan") == V62:
        result.update(science_plan=V62, provisional=config["provisional"])
    if result.get("provisional"):
        provisional_gate(result["core_parent_release"], mode)
    else:
        parent_gate(result["core_parent_release"])
    checked = read(pin(config["core_host"]["path"], config["core_host"]["sha256"]))
    require(checked["bank_receipt_sha256"] == BANK_RECEIPT and checked["optimizer_visits"] == 0
            and len(checked["actual_batches"]) == 2, "Corelab must read the exact learner input first")
    if mode != "probe":
        proof = config["probe_terminal"]
        require(read(pin(proof["path"], proof["sha256"]))["status"] == "QUALIFICATION_PASS",
                "An actual qualified H200 job is required before formal training")
        result["probe_terminal"] = proof
    base = 0 if mode in ("probe", "250") else ENDPOINTS[ENDPOINTS.index(int(mode)) - 1]
    require(type(result["start"]) is int and base <= result["start"] < result["end"]
            and (mode != "probe" or result["start"] == 0), "Invalid V6.1 SOURCE clock")
    if base:
        ref = config["core_segment_release"]
        validated_boundary(ref, base)
        result["core_segment_release"] = ref
    if result["start"]:
        result["previous"] = config["previous"]
        if result["start"] > base:
            result["partial_segment_readback"] = config["partial_segment_readback"]
    return config, result


def accounted_prior(paths, original, grant):
    require(original in paths, "Initial authorized V4 GPU accounting is required")
    accrued = read(original / "INTENT.json")["prior_upper_gpu_s"]
    seen, observed = set(), set()
    for path in paths:
        release = read(path / "RELEASE.json")
        job = release["job"]
        require(job not in seen, "Duplicate historical PBS accounting")
        seen.add(job)
        field = native(job)
        resource = field["Resource_List"]
        require(field["job_state"] == "F" and field["project"] == PROJECT
                and field["Job_Owner"].split("@")[0] == grant["account"] and int(resource["ngpus"]) == 1
                and wall_seconds(resource["walltime"]) == release["reserved_s"],
                "A prior GPU owner remains unresolved")
        observed.add(field["Job_Name"])
        accrued += release["reserved_s"]
    require(PRIOR_NAMES <= observed and V4_HISTORY <= {job.split(".")[0] for job in seen},
            "Include the original V4 course, R0 attempts and retained arms in accounting")
    return accrued


def submit(mode, capsule, config, paths, runtime, source, v4_code, original, prior):
    requirements, binding = validated_config(config, mode)
    inputs = [Path(paths[key]) for key in (
        "custody", "short", "prepared", "corrected", "extension", "single", "bank", "r4")]
    read_bank(inputs[6])
    budget = int(requirements["budget_seconds"])
    require(1800 <= budget <= MAX_WALL[mode] and budget % 60 == 0 and time.time() + budget < DEADLINE,
            "The original Hopper deadline and bounded walltime apply")
    with closing(OwnerLock(capsule / "SUBMIT.lock")):
        require(not any((capsule / name).exists() for name in ("INTENT.json", "JOB.json", "run")),
                "Reconcile the original exact PBS intent instead of submitting twice")
        modern = binding.get("science_plan") == V62
        try:
            grant = project_grant(budget)
        except BalanceLoginExpired:
            if not modern:
                raise
            from hopper_project_admission import project_period

            lineage = Path(requirements["project_lineage"])
            grant = project_period(requirements["project_basis"], lineage, budget)
            grant["lineage"] = str(lineage)
        prior_gpu_s = accounted_prior(prior, original, grant)
        require(prior_gpu_s + budget <= 320 * 3600, "The inherited 320 GPU-hour ceiling is exhausted")
        active = subprocess.check_output([str(PBS / "qselect"), "-q", f"@{SERVER}", "-u", grant["account"]],
                                         text=True, timeout=45).split()
        for job in active:
            state = native(job)
            require(state["job_state"] == "F" or not state["Job_Name"].startswith(("slav4", "slav6")),
                    "An existing MAIN or SOURCE owner is still active")

        name = "slav61q" if mode == "probe" else "slav61s" + mode
        if modern:
            name = "slav62q" if mode == "probe" else "slav62s" + mode
            capacity = write_capacity(capsule, inputs[7], mode, binding["start"], binding["end"], budget)
            write_once(capsule / "WRITE_CAPACITY.json", capacity)
            grant_file = capsule / "GRANT.json"
            write_once(grant_file, grant)
            binding.update(budget_seconds=budget, site_grant=dict(path=str(grant_file), sha256=sha(grant_file)))
        code = capsule / "code"
        worker = code / "v61_source_worker.py"
        declaration = capsule / "BINDING.json"
        write_once(declaration, binding)
        authorized = [config, declaration, *sorted(code.glob("*.py")), *sorted(v4_code.glob("*.py"))]
        authorized.extend(Path(x["path"]) for x in (binding["core_parent_release"], requirements["core_host"]))
        if modern:
            authorized.append(grant_file)
        list_file = capsule / "INPUTS.sha256"
        with list_file.open("x") as stream:
            stream.writelines(f"{sha(path)}  {path}\n" for path in authorized)
        python = runtime / "interpreter/bin/python3.12"
        import_paths = (code, v4_code, source, source / "src", runtime / "data_dependencies",
                        runtime / "upstream", runtime / "packages")
        command = shlex.join(map(str, (python, "-B", worker, mode, capsule, *inputs)))
        script = capsule / "job.pbs"
        with script.open("x") as stream:
            stream.write("\n".join(("#!/bin/bash -l", f"#PBS -N {name}", f"#PBS -P {PROJECT}", "#PBS -q auto",
                       f"#PBS -l select={H200}", f"#PBS -l walltime={budget//3600:02d}:{budget%3600//60:02d}:00",
                       "#PBS -m n", "#PBS -r n", f"#PBS -o {capsule / 'stdout.log'}",
                       f"#PBS -e {capsule / 'stderr.log'}", "set -euo pipefail", "umask 077",
                       f"cd {shlex.quote(str(capsule))}",
                       "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
                       "export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false",
                       f"export PBS_SERVER={shlex.quote(os.environ['PBS_SERVER'])}",
                       f"export PYTHONPATH={shlex.quote(':'.join(map(str, import_paths)))}",
                       "sha256sum --status -c INPUTS.sha256", "exec " + command, "")))
        intent = dict(plan=PLAN, name=name, role=mode, budget_seconds=budget, prior_upper_gpu_s=prior_gpu_s,
                      bank_receipt_sha256=BANK_RECEIPT, parent_release_sha256=binding["core_parent_release"]["sha256"],
                      program_sha256=sha(script), inputs_sha256=sha(list_file))
        if modern:
            intent.update(science_plan=V62, provisional=binding["provisional"], binding_sha256=sha(declaration))
        print(admit(capsule, intent, script, name, budget, grant), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    for arg in ("capsule", "config", "paths", "runtime", "source", "v4_code", "original"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("prior", type=Path, nargs="+")
    args = vars(parser.parse_args())
    args["paths"] = read(args["paths"])
    os.umask(0o077)
    submit(**args)
