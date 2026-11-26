"""Release one fixed retention contrast to Hopper after Corelab diagnosis."""

import argparse
from contextlib import closing
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

from hopper_pbs import PBS, PROJECT, native, project_grant, wall_seconds
from scripts.pi0292.durable import OwnerLock
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import DEADLINE
from v6_dual_source_submit import H200, SERVER, admit
from v6_retention_worker import ENDPOINTS, FACTORS, MAIN, read, require


BUDGET = 3600
NAMES = {"R1_LOW_LR": "slav6ret1", "R3_TARGET_GRADIENT_ONLY": "slav6ret3"}


def submit(role, capsule, custody, short, bank, source, runtime, v4_code, original, r0_first, r0_retry):
    with closing(OwnerLock(capsule / "SUBMIT.lock")):
        require(not any((capsule / name).exists() for name in ("INTENT.json", "JOB.json", "run")),
                "Reconcile the existing capsule with PBS before submitting")
        receipt = capsule / "CORE_RELEASE.json"
        evidence = read(receipt)
        require(evidence["role"] == role and evidence["parent"] == MAIN
                and evidence["corelab_diagnosis_complete"] and evidence["fixed_endpoint_pairs"] == 120,
                "Expected the fixed Corelab evidence and matched parent")
        require(time.time() + BUDGET < DEADLINE, "Remaining time is too short for a complete job")
        grant = project_grant(BUDGET)
        require(all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (
                    grant["required_units_at_queue_rate_one"], grant["balance_gpu_units"], grant["balance_su"])),
                "The current site grant could not be verified")
        prior = read(original / "INTENT.json")["prior_upper_gpu_s"]
        for prior_capsule in (original, r0_first, r0_retry):
            released = read(prior_capsule / "RELEASE.json")
            job = native(released["job"])
            require(job["job_state"] == "F" and job["project"] == PROJECT
                    and job["Job_Owner"].split("@")[0] == grant["account"]
                    and int(job["Resource_List"]["ngpus"]) == 1
                    and wall_seconds(job["Resource_List"]["walltime"]) == released["reserved_s"],
                    "A prior campaign capsule is unresolved")
            prior += released["reserved_s"]
        peers = {key: capsule.parent / label for key, label in NAMES.items() if key != role}
        for previous in peers.values():
            if (previous / "INTENT.json").exists():
                release = read(previous / "RELEASE.json")
                require(native(release["job"])["job_state"] == "F", "An existing retention arm is active")
                prior += release["reserved_s"]
        require(prior + BUDGET <= 320 * 3600, "Original campaign limit exceeded")
        active = subprocess.check_output([str(PBS / "qselect"), "-q", f"@{SERVER}", "-u", grant["account"]],
                                         text=True, timeout=45).split()
        for identifier in active:
            row = native(identifier)
            require(row["job_state"] == "F" or not row["Job_Name"].startswith(("slav4", "slav6")),
                    "Reconcile the active MAIN or SOURCE job before a new submission")

        binding = capsule / "BINDING.json"
        write_once(binding, dict(role=role, updates=ENDPOINTS[-1],
                                 core_release=dict(path=str(receipt), sha256=digest(receipt))))
        worker = capsule / "code/v6_retention_worker.py"
        inputs = capsule / "INPUTS.sha256"
        files = [receipt, binding, worker] + sorted(v4_code.glob("*.py"))
        with inputs.open("x") as stream:
            stream.writelines(f"{digest(p)}  {p}\n" for p in files)
        paths = (worker.parent, v4_code, source, source / "src", runtime / "data_dependencies",
                 runtime / "upstream", runtime / "packages")
        command = shlex.join(map(str, (runtime / "interpreter/bin/python3.12", "-B",
                                      worker, capsule, custody, short, bank)))
        name = NAMES[role]
        script = capsule / "job.pbs"
        with script.open("x") as stream:
            stream.write("\n".join(("#!/bin/bash -l", f"#PBS -N {name}", f"#PBS -P {PROJECT}", "#PBS -q auto",
                f"#PBS -l select={H200}", "#PBS -l walltime=01:00:00", "#PBS -m n", "#PBS -r n",
                f"#PBS -o {capsule / 'stdout.log'}", f"#PBS -e {capsule / 'stderr.log'}",
                "set -euo pipefail", "umask 077", f"cd {shlex.quote(str(capsule))}",
                "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                "export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false",
                f"export PBS_SERVER={shlex.quote(os.environ['PBS_SERVER'])}",
                f"export PYTHONPATH={shlex.quote(':'.join(map(str, paths)))}",
                "sha256sum --status -c INPUTS.sha256", "exec " + command, "")))
        intent = dict(role=role, name=name, budget_seconds=BUDGET, prior_upper_gpu_s=prior,
                      core_release_sha256=digest(receipt), program_sha256=digest(script),
                      inputs_sha256=digest(inputs))
        print(admit(capsule, intent, script, name, BUDGET, grant), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=tuple(FACTORS))
    for name in ("capsule", "custody", "short", "bank", "source", "runtime", "v4_code", "original",
                 "r0_first", "r0_retry"):
        parser.add_argument(name, type=Path)
    os.umask(0o077)
    submit(**vars(parser.parse_args()))
