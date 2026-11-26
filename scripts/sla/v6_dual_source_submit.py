"""Submit one V6 capsule with the original Hopper project's held admission."""

import argparse
from contextlib import closing
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from hopper_pbs import PBS, PROJECT, native, project_grant, wall_seconds
from scripts.pi0292.durable import OwnerLock
from scripts.pi0294.e1.journal import Journal
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import DEADLINE
from v6_dual_course import PARENT
from v6_dual_source_worker import STEPS, read, require


NAMES = {"probe": "slav6rdqual", "formal": "slav6rd1k"}
H200 = "1:ngpus=1:gpu_model=H200:ncpus=12:mem=225GB"
SERVER = "hopper-m-02"


def submit(mode, capsule, root, custody, short, bank, source, runtime, v4_code, corpus,
           original, r0_first, r0_retry, probe):
    budget = 3600 if mode == "probe" else 3 * 3600
    with closing(OwnerLock(capsule / "SUBMIT.lock")):
        require(not (root / "SITE_POLICY_HOLD.json").exists(),
                "Use the Core-reviewed retention course and staged Hopper release")
        require(not (capsule / "INTENT.json").exists() and not (capsule / "run").exists(),
                "Existing owner intent must be reconciled without resubmission")
        require(time.time() + budget < DEADLINE, "Original project deadline leaves insufficient time")
        ready = read(corpus / "READY.json")
        preflight = read(root / "PREFLIGHT.json")
        manifest = ready["manifest_sha256"]
        require(ready["hopper_readback"] and ready["threshold_for_v6_source_met"] and
                digest(corpus / "MANIFEST.private.json") == manifest and
                preflight["settings"]["source_extension_manifest"] == manifest and
                preflight["read_new_source_windows"] == 96 and preflight["optimizer_updates"] == 0,
                "The exact two-site released data and source host batch must be available")
        grant = project_grant(budget)
        required, balance = (grant["required_units_at_queue_rate_one"], grant["balance_gpu_units"])
        require(all(v is not None and math.isfinite(v) and v > 0
                    for v in (required, balance, grant["balance_su"])), "Live project grant unavailable")

        commitments = [original, r0_first, r0_retry]
        if mode == "formal":
            commitments.append(probe)
        prior = read(original / "INTENT.json")["prior_upper_gpu_s"]
        for entry in commitments:
            release = read(entry / "RELEASE.json")
            job = native(release["job"])
            if entry == probe:
                # A worker publishes its receipt immediately before PBS
                # records the terminal state.
                for _ in range(6):
                    if job["job_state"] == "F":
                        break
                    time.sleep(5)
                    job = native(release["job"])
            require(job["job_state"] == "F" and job["project"] == PROJECT
                    and job["Job_Owner"].split("@")[0] == grant["account"]
                    and int(job["Resource_List"]["ngpus"]) == 1
                    and wall_seconds(job["Resource_List"]["walltime"]) == release["reserved_s"],
                    "A prior campaign capsule is not reconciled")
            if entry in (original, probe):
                require(job["Exit_status"] == 0, "Expected a naturally closed scientific or qualification job")
            prior += release["reserved_s"]
        require(prior + budget <= 320 * 3600, "Original 320 GPU-hour budget exhausted")
        active = subprocess.check_output([str(PBS / "qselect"), "-q", f"@{SERVER}", "-u", grant["account"]],
                                         text=True, timeout=45).split()
        for identifier in active:
            row = native(identifier)
            require(row["job_state"] == "F" or not row["Job_Name"].startswith(("slav4", "slav6")),
                    "An existing Hopper experiment owner must be reconciled")

        binding = dict(mode=mode, manifest=manifest, parent=PARENT, steps=STEPS[mode], settings=preflight["settings"])
        if mode == "formal":
            result = probe / "run/TERMINAL.json"
            terminal = read(result)
            require(terminal["status"] == "PROBE_PASS"
                    and terminal["settings"] == preflight["settings"], "Expected identical qualification settings")
            binding["probe_terminal"] = dict(path=str(result), sha256=digest(result))
        write_once(capsule / "BINDING.json", binding)
        paths = (root / "code", v4_code, source, source / "src", runtime / "data_dependencies",
                 runtime / "upstream", runtime / "packages")
        command = shlex.join(map(str, (runtime / "interpreter/bin/python3.12", "-B",
            root / "code/v6_dual_source_worker.py", mode, capsule, custody, short, bank, corpus, manifest)))
        inputs = capsule / "INPUTS.sha256"
        files = sorted((root / "code").glob("*.py")) + [capsule / "BINDING.json", root / "PREFLIGHT.json",
                                                        corpus / "READY.json", corpus / "MANIFEST.private.json"]
        with inputs.open("x") as handle:
            handle.writelines(f"{digest(p)}  {p}\n" for p in files)
        name = NAMES[mode]
        pbs = capsule / "job.pbs"
        with pbs.open("x") as handle:
            handle.write("\n".join(("#!/bin/bash -l", f"#PBS -N {name}", f"#PBS -P {PROJECT}", "#PBS -q auto",
                f"#PBS -l select={H200}", f"#PBS -l walltime={budget // 3600:02d}:00:00", "#PBS -m n",
                "#PBS -r n", f"#PBS -o {capsule / 'stdout.log'}", f"#PBS -e {capsule / 'stderr.log'}",
                "set -euo pipefail", "umask 077", f"cd {shlex.quote(str(capsule))}",
                "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                "export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false",
                f"export PBS_SERVER={shlex.quote(os.environ['PBS_SERVER'])}",
                f"export PYTHONPATH={shlex.quote(':'.join(map(str, paths)))}", "sha256sum --status -c INPUTS.sha256",
                "exec " + command, "")))
        intent = dict(name=name, phase=mode, plan="d7c74953109374b69906036a0e2c6f39a32b3aa7",
                    prior_upper_gpu_s=prior, budget_seconds=budget, settings_sha256=digest(root / "PREFLIGHT.json"),
                    program_sha256=digest(pbs), inputs_sha256=digest(inputs))
        write_once(capsule / "INTENT.json", intent)
        journal = Journal(capsule / "journal", capsule.name)

        def hold():
            identifier = subprocess.check_output([str(PBS / "qsub"), "-f", "-h", "-q", f"auto@{SERVER}", str(pbs)],
                                                text=True, timeout=120).strip()
            require(re.fullmatch(r"\d+\.hopper-m-02", identifier), "Unknown effect, reconcile PBS")
            write_once(capsule / "JOB.json", dict(job=identifier, name=name))
            return identifier

        identifier = journal.once("qsub", intent, hold)
        row = native(identifier)
        resource = row["Resource_List"]
        rate = float(resource["queue_charge_rate"])
        charged = required * rate
        require(row["job_state"] == "H" and row["project"] == PROJECT and row["Job_Name"] == name
                and row["Job_Owner"].split("@")[0] == grant["account"] and resource["select"] == H200
                and int(resource["ngpus"]) == 1 and wall_seconds(resource["walltime"]) == budget
                and math.isfinite(rate) and rate > 0 and charged <= min(balance, grant["balance_su"]),
                "Held job resources or cost failed independent readback")
        admission = dict(job=identifier, charge_upper=charged, rate=rate, reserved_s=budget)
        write_once(capsule / "ADMITTED.json", admission)

        def release():
            subprocess.run([str(PBS / "qrls"), f"{identifier}@{SERVER}"], check=True, timeout=60)
            return admission

        write_once(capsule / "RELEASE.json", journal.once("release", admission, release))
        print(identifier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(NAMES))
    for name in ("capsule", "root", "custody", "short", "bank", "source", "runtime", "v4_code", "corpus",
                 "original", "r0_first", "r0_retry", "probe"):
        parser.add_argument(name, type=Path)
    submit(**vars(parser.parse_args()))
