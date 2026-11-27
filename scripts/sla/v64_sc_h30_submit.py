"""Submit the once-only V6.4 isolated H30 engineering allocation."""

import argparse
from contextlib import closing
import os
from pathlib import Path
import shlex
import subprocess
import time

from hopper_pbs import BalanceLoginExpired, PBS, native, project_grant
from hopper_project_admission import project_period
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock
from scripts.sla.hopper_parent_custody import write_once
from scripts.sla.hopper_site import DEADLINE
from v61_source_submit import accounted_prior
from v62_source_admission import write_capacity
from v64_sc_h30_qualification import A250_TERMINAL, DATA_MEMBERS, HOST_READER, PLAN, UPDATES
from v6_dual_source_submit import H200, PROJECT, SERVER, admit


NAME = "slav64h30q"
WALL = 3600


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def submit(root, code):
    previous = root / "adapt-v63-20261127-01"
    source_v62 = root / "source-v62-20261127-01"
    source_v61 = root / "source-v61-20261127-01"
    src250, a250 = source_v62 / "SOURCE250", previous / "A250-FORMAL-R1"
    capsule = source_v62 / "SCVLA-H30-QUALIFICATION-01"
    data = source_v62 / "SCVLA-H30-PORTABLE-01"
    paths_file = previous / "incoming/PATHS.json"
    paths = read(paths_file)
    custody = Path(paths["custody"])
    runtime = custody / "sla-core-hopper-v1-runtime-20261121-01/runtime.verified"
    source = custody / "qualification-source-b9581196/source"
    import_paths = (code, data / "code", previous / "code-formal-r1", source_v61 / "v4-local",
                    source, source / "src", runtime / "data_dependencies", runtime / "upstream", runtime / "packages")
    require(all(path.is_dir() for path in import_paths) and time.time() + WALL < DEADLINE,
            "The current allocation must cover a complete H30 qualification")
    with closing(OwnerLock(source_v62 / "H30-SUBMIT.lock")):
        require(not capsule.exists(), "Reconcile the original H30 allocation before re-submitting")
        a250_result = read(pin(a250 / "run/TERMINAL.json", A250_TERMINAL))
        require(a250_result["status"] == "FORMAL_ADAPT_A250_BOUNDARY" and a250_result["updates"] == 250
                and read(a250 / "RELEASE.json")["job"] == "736530.hopper-m-02", "The full A250 boundary is required")
        host = read(pin(data / "HOPPER-CPU-TRAIN-READBACK.json", HOST_READER))
        members = pin(data / "DATA-SHA256SUMS", DATA_MEMBERS)
        require(host["actual_verified_batches"] == 3 and host["received_source_train_episodes"] == 51
                and host["verified_file_manifest_sha256"] == DATA_MEMBERS and host["optimizer_visits"] == 0,
                "Use the existing real SC-VLA reader on Hopper")
        owner_trials = [source_v62 / name for name in ("QUALIFICATION", "QUALIFICATION-R2",
                                                       "QUALIFICATION-R3", "QUALIFICATION-R4")]
        ledgers = [trial / "run/UPDATES.jsonl" for trial in owner_trials]
        ledgers.append(previous / "A250/qualification/UPDATES.jsonl")
        used = sum(len(path.read_text().splitlines()) for path in ledgers if path.exists())
        require(used == 12 and used + UPDATES <= 24, "The actual V6 engineering ledger changed")
        require(read(src250 / "run/TERMINAL.json")["updates"] == 250,
                "Include the original SOURCE training cost")
        try:
            grant = project_grant(WALL)
        except BalanceLoginExpired:
            old = read(src250 / "CONFIG.json")
            lineage = Path(old["project_lineage"])
            grant = project_period(old["project_basis"], lineage, WALL)
            grant["lineage"] = str(lineage)
        original = custody.parent / "SLA-HOPPER-AUTONOMOUS-V4-MAIN1K-20261126-01"
        prior = [original, *(custody.parent / f"SLA-HOPPER-AUTONOMOUS-V4-TRACE-R0-20261126-0{i}"
                               for i in (1, 2))]
        prior.extend(root / "retention-20261126-01" / x for x in ("slav6ret1", "slav6ret2", "slav6rref", "slav6r5rel"))
        prior.extend((*owner_trials, src250, previous / "A250", a250))
        accrued = accounted_prior(prior, original, grant)
        require(accrued + WALL <= 320 * 3600, "The original GPU hour ceiling is exhausted")
        active = subprocess.check_output([str(PBS / "qselect"), "-q", f"@{SERVER}", "-u", grant["account"]],
                                         text=True, timeout=45).split()
        require(all(not native(job)["Job_Name"].startswith(("slav4", "slav6")) for job in active),
                "Another project optimizer owner is still active")

        capsule.mkdir(mode=0o700)
        grant_file = capsule / "GRANT.json"
        write_once(grant_file, grant)
        write_once(capsule / "WRITE_CAPACITY.json", write_capacity(capsule, Path(paths["r4"]), "probe", 0, UPDATES, WALL))
        binding = capsule / "BINDING.json"
        write_once(binding, dict(plan=PLAN, engineering_used=used, engineering_limit=UPDATES, budget_seconds=WALL,
                                 paths=ref(paths_file), data=str(data), data_members=ref(members),
                                 A250=ref(a250 / "run/TERMINAL.json"), site_grant=ref(grant_file),
                                 formal_source608=False))
        inputs = capsule / "INPUTS.sha256"
        authorized = [binding, grant_file, paths_file, members, data / "HOPPER-CPU-TRAIN-READBACK.json",
                      a250 / "run/TERMINAL.json", code / Path(__file__).name, *sorted(code.glob("v*_qualification.py")),
                      code / "v6_dual_course.py", *sorted((data / "code").glob("*.py"))]
        with inputs.open("x") as stream:
            stream.writelines(f"{sha(path)}  {path}\n" for path in authorized)
        python = runtime / "interpreter/bin/python3.12"
        worker = code / "v64_sc_h30_qualification.py"
        train = shlex.join(map(str, (python, "-B", worker, "train", capsule)))
        restore = shlex.join(map(str, (python, "-B", worker, "readback", capsule)))
        script = capsule / "job.pbs"
        with script.open("x") as out:
            out.write("\n".join(("#!/bin/bash -l", f"#PBS -N {NAME}", f"#PBS -P {PROJECT}", "#PBS -q auto",
                f"#PBS -l select={H200}", "#PBS -l walltime=01:00:00", "#PBS -m n", "#PBS -r n",
                f"#PBS -o {capsule/'stdout.log'}", f"#PBS -e {capsule/'stderr.log'}",
                "set -euo pipefail", "umask 077", "cd " + shlex.quote(str(capsule)),
                "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
                "export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false", f"export PBS_SERVER={SERVER}",
                "export PYTHONPATH=" + shlex.quote(":".join(map(str, import_paths))),
                "sha256sum --status -c INPUTS.sha256", train, "export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=''",
                "exec " + restore, "")))
        intent = dict(plan=PLAN, name=NAME, budget_seconds=WALL, prior_upper_gpu_s=accrued,
                      binding_sha256=sha(binding), program_sha256=sha(script), inputs_sha256=sha(inputs))
        print(admit(capsule, intent, script, NAME, WALL, grant), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("code", type=Path)
    os.umask(0o077)
    submit(**vars(parser.parse_args()))
