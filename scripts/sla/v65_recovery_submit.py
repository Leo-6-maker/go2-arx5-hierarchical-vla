"""Admit the prepared V6.5 first 1K through the existing held PBS path."""

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
from v61_source_course import R4
from v61_source_submit import accounted_prior
from v61_source_worker import parent_gate
from v62_source_admission import write_capacity
from v63_adapt_worker import PARENT_RELEASE, SOURCE_RELEASE
from v65_recovery_course import END, PLAN
from v65_recovery_worker import PROBE
from v6_dual_source_submit import H200, PROJECT, SERVER, admit


NAME, BUDGET = "slav65r3a1k", 10800


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def submit(root, output):
    base, code, incoming = root.parent, root / "code", root / "incoming"
    old_adapt = base / "adapt-v63-20261127-01"
    old_source = base / "source-v62-20261127-01"
    data = incoming / "DATA.json"
    config = read(data)
    paths = config["parent_paths"]
    custody, r4 = (Path(paths[k]) for k in ("custody", "r4"))
    runtime = custody / "sla-core-hopper-v1-runtime-20261121-01/runtime.verified"
    source = custody / "qualification-source-b9581196/source"
    v4 = base / "source-v61-20261127-01/v4-local"
    bank = incoming / "BANK/ACTUAL.json"
    batch = incoming / "HOPPER-FIRST-BATCH.json"
    recovery = incoming / "R3-HOPPER-READBACK.json"
    parent = old_adapt / "incoming/PARENT_RELEASE.json"
    hold = old_adapt / "incoming/SOURCE0250-RELEASE.json"
    earliest = custody.parent / "SLA-HOPPER-AUTONOMOUS-V4-MAIN1K-20261126-01"
    trials = [old_source / p for p in ("QUALIFICATION", "QUALIFICATION-R2", "QUALIFICATION-R3", "QUALIFICATION-R4")]
    prior = [earliest, *(custody.parent / ("SLA-HOPPER-AUTONOMOUS-V4-TRACE-R0-20261126-" + i) for i in ("01", "02"))]
    prior.extend(base / "retention-20261126-01" / x for x in ("slav6ret1", "slav6ret2", "slav6rref", "slav6r5rel"))
    h30 = old_source / "SCVLA-H30-QUALIFICATION-01"
    source_formal = old_source / "SOURCE250"
    old_probe, old_formal = (old_adapt / p for p in ("A250", "A250-FORMAL-R1"))
    prior.extend((*trials, source_formal, old_probe, old_formal, h30))
    require(time.time() + BUDGET < DEADLINE and output.parent.resolve() == custody.parent.resolve(),
            "The original project scratch and allocation deadline must cover V6.5")
    with closing(OwnerLock(root / "SUBMIT.lock")):
        require(not output.exists(), "Resolve the existing V6.5 intent and its PBS owner before dispatch")
        packed, checked, course = read(recovery), read(batch), read(bank)
        require(packed["manifest_sha256"] == config["recovery_manifest_sha256"] == course["recovery_manifest_sha256"]
                and packed["windows"] == 115 and packed["supported_action_points"] == 4420
                and checked["plan"] == course["plan"] == PLAN and sha(bank) == checked["bank"]
                and checked["actual_shapes"] == {"target": [96,40,32], "robodojo": [16,50,32], "libero": [16,10,32]}
                and checked["auxiliary_shape"] == [4, 40] and checked["optimizer_visits"] == packed["optimizer_visits"] == 0,
                "Use the actual bank and CPU training readback")
        pin(parent, PARENT_RELEASE)
        parent_gate(ref(parent))
        require(not read(pin(hold, SOURCE_RELEASE))["released"], "Keep the old SOURCE500 hold")
        adapted, old_s = (read(p / "run/TERMINAL.json") for p in (old_formal, source_formal))
        require(adapted["status"] == "FORMAL_ADAPT_A250_BOUNDARY" and old_s["status"] == "FORMAL_SOURCE_BOUNDARY"
                and adapted["updates"] == old_s["updates"] == 250, "Original formal ledger must reconcile")
        logs = [p / "run/UPDATES.jsonl" for p in trials] + [old_probe / "qualification/UPDATES.jsonl"]
        used = sum(len(log.read_text().splitlines()) for log in logs if log.is_file())
        h30_updates = len(list((h30 / "run").glob("*.RETURNED.json")))
        require(h30_updates == read(h30 / "run/TERMINAL.json")["updates"] == 3, "Resolve H30 engineering usage")
        used += h30_updates
        require(used <= 24-PROBE, "The inherited engineering allowance cannot cover the qualified run")
        try:
            grant = project_grant(BUDGET)
        except BalanceLoginExpired:
            source_config = read(source_formal / "CONFIG.json")
            lineage = Path(source_config["project_lineage"])
            grant = project_period(source_config["project_basis"], lineage, BUDGET)
            grant["lineage"] = str(lineage)
        accrued = accounted_prior(prior, earliest, grant)
        require(accrued+BUDGET <= 320*3600, "The original GPU-hour allowance cannot cover this PBS request")
        active = subprocess.check_output([str(PBS/"qselect"), "-q", f"@{SERVER}", "-u", grant["account"]],
                                         text=True, timeout=45).split()
        require(all(not native(job)["Job_Name"].startswith(("slav4", "slav6")) for job in active),
                "A previous project GPU owner is active or unresolved")
        output.mkdir(mode=0o700)
        write_once(output / "WRITE_CAPACITY.json", write_capacity(output, r4, "1000", 0, END, BUDGET))
        write_once(output / "GRANT.json", grant)
        declaration = output / "BINDING.json"
        binding = dict(plan=PLAN, formal_end=END, r4_receipt=R4, parent=ref(parent), source_hold=ref(hold),
                       data=ref(data), bank=ref(bank), host_batch=ref(batch), recovery=ref(recovery),
                       engineering_updates=PROBE, engineering_remaining=24-used, formal_prior_updates=500,
                       site_grant=ref(output / "GRANT.json"), budget_seconds=BUDGET)
        write_once(declaration, binding)
        cold_readers = [old_adapt / f"v64-a{v}-reader-code-01" for v in (78,83,87,93)]
        hot_code = old_adapt / "code-formal-r1"
        inputs = output / "INPUTS.sha256"
        pinned = (declaration, data, bank, batch, recovery, parent, hold, output / "GRANT.json",
                  incoming / "BANK/BANK.npz", incoming / "R3/MANIFEST.json")
        roots = (code, *cold_readers, hot_code, v4)
        with inputs.open("x") as stream:
            stream.writelines(f"{sha(p)}  {p}\n" for p in (*pinned, *(p for directory in roots for p in sorted(directory.glob("*.py")))))
        package_paths = (source, source/"src", runtime/"data_dependencies", runtime/"upstream", runtime/"packages")
        script = output / "job.pbs"
        command = shlex.join(map(str, (runtime/"interpreter/bin/python3.12", "-B", code/"v65_recovery_worker.py", output)))
        with script.open("x") as stream:
            stream.write("\n".join(("#!/bin/bash -l", f"#PBS -N {NAME}", f"#PBS -P {PROJECT}", "#PBS -q auto",
                 f"#PBS -l select={H200}", "#PBS -l walltime=03:00:00", "#PBS -m n", "#PBS -r n",
                 f"#PBS -o {output/'stdout.log'}", f"#PBS -e {output/'stderr.log'}", "set -euo pipefail", "umask 077",
                 "cd "+shlex.quote(str(output)), "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
                 "export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false", f"export PBS_SERVER={SERVER}",
                 "export PYTHONPATH="+shlex.quote(":".join(map(str, (*roots, *package_paths)))),
                 "sha256sum --status -c INPUTS.sha256", "exec "+command, "")))
        intent = dict(plan=PLAN, name=NAME, budget_seconds=BUDGET, prior_upper_gpu_s=accrued,
                      binding_sha256=sha(declaration), program_sha256=sha(script), inputs_sha256=sha(inputs))
        print(admit(output, intent, script, NAME, BUDGET, grant), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    os.umask(0o077)
    submit(**vars(cli.parse_args()))
