"""Submit the qualified A74 first interval using the existing held PBS journal."""

import argparse
from contextlib import closing
import os
from pathlib import Path
import shlex
import subprocess
import time

from hopper_pbs import BalanceLoginExpired, PBS, native, project_grant
from hopper_project_admission import project_period
from m38_gpu_preflight_v1 import read, require, sha
from scripts.pi0292.durable import OwnerLock
from scripts.sla.hopper_parent_custody import write_once
from scripts.sla.hopper_site import DEADLINE
from v61_source_course import R4
from v61_source_submit import accounted_prior
from v62_source_admission import write_capacity
from v63_adapt_worker import FIRST, PLAN, PROBE_STEPS, validate_boundary
from v6_dual_source_submit import H200, PROJECT, SERVER, admit


NAME = "slav63a250"
BUDGET = 7200


def ref(path):
    return dict(path=str(path),sha256=sha(path))


def submit(root, *, after_probe=False):
    paths = root / "incoming/PATHS.json"
    config = read(paths)
    incoming = paths.parent
    code = root / ("code-formal-r1" if after_probe else "code")
    custody, r4, bank = (Path(config[k]) for k in ("custody","r4","adapt_bank"))
    runtime = custody / "sla-core-hopper-v1-runtime-20261121-01/runtime.verified"
    source = custody / "qualification-source-b9581196/source"
    previous_root = root.parent / "source-v62-20261127-01"
    earlier = root.parent / "source-v61-20261127-01"
    owner_trials = [previous_root / p for p in ("QUALIFICATION","QUALIFICATION-R2","QUALIFICATION-R3","QUALIFICATION-R4")]
    source_formal = previous_root / "SOURCE250"
    base = custody.parent
    original = base / "SLA-HOPPER-AUTONOMOUS-V4-MAIN1K-20261126-01"
    prior = [original,base/"SLA-HOPPER-AUTONOMOUS-V4-TRACE-R0-20261126-01",
             base/"SLA-HOPPER-AUTONOMOUS-V4-TRACE-R0-20261126-02"]
    prior.extend(root.parent / "retention-20261126-01" / x
                 for x in ("slav6ret1","slav6ret2","slav6rref","slav6r5rel"))
    prior.extend((*owner_trials,source_formal))
    capsule = root / ("A250-FORMAL-R1" if after_probe else "A250")
    require(time.time()+BUDGET < DEADLINE, "The original allocation deadline must cover this job")
    with closing(OwnerLock(root / "SUBMIT.lock")):
        require(not capsule.exists(), "Reconcile the existing A250 capsule before re-submitting")
        host = incoming / "HOPPER_HOST_READBACK.json"
        checked = read(host)
        bsha = sha(bank / "ACTUAL.json")
        require(checked["plan"] == PLAN and checked["bank_sha256"] == bsha
                and checked["target_windows"] == 6544
                and checked["reader_population"] == {"base":4450,"legacy":1650,"newest":444}
                and checked["actual_batch_shapes"] == {"target":[64,40,32],"single":[16,40,32],
                     "robodojo":[32,50,32],"libero":[16,10,32]}
                and checked["auxiliary_shape"] == [4,40]
                and checked["optimizer_visits"] == 0, "First true two-site training batch is required")
        used = 0
        for job in owner_trials:
            ledger = job / "run/UPDATES.jsonl"
            if ledger.exists():
                used += len(ledger.read_text().splitlines())
        previous = None
        if after_probe:
            first = root / "A250"
            old_job = native(read(first / "RELEASE.json")["job"])
            qualification = first / "qualification/TERMINAL.json"
            proof = read(qualification)
            log = first / "qualification/UPDATES.jsonl"
            lines = log.read_text().splitlines()
            require(old_job["job_state"] == "F" and old_job["Job_Name"] == NAME and not (first / "run").exists()
                    and proof["status"] == "QUALIFICATION_INTERRUPTED" and proof["restore"]["prediction_equal"]
                    and proof["updates"] == len(lines) == PROBE_STEPS and proof["course"]["bank_sha256"] == bsha
                    and "Measured cold qualification leaves insufficient time for A250" in (first / "stderr.log").read_text(),
                    "Only the closed, restored engineering stage can be reused without replay")
            previous = ref(qualification)
            prior.append(first)
            used += len(lines)
        require(0 <= used <= 24-(0 if after_probe else PROBE_STEPS),
                "The actual existing engineering budget has been consumed")
        formal = read(source_formal / "run/TERMINAL.json")
        require(formal["status"] == "FORMAL_SOURCE_BOUNDARY" and formal["updates"] == 250
                and read(source_formal / "RELEASE.json")["job"] == "736211.hopper-m-02",
                "The existing actual SOURCE cost has not been reconciled")
        try:
            grant = project_grant(BUDGET)
        except BalanceLoginExpired:
            previous = read(source_formal / "CONFIG.json")
            lineage = Path(previous["project_lineage"])
            grant = project_period(previous["project_basis"],lineage,BUDGET)
            grant["lineage"] = str(lineage)
        accrued = accounted_prior(prior,original,grant)
        require(accrued+BUDGET <= 320*3600, "This job exceeds the inherited GPU-hour ceiling")
        active = subprocess.check_output([str(PBS/"qselect"),"-q",f"@{SERVER}","-u",grant["account"]],
                                         text=True,timeout=45).split()
        require(all(not native(job)["Job_Name"].startswith(("slav4","slav6")) for job in active),
                "An older project training owner is still active")
        binding = dict(plan=PLAN,end=FIRST,budget_seconds=BUDGET,r4_receipt=R4,
                       engineering_updates=PROBE_STEPS,engineering_upper_left=24-used,
                       formal_prior_updates=250,inputs=ref(paths),bank_sha256=bsha,host_readback=ref(host),
                       parent=ref(incoming/"PARENT_RELEASE.json"),source_hold=ref(incoming/"SOURCE0250-RELEASE.json"))
        if previous:
            binding["previous_qualification"] = previous
        validate_boundary(binding)
        capsule.mkdir(mode=0o700)
        write_once(capsule/"WRITE_CAPACITY.json",write_capacity(capsule,r4,"250",0,FIRST,BUDGET))
        write_once(capsule/"GRANT.json",grant)
        binding["site_grant"] = ref(capsule/"GRANT.json")
        declaration = capsule/"BINDING.json"
        write_once(declaration,binding)
        authorized = [declaration,paths,host,incoming/"PARENT_RELEASE.json",incoming/"SOURCE0250-RELEASE.json",
                      capsule/"GRANT.json",bank/"ACTUAL.json",*sorted(code.glob("*.py")),
                      *sorted((earlier/"v4-local").glob("*.py"))]
        if previous:
            authorized.append(Path(previous["path"]))
        inputs = capsule/"INPUTS.sha256"
        with inputs.open("x") as handle:
            handle.writelines(f"{sha(p)}  {p}\n" for p in authorized)
        import_paths = (code,earlier/"v4-local",source,source/"src",runtime/"data_dependencies",
                        runtime/"upstream",runtime/"packages")
        python = runtime/"interpreter/bin/python3.12"
        command = shlex.join(map(str,(python,"-B",code/"v63_adapt_worker.py",capsule)))
        script = capsule/"job.pbs"
        with script.open("x") as out:
            out.write("\n".join(("#!/bin/bash -l",f"#PBS -N {NAME}",f"#PBS -P {PROJECT}","#PBS -q auto",
                      f"#PBS -l select={H200}","#PBS -l walltime=02:00:00","#PBS -m n","#PBS -r n",
                      f"#PBS -o {capsule/'stdout.log'}",f"#PBS -e {capsule/'stderr.log'}","set -euo pipefail","umask 077",
                      "cd "+shlex.quote(str(capsule)),
                      "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
                      "export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false",f"export PBS_SERVER={SERVER}",
                      "export PYTHONPATH="+shlex.quote(":".join(map(str,import_paths))),
                      "sha256sum --status -c INPUTS.sha256","exec "+command,"")))
        intent = dict(plan=PLAN,name=NAME,budget_seconds=BUDGET,prior_upper_gpu_s=accrued,
                      binding_sha256=sha(declaration),program_sha256=sha(script),inputs_sha256=sha(inputs))
        print(admit(capsule,intent,script,NAME,BUDGET,grant),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--after-probe",action="store_true")
    os.umask(0o077)
    submit(**vars(parser.parse_args()))
