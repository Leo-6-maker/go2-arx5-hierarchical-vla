"""Submit the fixed MAIN capsule once using Hopper's existing PBS journal."""

import argparse
from contextlib import closing
import json
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


EARLIER_NAMES = ("slav4m250a", "slav4m250b")
NAME = "slav4m250c"
WALL = 6 * 3600
PLAN = "f72f3d57b470de3e62821cd5e1a33d938a669cb7"


def submit(capsule,custody,source,runtime):
    with closing(OwnerLock(capsule/"SUBMIT.lock")):
        if (capsule/"INTENT.json").exists():
            raise RuntimeError("Existing submission intent must be reconciled in PBS")
        if time.time()+WALL>=DEADLINE:
            raise PermissionError("Original experiment deadline leaves insufficient time")
        grant=project_grant(WALL)
        required = grant.get("required_units_at_queue_rate_one")
        balance = grant.get("balance_gpu_units")
        if required is None or balance is None:
            raise PermissionError("Use the charge-aware site grant helper before submitting to PBS")
        if not all(math.isfinite(value) and value > 0 for value in (required, balance, grant["balance_su"])):
            raise PermissionError("Held admission requires finite project balances and rate")
        old=[json.loads(f.read_bytes()) for f in (custody/"HOPPER-ADAPT20K-V1/allocations").glob("*/RESERVATION.json")]
        legacy=[x["job"] for x in old if "reserved_gpu_count" not in x]
        scheduler=json.loads((capsule/"LEGACY_SCHEDULER_READBACK.json").read_bytes())
        if set(scheduler)!=set(legacy) or any(scheduler[j]["job_state"]!="F" for j in legacy):
            raise PermissionError("Missing original one-GPU diagnostic settlement")
        prior=0
        for reservation in old:
            devices=reservation.get("reserved_gpu_count")
            if devices is None:
                devices=scheduler[reservation["job"]]["ngpus"]
            prior+=int(reservation["reserved_seconds"])*int(devices)
        # Keep the two independently registered A/B allocations in the shared upper bound.
        for arm in ("A","B"):
            f=custody/"MAIN3800-AUXILIARY-COMPARISON-20261122-02"/arm/"RESERVATION.json"
            reservation=json.loads(f.read_bytes())
            prior+=int(reservation["reserved_seconds"])*int(reservation["reserved_gpu_count"])
        previous_path = capsule / "PREVIOUS_MAIN_READBACK.json"
        previous = json.loads(previous_path.read_bytes())
        if (previous["job_state"] != "F" or previous["job_name"] != EARLIER_NAMES[-1]
                or previous["exit_status"] != 1 or previous["updates"] != 0
                or previous["gpus"] != 1):
            raise PermissionError("The predecessor was not settled as a zero-update technical attempt")
        prior += previous["reserved_wall_seconds"] * previous["gpus"]
        if prior+WALL>320*3600:
            raise PermissionError("Original finite campaign GPU budget exhausted")

        existing=[]
        for name in (*EARLIER_NAMES, NAME):
            existing.extend(subprocess.check_output([str(PBS/"qselect"),"-u",grant["account"],"-N",name],
                                                    text=True,timeout=45).split())
        if existing or (capsule/"run").exists():
            raise RuntimeError("An existing MAIN owner must be reconciled")
        assets=sorted(p for p in (capsule/"incoming").rglob("*") if p.is_file())
        assets.extend((capsule/"LEGACY_SCHEDULER_READBACK.json", previous_path))
        staged=sorted((capsule/"code").glob("*.py"))
        manifests={str(p.relative_to(capsule)):digest(p) for p in assets+staged}
        if not staged or "incoming/reference/V3_RUN_CONTRACT.json" not in manifests:
            raise ValueError("Missing bound original command or learner assets")
        check=capsule/"INPUTS.sha256"
        with check.open("x") as f:
            f.writelines(f"{value}  {name}\n" for name,value in manifests.items())
        paths=":".join(map(str,(capsule/"code",source,source/"src",runtime/"data_dependencies",
                                     runtime/"upstream",runtime/"packages")))
        command=shlex.join([str(runtime/"interpreter/bin/python3.12"),"-B",
            str(capsule/"code/v4_main250_hopper.py"),str(capsule),str(custody)])
        script=capsule/"job.pbs"
        with script.open("x") as f:
            f.write("\n".join(["#!/bin/bash -l",f"#PBS -N {NAME}",f"#PBS -P {PROJECT}","#PBS -q auto",
                "#PBS -l select=1:ngpus=1:gpu_model=H200:ncpus=12:mem=225GB","#PBS -l walltime=06:00:00",
                "#PBS -m n","#PBS -r n",f"#PBS -o {capsule/'stdout.log'}",f"#PBS -e {capsule/'stderr.log'}",
                "set -euo pipefail","umask 077",f"cd {shlex.quote(str(capsule))}",
                "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                "export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "export PBS_SERVER="+shlex.quote(os.environ["PBS_SERVER"]),
                f"export PYTHONPATH={shlex.quote(paths)}","sha256sum --status -c INPUTS.sha256","exec "+command,""]))
        intent=dict(plan=PLAN,name=NAME,project=PROJECT,wall_s=WALL,prior_upper_gpu_s=prior,
                    grant=grant,batch_sha256=digest(script),inputs_sha256=digest(check))
        write_once(capsule/"INTENT.json",intent)
        journal=Journal(capsule/"journal",capsule.name)

        def hold():
            job=subprocess.check_output([str(PBS/"qsub"),"-f","-h",str(script)],text=True,timeout=120).strip()
            if not re.fullmatch(r"\d+\.hopper-m-02",job):
                raise RuntimeError("Unknown submission effect, stop and reconcile")
            write_once(capsule/"JOB.json",dict(job=job,name=NAME))
            return job

        job=journal.once("qsub",intent,hold)
        row=native(job)
        spec=row["Resource_List"]
        rate=float(spec["queue_charge_rate"])
        charge=required*rate
        if (row["job_state"]!="H" or row["project"]!=PROJECT or row["Job_Name"]!=NAME
                or row["Job_Owner"].split("@")[0]!=grant["account"] or int(spec["ngpus"])!=1
                or spec["select"]!="1:ngpus=1:gpu_model=H200:ncpus=12:mem=225GB"
                or wall_seconds(spec["walltime"])!=WALL or not math.isfinite(rate) or rate<=0
                or charge>min(balance,grant["balance_su"])):
            raise PermissionError("Held job resources or charge differ from the live grant")
        admission=dict(job=job,charge_upper=charge,rate=rate,reserved_s=WALL)
        write_once(capsule/"ADMITTED.json",admission)

        def release():
            subprocess.run([str(PBS/"qrls"),job],check=True,timeout=60)
            return admission

        write_once(capsule/"RELEASE.json",journal.once("release",admission,release))
        print(json.dumps(admission))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for arg in ("capsule","custody","source","runtime"):
        cli.add_argument(arg,type=Path)
    os.umask(0o077)
    submit(**vars(cli.parse_args()))
