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
SUPERSEDED_LONG_PROBE = "slav4longq1"
LONG_PROBE = "slav4longq2"
LONG_PROBE_RECOVERY = "slav4longq3"
LONG_FORMAL = "slav4main1k"
LONG_TRACE = "slav4trace"


def submit(capsule,custody,source,runtime,*,short_run=None,bank_root=None,completed_probe=None,prior_probe=None,profile_of=None,milestone=None):
    long_run = short_run is not None
    formal = completed_probe is not None
    if long_run != (bank_root is not None) or (formal or prior_probe is not None) and not long_run:
        raise ValueError("Bind both the exact completed short run and the real long data")
    profiling = profile_of is not None
    if profiling != (milestone is not None) or profiling and (not formal or prior_probe is not None):
        raise ValueError("A profile needs the independently measured probe and the exact formal parent")
    if profiling:
        name = LONG_TRACE
    elif formal:
        name = LONG_FORMAL
    elif prior_probe is not None:
        name = LONG_PROBE_RECOVERY
    else:
        name = LONG_PROBE if long_run else NAME
    wall = 2 * 3600 if long_run and not formal else WALL
    if profiling:
        wall = 3600
    elif formal:
        from v4_long_train import UPDATES, select

        charter = json.loads((capsule / "incoming/FORMAL_BINDING.json").read_bytes())
        selected = select(completed_probe, charter["performance_sha256"])
        if charter["updates"] != UPDATES or charter["profile"] != selected["profile"]:
            raise PermissionError("Formal duration and measured batch must be pinned")
        needed = UPDATES * selected["p95_s"] + 1800
        wall = min(WALL, max(3600, math.ceil(needed / 900) * 900))
        if wall < needed:
            raise PermissionError("Observed training plus checkpoint reserve requires a segmented job")
    wall_label = f"{wall//3600:02d}:{wall//60%60:02d}:00"
    with closing(OwnerLock(capsule/"SUBMIT.lock")):
        if (capsule/"INTENT.json").exists():
            raise RuntimeError("Existing submission intent must be reconciled in PBS")
        if time.time()+wall>=DEADLINE:
            raise PermissionError("Original experiment deadline leaves insufficient time")
        grant=project_grant(wall)
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
        assets=sorted(p for p in (capsule/"incoming").rglob("*") if p.is_file())
        if long_run:
            accepted_path = capsule / "COMPLETED_MAIN_READBACK.json"
            accepted = json.loads(accepted_path.read_bytes())
            endpoint = json.loads((short_run / "run/TRAIN_TERMINAL.json").read_bytes())
            if (accepted["job_state"] != "F" or accepted["job_name"] != NAME
                    or accepted["exit_status"] != 0 or accepted["gpus"] != 1
                    or not endpoint["formal_endpoint"] or endpoint["completed_updates"] != 250
                    or accepted["terminal_sha256"] != digest(short_run / "run/TRAIN_TERMINAL.json")):
                raise PermissionError("Read the completed short learner before admitting the long probe")
            prior += accepted["reserved_wall_seconds"] * accepted["gpus"]
            assets.append(accepted_path)
        if formal:
            release = json.loads((completed_probe / "RELEASE.json").read_bytes())
            probe = native(release["job"])
            if (probe["Job_Name"] not in (LONG_PROBE, LONG_PROBE_RECOVERY) or probe["job_state"] != "F"
                    or probe["Exit_status"] != 0 or probe["project"] != PROJECT
                    or probe["Job_Owner"].split("@")[0] != grant["account"]
                    or int(probe["Resource_List"]["ngpus"]) != 1
                    or wall_seconds(probe["Resource_List"]["walltime"]) != release["reserved_s"]):
                raise PermissionError("The measured probe must have an exact terminal scheduler readback")
            prior += release["reserved_s"]
            previous_probe = completed_probe / "PREVIOUS_PROBE_READBACK.json"
            if previous_probe.exists():
                prior += json.loads(previous_probe.read_bytes())["wall_s"]
        if profiling:
            from v4_long_profile import START, TRACED, WARM

            binding = json.loads((capsule / "incoming/PROFILE_BINDING.json").read_bytes())
            terminal_path = profile_of / "run/TRAIN_TERMINAL.json"
            terminal = json.loads(terminal_path.read_bytes())
            actual = json.loads((profile_of / "RELEASE.json").read_bytes())
            row = native(actual["job"])
            if (row["job_state"] != "F" or row["Exit_status"] != 0 or row["Job_Name"] != LONG_FORMAL
                    or row["project"] != PROJECT or row["Job_Owner"].split("@")[0] != grant["account"]
                    or int(row["Resource_List"]["ngpus"]) != 1 or not terminal["formal_endpoint"]
                    or terminal["completed_updates"] != 1000 or binding["main_terminal_sha256"] != digest(terminal_path)
                    or binding["milestone_sha256"] != digest(milestone)
                    or binding["replay_start"] != START or binding["traced"] != TRACED
                    or binding["warmup"] != WARM):
                raise PermissionError("Wait for the exact original 1K scheduler terminal and fixed trace admission")
            old_engineering = json.loads((completed_probe / "run/ACCEPTED.json").read_bytes())["engineering_updates"]
            if old_engineering + WARM + TRACED > 56:
                raise PermissionError("The trace exceeds the existing finite engineering scope")
            prior += actual["reserved_s"]
        if prior_probe is not None:
            release = json.loads((prior_probe / "RELEASE.json").read_bytes())
            probe = native(release["job"])
            if (probe["Job_Name"] != LONG_PROBE or probe["job_state"] != "F"
                    or probe["Exit_status"] == 0 or probe["project"] != PROJECT
                    or probe["Job_Owner"].split("@")[0] != grant["account"]
                    or int(probe["Resource_List"]["ngpus"]) != 1):
                raise PermissionError("Resolve the prior probe before any technical continuation")
            prior += release["reserved_s"]
            readback = capsule / "PREVIOUS_PROBE_READBACK.json"
            write_once(readback, dict(job=release["job"], wall_s=release["reserved_s"],
                       measured_sha256=digest(prior_probe / "run/PROFILES.json")))
            assets.append(readback)
        if prior+wall>320*3600:
            raise PermissionError("Original finite campaign GPU budget exhausted")

        existing=[]
        active = subprocess.check_output([str(PBS/"qselect"),"-u",grant["account"]],
                                         text=True,timeout=45).split()
        for job in active:
            row = native(job)
            if row["Job_Name"] in (*EARLIER_NAMES, NAME, SUPERSEDED_LONG_PROBE, LONG_PROBE, LONG_PROBE_RECOVERY, LONG_FORMAL, LONG_TRACE) and row["job_state"] != "F":
                existing.append(job)
        if existing or (capsule/"run").exists():
            raise RuntimeError("An existing MAIN owner must be reconciled")
        assets.extend((capsule/"LEGACY_SCHEDULER_READBACK.json", previous_path))
        staged=sorted((capsule/"code").glob("*.py"))
        manifests={str(p.relative_to(capsule)):digest(p) for p in assets+staged}
        if not staged or (not long_run and "incoming/reference/V3_RUN_CONTRACT.json" not in manifests):
            raise ValueError("Missing bound original command or learner assets")
        check=capsule/"INPUTS.sha256"
        with check.open("x") as f:
            f.writelines(f"{value}  {name}\n" for name,value in manifests.items())
        paths=":".join(map(str,(capsule/"code",source,source/"src",runtime/"data_dependencies",
                                     runtime/"upstream",runtime/"packages")))
        if profiling:
            entrypoint = "v4_long_profile.py"
        else:
            entrypoint = "v4_long_train.py" if formal else "v4_long_probe.py" if long_run else "v4_main250_hopper.py"
        argv = [runtime/"interpreter/bin/python3.12", "-B", capsule/"code"/entrypoint, capsule, custody]
        if long_run:
            argv.extend((short_run, bank_root))
        if profiling:
            argv.extend((profile_of, milestone))
        elif formal:
            argv.append(completed_probe)
        if prior_probe is not None:
            argv.extend(("--prior-probe", prior_probe))
        command=shlex.join(map(str, argv))
        profiler_environment = []
        if profiling:
            cupti = runtime / "packages/nvidia/cuda_cupti/lib"
            if not (cupti / "libcupti.so.12").is_file():
                raise ValueError("The profiled GPU job needs the pinned CUDA CUPTI collector")
            profiler_environment.append("export LD_LIBRARY_PATH=" + shlex.quote(str(cupti))
                                        + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}')
        script=capsule/"job.pbs"
        with script.open("x") as f:
            f.write("\n".join(["#!/bin/bash -l",f"#PBS -N {name}",f"#PBS -P {PROJECT}","#PBS -q auto",
                "#PBS -l select=1:ngpus=1:gpu_model=H200:ncpus=12:mem=225GB",f"#PBS -l walltime={wall_label}",
                "#PBS -m n","#PBS -r n",f"#PBS -o {capsule/'stdout.log'}",f"#PBS -e {capsule/'stderr.log'}",
                "set -euo pipefail","umask 077",f"cd {shlex.quote(str(capsule))}",
                "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                "export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "export PBS_SERVER="+shlex.quote(os.environ["PBS_SERVER"]),
                f"export PYTHONPATH={shlex.quote(paths)}",*profiler_environment,
                "sha256sum --status -c INPUTS.sha256","exec "+command,""]))
        intent=dict(plan=PLAN,name=name,project=PROJECT,wall_s=wall,prior_upper_gpu_s=prior,
                    grant=grant,batch_sha256=digest(script),inputs_sha256=digest(check))
        write_once(capsule/"INTENT.json",intent)
        journal=Journal(capsule/"journal",capsule.name)

        def hold():
            job=subprocess.check_output([str(PBS/"qsub"),"-f","-h",str(script)],text=True,timeout=120).strip()
            if not re.fullmatch(r"\d+\.hopper-m-02",job):
                raise RuntimeError("Unknown submission effect, stop and reconcile")
            write_once(capsule/"JOB.json",dict(job=job,name=name))
            return job

        job=journal.once("qsub",intent,hold)
        row=native(job)
        spec=row["Resource_List"]
        rate=float(spec["queue_charge_rate"])
        charge=required*rate
        if (row["job_state"]!="H" or row["project"]!=PROJECT or row["Job_Name"]!=name
                or row["Job_Owner"].split("@")[0]!=grant["account"] or int(spec["ngpus"])!=1
                or spec["select"]!="1:ngpus=1:gpu_model=H200:ncpus=12:mem=225GB"
                or wall_seconds(spec["walltime"])!=wall or not math.isfinite(rate) or rate<=0
                or charge>min(balance,grant["balance_su"])):
            raise PermissionError("Held job resources or charge differ from the live grant")
        admission=dict(job=job,charge_upper=charge,rate=rate,reserved_s=wall)
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
    cli.add_argument("--short-run",type=Path)
    cli.add_argument("--bank-root",type=Path)
    cli.add_argument("--completed-probe",type=Path)
    cli.add_argument("--profile-of",type=Path)
    cli.add_argument("--milestone",type=Path)
    cli.add_argument("--prior-probe",type=Path)
    os.umask(0o077)
    submit(**vars(cli.parse_args()))
