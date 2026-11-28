"""Admit the fixed V7-A 500/500 pair after the independent GPU qualification closes."""

import argparse
from contextlib import closing
from pathlib import Path
import shlex
import subprocess
import time

from hopper_pbs import PBS, native, project_grant
from m38_gpu_preflight_v1 import read, require, sha
from scripts.pi0292.durable import OwnerLock
from scripts.sla.hopper_parent_custody import write_once
from scripts.sla.hopper_site import DEADLINE
from v3_main_place_learn import endpoint_files
from v61_source_course import R4
from v6_dual_source_submit import PROJECT, SERVER, admit
from v7_formal_worker import allocated_bytes, capacity
from v7_full_model_qualification import ARMS, PLAN, STEPS, file_ref
from v7_submit_qualification import ref


NAME = "slav7apair"
BUDGET = 10800


def qualification(capsule, binding):
    end = read(capsule/"TERMINAL.json")
    release = read(capsule/"RELEASE.json")
    job = native(release["job"])
    require(job["job_state"] == "F" and job["Exit_status"] == 0 and job["project"] == PROJECT
            and end["status"] == "GPU_QUALIFIED" and end["r4"] == binding["r4_receipt"] == R4
            and end["plan"] == binding["plan"] == PLAN and end["actual_engineering_updates"] == 4
            and end["formal_updates"] == binding["formal_updates"] == 0,
            "Formal continuation requires a naturally closed, complete V7-A qualification")
    footprints = []
    for report, arm in zip(end["cold_restore"], ARMS, strict=True):
        checkpoint = capsule/f"{arm}-STEP0001"
        require(report["arm"] == arm and report["exact_next_gradient"]
                and report["files"] == endpoint_files(checkpoint), "Qualification cold state no longer matches its bytes")
        footprints.append(allocated_bytes(checkpoint))
    inputs = capsule/"INPUTS.sha256"
    intent = read(capsule/"INTENT.json")
    require(sha(inputs) == intent["inputs_sha256"] and sha(capsule/"BINDING.json") == intent["binding_sha256"],
            "Qualification code and data identity changed")
    return job, int(intent["prior_upper_gpu_s"]) + int(release["reserved_s"]), (3*max(footprints)+1)//2


def submit(probe, stage, output):
    old = read(probe/"BINDING.json")
    root = Path(old["quota_root"]).resolve(strict=True)
    require(output.parent == stage and not output.exists() and stage.resolve().is_relative_to(root)
            and time.time()+BUDGET < DEADLINE, "Use a new isolated personal stage before the project deadline")
    with closing(OwnerLock(stage/"SUBMIT.lock")):
        require(not output.exists(), "Resolve an existing formal capsule without repeated effects")
        job, prior, bound = qualification(probe, old)
        grant = project_grant(BUDGET)
        require(prior+BUDGET <= 320*3600 and 3600+len(ARMS)*STEPS <= 8000
                and job["Job_Owner"].split("@")[0] == grant["account"], "Original accounting cannot cover this pair")
        active = subprocess.check_output([str(PBS/"qselect"),"-q",f"@{SERVER}","-u",grant["account"]],
                                         text=True,timeout=45).split()
        require(all(not native(identifier)["Job_Name"].startswith(("slav4","slav6","slav7")) for identifier in active),
                "An experiment GPU owner is active or unresolved")
        for key in ("manifest","bank","context","data","parent"):
            file_ref(old[key])

        output.mkdir(mode=0o700)
        write_once(output/"WRITE_CAPACITY.json",capacity(output,root,bound))
        write_once(output/"GRANT.json",grant)
        binding = output/"BINDING.json"
        write_once(binding,dict(old,formally_authorized=True,arms=list(ARMS),formal_steps_per_arm=STEPS,
                   budget_seconds=BUDGET,engineering_updates=0,prior_formal_updates=3600,
                   qualification_terminal=ref(probe/"TERMINAL.json"),qualification_input=ref(probe/"STEP0-INPUTS.json"),
                   qualification_parameters=ref(probe/"PARAMETERS.json"),qualification_job=read(probe/"JOB.json")["job"],
                   checkpoint_bytes=bound,site_grant=ref(output/"GRANT.json")))
        pins = output/"INPUTS.sha256"
        original = probe/"INPUTS.sha256"
        with pins.open("x") as stream:
            files = (binding,output/"GRANT.json",original,probe/"TERMINAL.json",probe/"STEP0-INPUTS.json",
                     probe/"PARAMETERS.json",*sorted(stage.glob("*.py")))
            stream.writelines(f"{sha(f)}  {f}\n" for f in files)
        source = (probe/"job.pbs").read_text().splitlines()
        changes = {"#PBS -N ":f"#PBS -N {NAME}","#PBS -l walltime=":"#PBS -l walltime=03:00:00",
                   "#PBS -o ":f"#PBS -o {output/'stdout.log'}","#PBS -e ":f"#PBS -e {output/'stderr.log'}",
                   "cd ":"cd "+shlex.quote(str(output))}
        for prefix, updated in changes.items():
            i, = (i for i,line in enumerate(source) if line.startswith(prefix))
            source[i] = updated
        i, = (i for i,line in enumerate(source) if line.startswith("export PYTHONPATH="))
        search, = shlex.split(source[i].split("=",1)[1])
        source[i] = "export PYTHONPATH="+shlex.quote(f"{stage}:{search}")
        j = next(i for i,line in enumerate(source) if line.startswith("sha256sum "))
        _, python, *_ = shlex.split(next(line for line in source if line.startswith("exec ")))
        source = source[:j] + ["sha256sum --status -c INPUTS.sha256",
                   "sha256sum --status -c "+shlex.quote(str(original)),
                   *(line for line in source[j+1:] if line.startswith("sha256sum ")),
                   "exec "+shlex.join((python,"-B",str(stage/"v7_formal_worker.py"),str(output)))]
        script = output/"job.pbs"
        with script.open("x") as stream:
            stream.write("\n".join(source)+"\n")
        subprocess.run(["bash","-n",str(script)],check=True)
        intent = dict(plan=PLAN,name=NAME,budget_seconds=BUDGET,prior_upper_gpu_s=prior,
                      binding_sha256=sha(binding),program_sha256=sha(script),inputs_sha256=sha(pins))
        return admit(output,intent,script,NAME,BUDGET,grant)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("probe","stage","output"):
        cli.add_argument(field,type=Path)
    print(submit(**vars(cli.parse_args())),flush=True)
