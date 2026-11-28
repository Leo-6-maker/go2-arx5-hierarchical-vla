"""Submit V7-A qualification through the existing held Hopper admission."""

import argparse
from contextlib import closing
import os
from pathlib import Path
import shlex
import subprocess
import time

from hopper_pbs import PBS, native, project_grant
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock
from scripts.sla.hopper_parent_custody import write_once
from scripts.sla.hopper_site import DEADLINE
from v61_source_course import R4
from v62_source_admission import write_capacity
from v6_dual_source_submit import PROJECT, SERVER, admit
from v7_full_model_qualification import ARRAYS, BANK, CONTEXT, ENGINEERING_UPDATES, PLAN


NAME = "slav7aqual"
BUDGET = 7200
FINAL_3K = "6def8f651196a0e5f07bb4e4f99a5c7c5edd2cd5f5f3690d7d47eb847f39dea5"


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def submit(preceding, stage, output):
    personal = Path("/hpctmp") / os.environ["USER"]
    require(not output.exists() and output.parent == stage and stage.resolve().is_relative_to(personal)
            and time.time()+BUDGET < DEADLINE, "Resolve any existing capsule and preserve the project deadline")
    with closing(OwnerLock(stage / "SUBMIT.lock")):
        require(not output.exists(), "Concurrent V7 submission must be reconciled")
        old = read(preceding/"BINDING.json")
        seal = read(preceding/"FINAL-HOPPER-READBACK.json")
        release = read(preceding/"RELEASE.json")
        previous = native(release["job"])
        require(seal["step"] == 3000 and seal["final_3k"] and seal["pbs"]["job"] == release["job"]
                and sha(preceding/"run/STEP3000_SEALED.json") == FINAL_3K
                and old["r4_receipt"] == R4 and old["formal_prior_updates"] + 200 == 3600
                and old["engineering_remaining"] >= ENGINEERING_UPDATES
                and previous["job_state"] == "F" and previous["Exit_status"] == 0
                and previous["project"] == PROJECT, "The sealed V6.5 terminal or remaining allowance changed")
        grant = project_grant(BUDGET)
        account = grant["account"]
        require(previous["Job_Owner"].split("@")[0] == account, "Hopper grant owner changed")
        active = subprocess.check_output([str(PBS/"qselect"), "-q", f"@{SERVER}", "-u", account],
                                         text=True, timeout=45).split()
        require(all(not native(job)["Job_Name"].startswith(("slav4","slav6","slav7")) for job in active),
                "An earlier Hopper experiment owner is live or unresolved")
        prior = read(preceding/"INTENT.json")["prior_upper_gpu_s"] + release["reserved_s"]
        require(prior+BUDGET <= 320*3600, "The inherited GPU-hour ceiling does not cover qualification")
        require(not read(pin(old["source_hold"]["path"],old["source_hold"]["sha256"]))["released"],
                "SOURCE500 pause changed")
        paths = read(pin(old["data"]["path"],old["data"]["sha256"]))["parent_paths"]
        manifest, bank, context = (stage/f for f in ("MATCHED_INPUTS.json","BANK.npz","PREVIOUS_COMMAND.npz"))
        for path, digest in ((manifest,BANK),(bank,ARRAYS),(context,CONTEXT)):
            pin(path,digest)
        previous_inputs = preceding/"INPUTS.sha256"
        require(sha(previous_inputs) == read(preceding/"INTENT.json")["inputs_sha256"],
                "Earlier exact runtime pins changed")

        output.mkdir(mode=0o700)
        write_once(output/"WRITE_CAPACITY.json",write_capacity(output,Path(paths["r4"]),"probe",0,0,BUDGET,
                                                             quota_root=personal))
        write_once(output/"GRANT.json",grant)
        binding = output/"BINDING.json"
        write_once(binding,dict(plan=PLAN,r4_receipt=R4,parent=old["parent"],data=old["data"],
                   manifest=ref(manifest),bank=ref(bank),context=ref(context),site_grant=ref(output/"GRANT.json"),
                   quota_root=str(personal),budget_seconds=BUDGET,engineering_updates=ENGINEERING_UPDATES,formal_updates=0))
        inputs = output/"INPUTS.sha256"
        bound = (binding,manifest,bank,context,output/"GRANT.json",previous_inputs,
                 *(p for p in sorted(stage.glob("*.py")) if p.is_file()))
        with inputs.open("x") as stream:
            stream.writelines(f"{sha(p)}  {p}\n" for p in bound)

        lines = (preceding/"job.pbs").read_text().splitlines()
        updates = {
            "#PBS -N ": f"#PBS -N {NAME}", "#PBS -l walltime=": "#PBS -l walltime=02:00:00",
            "#PBS -o ": f"#PBS -o {output/'stdout.log'}", "#PBS -e ": f"#PBS -e {output/'stderr.log'}",
            "cd ": "cd "+shlex.quote(str(output)),
        }
        for prefix,replacement in updates.items():
            index, = [i for i,line in enumerate(lines) if line.startswith(prefix)]
            lines[index] = replacement
        inputs_start = next(i for i,line in enumerate(lines) if line.startswith("sha256sum "))
        import_index, = [i for i,line in enumerate(lines) if line.startswith("export PYTHONPATH=")]
        search, = shlex.split(lines[import_index].split("=",1)[1])
        lines[import_index] = "export PYTHONPATH="+shlex.quote(f"{stage}:{search}")
        command, = [line for line in lines if line.startswith("exec ")]
        _,python,*_ = shlex.split(command)
        lines = lines[:inputs_start] + ["sha256sum --status -c INPUTS.sha256",
                 "sha256sum --status -c "+shlex.quote(str(previous_inputs)),
                 "exec "+shlex.join((python,"-B",str(stage/"v7_full_model_qualification.py"),str(output)))]
        script=output/"job.pbs"
        with script.open("x") as stream:
            stream.write("\n".join(lines)+"\n")
        subprocess.run(["bash","-n",str(script)],check=True)
        intent=dict(plan=PLAN,name=NAME,budget_seconds=BUDGET,prior_upper_gpu_s=prior,
                    binding_sha256=sha(binding),program_sha256=sha(script),inputs_sha256=sha(inputs))
        return admit(output,intent,script,NAME,BUDGET,grant)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("preceding","stage","output"):
        parser.add_argument(key,type=Path)
    os.umask(0o077)
    print(submit(**vars(parser.parse_args())),flush=True)
