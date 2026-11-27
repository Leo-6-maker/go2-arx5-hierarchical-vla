"""Submit the user-directed, exact 1K-to-3K V6.5 continuation to Hopper."""

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
from v62_source_admission import write_capacity
from v65_recovery_chain import boundary
from v65_recovery_extension import BASE_SEAL, BASE_TERMINAL, COURSE, END, START, original
from v65_recovery_submit import ref
from v6_dual_source_submit import PROJECT, SERVER, admit


NAME = "slav65r3a3k"


def submit(parent, stage, output, *, after=None, hours=4):
    budget = hours * 3600
    require(1 <= hours <= 6 and time.time()+budget < DEADLINE and output.parent.resolve() == parent.parent.resolve(),
            "Original scratch and project deadline must cover the continuation")
    with closing(OwnerLock(stage / "SUBMIT.lock")):
        require(not output.exists(), "Reconcile the existing continuation owner before submission")
        base, _ = original(parent)
        bank = stage / "BANK-3K/ACTUAL.json"
        prior, start = {}, START
        name = parent.name.replace("-1K-", "-3K-")
        if after is not None:
            require(after.parent == parent.parent and after != parent, "Unknown continuation capsule")
            previous_boundary, _ = boundary(parent, after, bank)
            start = previous_boundary["step"]
            name += f"-from-{start:04d}"
            prior["previous_boundary"] = previous_boundary
        require(output.name == name, "Use the unique capsule for this observed boundary")
        preceding = after or parent
        release = read(preceding / "RELEASE.json")
        previous = native(release["job"])
        require(previous["job_state"] == "F" and previous["Exit_status"] == 0 and previous["project"] == PROJECT
                and release["job"] == read(preceding / "JOB.json")["job"], "The preceding PBS must have closed normally")
        require(base["formal_prior_updates"] + END <= 8000, "Continuation would exceed formal budget")
        accounted = read(preceding / "INTENT.json")["prior_upper_gpu_s"] + release["reserved_s"]
        require(accounted+budget <= 320*3600, "Original GPU-hour allowance cannot cover this run")
        try:
            grant = project_grant(budget)
        except BalanceLoginExpired:
            old = read(pin(base["site_grant"]["path"], base["site_grant"]["sha256"]))
            lineage = Path(old["lineage"])
            grant = project_period(old["basis"], lineage, budget)
            grant["lineage"] = str(lineage)
        require(previous["Job_Owner"].split("@")[0] == grant["account"], "The PBS owner differs from the original run")
        active = subprocess.check_output([str(PBS/"qselect"), "-q", f"@{SERVER}", "-u", grant["account"]],
                                         text=True, timeout=45).split()
        require(all(not native(job)["Job_Name"].startswith(("slav4", "slav6")) for job in active),
                "A previous project training owner is still active or unresolved")
        packed = read(bank)
        first = read(bank.with_name("FIRST-BATCH.json"))
        require(packed["course"] == COURSE and packed["start"] == START and packed["end"] == END
                and packed["original_seal"] == BASE_SEAL and packed["original_bank"] == base["bank"]["sha256"]
                and packed["bank_sha256"] == sha(bank.with_name("BANK.npz"))
                and first["bank"] == sha(bank) and first["cursor"] == START
                and first["actual_shapes"] == {"target": [96,40,32], "robodojo": [16,50,32], "libero": [16,10,32]}
                and first["auxiliary_shape"] == [4, 40] and first["optimizer_visits"] == packed["actual_optimizer_visits"] == 0,
                "Require the real continuation first batch and the sealed appended bank")
        paths = read(pin(base["data"]["path"], base["data"]["sha256"]))["parent_paths"]
        output.mkdir(mode=0o700)
        write_once(output / "WRITE_CAPACITY.json", write_capacity(output, Path(paths["r4"]), str(END), start, END, budget))
        write_once(output / "GRANT.json", grant)
        declaration = output / "BINDING.json"
        binding = {**base, **prior, "course": COURSE, "formal_start": start, "formal_end": END, "engineering_updates": 0,
                   "formal_prior_updates": base["formal_prior_updates"]+start, "base_capsule": str(parent),
                   "base_seal": BASE_SEAL, "base_terminal": BASE_TERMINAL, "base_pbs": read(parent / "JOB.json")["job"],
                   "bank": ref(bank), "host_batch": ref(bank.with_name("FIRST-BATCH.json")),
                   "site_grant": ref(output / "GRANT.json"), "budget_seconds": budget}
        write_once(declaration, binding)
        source = (parent / "job.pbs").read_text().splitlines()
        inputs = output / "INPUTS.sha256"
        # Keep the original source and dependency pins in the original immutable capsule.
        original_inputs = parent / "INPUTS.sha256"
        require(sha(original_inputs) == read(parent / "INTENT.json")["inputs_sha256"], "Original runtime pins changed")
        with inputs.open("x") as stream:
            stream.writelines(f"{sha(p)}  {p}\n" for p in (declaration, bank, bank.with_name("BANK.npz"),
                bank.with_name("FIRST-BATCH.json"), output / "GRANT.json", original_inputs, *sorted((stage / "code").glob("*.py"))))
        additions = {
            "#PBS -N ": f"#PBS -N {NAME}", "#PBS -l walltime=": f"#PBS -l walltime={hours:02d}:00:00",
            "#PBS -o ": f"#PBS -o {output/'stdout.log'}", "#PBS -e ": f"#PBS -e {output/'stderr.log'}",
            "cd ": "cd "+shlex.quote(str(output)),
            "sha256sum ": "sha256sum --status -c INPUTS.sha256\nsha256sum --status -c "+shlex.quote(str(original_inputs)),
        }
        for prefix, new in additions.items():
            index, = [i for i, line in enumerate(source) if line.startswith(prefix)]
            source[index] = new
        imports, = [i for i, line in enumerate(source) if line.startswith("export PYTHONPATH=")]
        search_path, = shlex.split(source[imports].split("=", 1)[1])
        source[imports] = "export PYTHONPATH="+shlex.quote(f"{stage/'code'}:{search_path}")
        command, = [i for i, line in enumerate(source) if line.startswith("exec ")]
        _, python, *_ = shlex.split(source[command])
        source[command] = "exec "+shlex.join((python, "-B", str(stage/"code/v65_recovery_worker.py"), str(output)))
        script = output / "job.pbs"
        with script.open("x") as stream:
            stream.write("\n".join(source)+"\n")
        intent = dict(plan=base["plan"], course=COURSE, name=NAME, budget_seconds=budget, prior_upper_gpu_s=accounted,
                      binding_sha256=sha(declaration), program_sha256=sha(script), inputs_sha256=sha(inputs))
        print(admit(output, intent, script, NAME, budget, grant), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "stage", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--after", type=Path)
    cli.add_argument("--hours", type=int, choices=range(1, 7), default=4)
    os.umask(0o077)
    submit(**vars(cli.parse_args()))
