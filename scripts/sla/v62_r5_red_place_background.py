"""Run two verified frozen-model replays after the shell disconnects."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def read(path):
    return path.read_bytes()


def run(work, roster_sha):
    base = work.parent
    retention = base / "v6-retention-20261126-01"
    worker_path = base / "V2-M-PLACE-01/worker/EXECUTION_SPEC.json"
    worker_data = read(worker_path)
    if hashlib.sha256(worker_data).hexdigest() != "9b842ca2587e472dec961eed536208189a6d9309194f5b55446291298700e136":
        raise ValueError("The qualified offline inference worker changed")
    worker = json.loads(worker_data)
    roster = work / "R5-RED-PLACE-ROSTER-01/ROSTER.json"
    if hashlib.sha256(read(roster)).hexdigest() != roster_sha:
        raise ValueError("The recorded paired roster changed")
    code = Path(__file__).resolve().parent
    env = os.environ | worker["native_environment"] | worker["gpu_environment"]
    env["PYTHONPATH"] = ":".join(map(str, (code, work, retention / "code",
                             base / "v4-long-eval-source-01/scripts/sla", env["PYTHONPATH"])))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    roles = ("R4", "R5")
    for role in roles:
        verified = json.loads(read(work / f"R5-RED-PLACE-{role}-CPU.json"))
        if (verified["roster"]["sha256"] != roster_sha or verified["role"] != role
                or verified["windows"] != 12 or verified["model_forwards"] != 0
                or (work / f"R5-RED-PLACE-{role}-ACTUAL").exists()):
            raise ValueError("Both unique model and input bindings must be verified before execution")
    for role in roles:
        target = work / f"R5-RED-PLACE-{role}-ACTUAL"
        scratch = work / f"R5-RED-PLACE-{role}-SCRATCH"
        scratch.mkdir(mode=0o700)
        env["TMPDIR"] = env["XDG_CACHE_HOME"] = str(scratch)
        subprocess.run([worker["python"], "-Bu", str(code / "v62_r5_teacher_replay.py"), role,
                        str(worker_path), str(roster), str(retention), str(target),
                        "--roster-sha", roster_sha, "--execute"], env=env, check=True, timeout=1050)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("roster_sha")
    run(**vars(parser.parse_args()))
