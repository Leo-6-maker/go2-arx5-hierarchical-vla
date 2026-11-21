"""Private direct transfer of the fixed ADAPT2K restart state. CPU-only PBS role."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

PLAN = "171b5bedb290258954aac140157743e92e3b152e"
PARENT = "2bff547b6f2c3aca0e7e2fd8a790ce6b30ef1c5f33b004c6897e3083647358d5"
SSH = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
       "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3"]
HOST = "hopper.nus.edu.sg"


def sync_dir(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        json.dump(value, out, sort_keys=True, indent=2)
        out.flush()
        os.fsync(out.fileno())
    try:
        os.link(out.name, path)
        sync_dir(path.parent)
    except FileExistsError:
        if json.loads(path.read_bytes()) != value:
            raise RuntimeError("Conflicting receipt: " + path.name)
    finally:
        os.unlink(out.name)


def digest(path, flush=False):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected regular file: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
        if flush:
            os.fsync(f.fileno())
    return h.hexdigest()


def members(root):
    seal = root / "COMPLETE.json"
    if digest(seal) != PARENT:
        raise ValueError("Parent seal digest mismatch")
    header = json.loads(seal.read_bytes())
    arrays = header["arrays"]
    names = ["%06d.npy" % n for n in range(140)]
    if (header["phase_step"] != 2000 or [m["file"] for m in arrays] != names
            or header["progress"]["lineage_total_updates"] != 2500):
        raise ValueError("Parent range or identity mismatch")
    return {**{m["file"]: m["sha256"] for m in arrays}, "COMPLETE.json": PARENT}


def inspect(root, flush=False):
    expected = members(root)
    if {p.name for p in root.iterdir()} != set(expected):
        raise ValueError("Unexpected parent directory membership")
    sizes = {}
    for name, sha in expected.items():
        path = root / name
        if digest(path, flush) != sha:
            raise ValueError("Array digest mismatch: " + name)
        sizes[name] = path.stat().st_size
    return sizes


def remote(*argv):
    return subprocess.check_output([*SSH, HOST, shlex.join(argv)], text=True).strip()


def receive(base, expectation):
    sizes = json.loads(expectation.read_bytes())
    incoming = base / "parent.incoming"
    final = base / "parent.verified"
    folder = final if final.is_dir() else incoming
    if inspect(folder, flush=True) != sizes:
        raise ValueError("Parent sizes mismatch")
    if folder == incoming:
        sync_dir(incoming)
        incoming.rename(final)
        sync_dir(base)
    receipt = {"status": "PARENT_BYTES_VERIFIED", "goal_plan": PLAN, "seal": PARENT,
               "arrays": 140, "bytes": sum(sizes.values()), "job_id": expectation.parent.name,
               "training_ready": False, "updates": 0, "generations": 0, "native": 0}
    write_once(base / "PARENT_BYTES_VERIFIED.private.json", receipt)
    print(json.dumps(receipt))


def transfer(source, base, source_sha):
    root = Path.cwd()
    job = os.environ["PBS_JOBID"]
    if not re.fullmatch(r"[0-9]+\.stdct-mgmt-02", job):
        raise ValueError("Unexpected PBS identity")
    registration = json.loads((root / "MOUNT.private.json").read_bytes())
    if (registration["job_id"] != job or registration["source_sha"] != source_sha
            or time.time() >= registration["deadline_epoch"] or (root / "STOP").exists()):
        raise ValueError("Transfer owner, deadline or STOP")
    if digest(Path(__file__)) != registration["worker_sha256"]:
        raise ValueError("Worker SHA mismatch")
    sizes = inspect(source)
    receipt_dir = base / job
    inputs = root / "MEMBER_SIZES.private.json"
    write_once(inputs, sizes)
    remote("bash", "-c", 'umask 077; mkdir -p -- "$1" "$2"', "bash",
           str(base / "parent.incoming"), str(receipt_dir))
    endpoint = HOST + ":"
    rsync = ["rsync", "-rlt", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=", "-e", shlex.join(SSH)]
    subprocess.run([*rsync, str(Path(__file__).resolve()), str(inputs),
                    endpoint + str(receipt_dir) + "/"], check=True)
    subprocess.run([*rsync, "--files-from=-", str(source) + "/",
                    endpoint + str(base / "parent.incoming") + "/"],
                   input="\n".join(sizes) + "\n", text=True, check=True)
    if (root / "STOP").exists() or time.time() >= registration["deadline_epoch"]:
        raise RuntimeError("Stopped before byte-verification receipt")
    result = remote("bash", "-c",
                    'set -eu; cd "$1"; test "$(sha256sum hopper_parent_custody.py | cut -d" " -f1)" = "$2"; '
                    'exec python3 -B hopper_parent_custody.py receive "$3" "$1/MEMBER_SIZES.private.json"',
                    "bash", str(receipt_dir), registration["worker_sha256"], str(base))
    receipt = json.loads(result)
    if receipt["seal"] != PARENT or receipt["bytes"] != sum(sizes.values()) or receipt["job_id"] != job:
        raise ValueError("Returned receipt does not match transfer")
    write_once(root / "RESULT.private.json", receipt)


def main():
    os.umask(0o077)
    mode, *args = sys.argv[1:]
    if mode == "receive":
        receive(Path(args[0]), Path(args[1]))
    elif mode == "transfer":
        with Path("OWNER.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                transfer(Path(args[0]), Path(args[1]), args[2])
            except Exception as exc:
                write_once(Path("FAILURE.private.json"), {"error": str(exc), "job": os.environ.get("PBS_JOBID")})
                raise
    else:
        raise ValueError("Unknown role")


if __name__ == "__main__":
    main()
