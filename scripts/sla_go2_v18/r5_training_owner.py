"""Corelab admission and GPU ownership for the finite R5 smoke run."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parents[2]
SCOPE = "SLA_GO2_V18_R5_RGB_SMOKE"
RECEIVER_SHA256 = "a2e2765ce76dfd163345587ad1e63de421561f410e01aa4c7c02ca55dc5c7074"
REQUIRED_SOURCES = {
    "reports/SLA_GO2_V17/ARCHIVE_READBACK.json",
    "reports/SLA_GO2_V17/RIGHT_ARM_RECIPE_CPU.json",
    "reports/SLA_GO2_V17/TRAIN_QUALIFICATION.json",
    "reports/SLA_GO2_V18/P1_P2/OPTIMIZER_ABI.json",
    "reports/SLA_GO2_V18/P1_P2/TRAIN_INPUT_ACTUAL.json",
    "scripts/pi0292/durable.py",
    "scripts/sla_go2_v18/audit_r5_training.py",
    "scripts/sla_go2_v18/r5_checkpoint.py",
    "scripts/sla_go2_v18/r5_optimizer_preflight.py",
    "scripts/sla_go2_v18/r5_training_kernel.py",
    "scripts/sla_go2_v18/r5_training_owner.py",
    "scripts/sla_hierarchical_v16/checkpoint_inputs.py",
    "scripts/sla_hierarchical_v16/demonstrated_grip.py",
    "scripts/sla_hierarchical_v16/observation_stream.py",
    "scripts/sla_hierarchical_v16/right_arm_recipe.py",
    "scripts/sla_hierarchical_v16/target_labels.py",
    "scripts/sla_hierarchical_v16/visual_lift_panel.py",
    "scripts/sla_hierarchical_v16/visual_objects.py",
}
DESKTOP = {"/usr/lib/xorg/Xorg", "/usr/bin/gnome-shell"}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned(pin):
    path = Path(pin["path"])
    if not path.is_file() or checksum(path) != pin["sha256"]:
        raise PermissionError("R5_PINNED_RECEIPT_REQUIRED:" + path.name)
    return json.loads(path.read_bytes())


def authorize(root, spec, now):
    if not spec.get("scientific_allocation"):
        raise PermissionError("R5_SCIENTIFIC_ALLOCATION_REQUIRED")
    allocation = pinned(spec["scientific_allocation"])
    start, stop = spec["start_step"], spec["stop_step"]
    if (allocation.get("scope") != SCOPE or allocation.get("status") != "APPROVED"
            or allocation.get("scientific_allocation_kind") != "GO2_POST_TRAINING_REAL_GPU_SMOKE"
            or allocation.get("trial") != spec["trial"]
            or allocation.get("mode") != spec["mode"]
            or spec["mode"] not in ("START", "RESUME")
            or (spec["mode"] == "START" and start != 0)
            or allocation.get("start_step") != start or allocation.get("stop_step") != stop
            or allocation.get("training_step_cap") != spec["training_step_cap"]
            or not all(type(x) is int for x in (start, stop, spec["training_step_cap"]))
            or not 0 <= start < stop <= spec["training_step_cap"] <= 100
            or allocation.get("source_files") != spec["source_files"]
            or allocation.get("val_input_sha256") != spec["independent_val"]["sha256"]
            or allocation.get("native_starts") != 0 or allocation.get("model_sampling_forwards") != 0
            or allocation.get("sampler") != "post_grasp_reserved"
            or not now < spec["deadline_epoch"] <= allocation["deadline_epoch"]):
        raise PermissionError("R5_EXACT_FINITE_GPU_SCIENCE_SCOPE_REQUIRED")
    if not re.fullmatch(r"r5-smoke-[a-z0-9][a-z0-9-]{1,44}", spec["trial"]):
        raise PermissionError("R5_ISOLATED_TRIAL_NAME_REQUIRED")

    val = pinned(spec["independent_val"])
    val_ids = [row["slot"] for row in val.get("examples", ())]
    if (val.get("scope") != "INDEPENDENT_VAL_MODEL_INPUTS"
            or not val_ids or len(val_ids) != len(set(val_ids))
            or val.get("independent_VAL_episodes") != len(val_ids)
            or not set(val_ids) <= set(val["sample_accounting"]["started"])
            or any(not row.get("official_rows") or not row.get("teacher_pass") for row in val["examples"])
            or val.get("TEST_episodes") != 0 or val.get("model_forwards") != 0):
        raise PermissionError("R5_INDEPENDENT_VAL_INPUT_RECEIPT_REQUIRED")
    if not REQUIRED_SOURCES <= spec["source_files"].keys():
        raise PermissionError("R5_EXACT_EXECUTION_SOURCE_CLOSURE_REQUIRED")
    for name, expected in spec["source_files"].items():
        path = (BASE / name).resolve()
        if not path.is_relative_to(BASE) or not path.is_file() or checksum(path) != expected:
            raise PermissionError("R5_EXECUTION_SOURCE_CHANGED:" + name)

    python = root / "vendor/XPolicyLab/policy/SLA_Memory/openpi/.venv/bin/python"
    if Path(sys.executable) != python:
        raise PermissionError("R5_PINNED_OFFICIAL_PYTHON_REQUIRED")

    # Keep all new state under a trial directory separate from prior native and
    # model receipts. Resolve the path before opening the canonical GPU lock.
    output = (root / "v18" / spec["trial"]).resolve()
    if output.parent != (root / "v18").resolve() or output.name != spec["trial"]:
        raise PermissionError("R5_ISOLATED_OUTPUT_REQUIRED")
    return output


def hardware_lease(spec, now):
    lease_ref = spec.get("resource_lease")
    if not lease_ref:
        raise PermissionError("R5_LIVE_CORELAB_GPU_LEASE_REQUIRED")
    lease = pinned(lease_ref)
    receiver = pinned(spec["canonical_receiver"])
    if (spec["canonical_receiver"]["sha256"] != RECEIVER_SHA256
            or lease.get("status") != "ACTIVE" or lease.get("scope") != SCOPE
            or lease.get("trial") != spec["trial"] or lease.get("host") != "corelab-4080s"
            or socket.gethostname().split(".")[0] != lease["host"]
            or lease.get("uid") != os.getuid() or os.getuid() != 2009
            or lease.get("gpu_uuid") != receiver["gpu"]
            or type(lease.get("min_free_vram_mib")) is not int or lease["min_free_vram_mib"] <= 0
            or Path(lease["gpu_lock"]).resolve(strict=True) != Path(receiver["gpu_lock"]).resolve(strict=True)
            or not lease["issued_epoch"] <= now < spec["deadline_epoch"] <= lease["deadline_epoch"]):
        raise PermissionError("R5_LIVE_CORELAB_GPU_LEASE_REQUIRED")
    return lease


def gpu_status(lease):
    report = subprocess.check_output(
        ["nvidia-smi", "-i", lease["gpu_uuid"], "-q", "-x"], text=True, timeout=12,
    )
    info = ET.fromstring(report).find("gpu")
    if info is None or info.findtext("uuid") != lease["gpu_uuid"]:
        raise PermissionError("R5_GPU_IDENTITY_CHANGED")
    approved = {int(v) for v in lease["shared_desktop_pids"]}
    desktop = info.findall("./processes/process_info")
    if any(row.findtext("type") != "G" or row.findtext("process_name") not in DESKTOP
           or int(row.findtext("pid")) not in approved for row in desktop):
        raise PermissionError("R5_GPU_OCCUPANCY_CHANGED")
    memory = info.findtext("fb_memory_usage/free")
    available_mib = int(memory.split()[0])
    if available_mib < lease["min_free_vram_mib"]:
        raise PermissionError("R5_INSUFFICIENT_FREE_GPU_MEMORY")
    return available_mib


@contextmanager
def gpu_owner(lease):
    with Path(lease["gpu_lock"]).open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PermissionError("R5_CANONICAL_GPU_LOCK_BUSY") from exc
        yield handle.fileno(), gpu_status(lease)



def require_inherited_gpu_owner(lease, fd):
    canonical = Path(lease["gpu_lock"])
    expected, inherited = canonical.stat(), os.fstat(fd)
    if (expected.st_dev, expected.st_ino) != (inherited.st_dev, inherited.st_ino):
        raise PermissionError("R5_WRONG_GPU_LOCK_DESCRIPTOR")
    with canonical.open("rb") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise PermissionError("R5_GPU_LOCK_NOT_HELD")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise PermissionError("R5_GPU_LOCK_NOT_INHERITED") from exc


def launch(root, spec_path, spec, spec_sha256):
    now = time.time()
    output = authorize(root, spec, now)
    lease = hardware_lease(spec, now)
    bytes_per_step = 3 * 22_118_400 * 4
    space = shutil.disk_usage(output.parent).free
    if space < (50 << 30) + bytes_per_step * (spec["stop_step"] - spec["start_step"] + 1):
        raise PermissionError("R5_CHECKPOINT_VOLUME_INSUFFICIENT")
    with gpu_owner(lease) as (fd, available):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=lease["gpu_uuid"], JAX_PLATFORMS="cuda",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=str(root / "vendor"), OPENPI_DATA_HOME=str(root / "openpi-cache"),
                   JAX_COMPILATION_CACHE_DIR=str(output.parent / ("." + spec["trial"] + "-jax")),
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        remaining = spec["deadline_epoch"] - time.time()
        if remaining <= 0:
            raise PermissionError("R5_FINITE_LEASE_EXPIRED")
        command = [sys.executable, "-B", __file__, str(root), str(spec_path), "--inherited",
                   str(fd), spec_sha256, str(available)]
        try:
            return subprocess.run(command, env=env, pass_fds=(fd,), timeout=remaining).returncode
        except subprocess.TimeoutExpired:
            print("R5_DEADLINE: owned CUDA worker exited; reconcile any uncommitted intent", file=sys.stderr)
            return 124


def run_inherited(root, spec, spec_sha256, launch_data):
    fd, expected, available = launch_data
    if spec_sha256 != expected:
        raise PermissionError("R5_EXECUTION_SPEC_CHANGED")
    now = time.time()
    output = authorize(root, spec, now)
    lease = hardware_lease(spec, now)
    require_inherited_gpu_owner(lease, int(fd))
    from r5_training_kernel import train

    return train(root, output, spec, int(available))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--inherited", nargs=3, metavar=("FD", "SPEC_SHA256", "FREE_VRAM_MIB"))
    args = parser.parse_args()
    path = args.spec.resolve()
    data = path.read_bytes()
    spec = json.loads(data)
    sha = hashlib.sha256(data).hexdigest()
    if args.inherited:
        raise SystemExit(run_inherited(args.root.resolve(), spec, sha, args.inherited))
    raise SystemExit(launch(args.root.resolve(), path, spec, sha))
