"""V6.2 scope and write capacity checks for the unchanged SOURCE learner."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import time

from m38_gpu_preflight_v1 import pin, read, require, sha


V62 = "90efbfb8785dbdf7fbb1219fbe179e049ec41e31"
ROSTER = "c2af53f5b70340d1543082adbdbb769ea8e0e66231ac63d56059485136669605"
N02 = "db77327c1ff1bd3d1dd759d51afb4b7ada4c5d230b0da937aa893d0d8fbb243d"
PLACE = "ce7077c88ed98d00277340098a3851fff1acfe6e35dc529d0b69e65c96e9806d"
R4 = "f1f5a994005a9fc831934a156ca823d7e4b67794ce2533d72e087c2aeb29b9cb"


def provisional_gate(reference, stage):
    proof = read(pin(reference["path"], reference["sha256"]))
    require(stage in ("probe", "250") and proof["plan"] == V62 and proof["roster"] == ROSTER
            and proof["r4"] == R4 and proof["n02"] == N02 and proof["place"] == PLACE
            and proof["hold_success"] == 2 and proof["place_grasps"] == 4
            and proof["previous_safety_stops"] == 0 and not proof["independent_parent_released"]
            and not proof["known_independent_regression"] and proof["provisional_ceiling"] == 250,
            "Only the incomplete, non-regressing R4 independent panel permits the V6.2 first segment")


def checkpoint_bound(r4):
    baseline = int(subprocess.check_output(["du", "-sB1", "--", str(r4 / "STEP0200")], text=True).split()[0])
    require(baseline > 0, "Read the actual R4 complete-state footprint")
    # Reserve one extra half of the inherited complete state for the new IO
    # parameters, their Adam moments and Orbax write overhead.
    return (3 * baseline + 1) // 2


def write_capacity(capsule, r4, stage, start, end, budget, *, quota_root=None):
    slot_bytes = checkpoint_bound(r4)
    # Periodic clock and update checkpoints can both be exercised. Include
    # the terminal and one overlapping atomic save as separate reservations.
    writes = 2 if stage == "probe" else end // 100 - start // 100 + budget // 1800 + 2
    expected = (writes + 1) * slot_bytes
    free = shutil.disk_usage(capsule).free
    if quota_root is not None:
        root = quota_root.resolve(strict=True)
        require(capsule.resolve().is_relative_to(root) and root.stat().st_uid == os.getuid()
                and root.stat().st_dev == capsule.stat().st_dev, "Unbound personal quota root")
        free = min(free, shutil.disk_usage(root).free)
    require(free > expected, f"Actual safe checkpoint capacity unavailable: {free} <= {expected}")
    with tempfile.TemporaryFile(dir=capsule) as trial:
        os.posix_fallocate(trial.fileno(), 0, 64 * 1024**2)
        os.fsync(trial.fileno())
    proof = dict(free_bytes=free, planned_bytes=expected, checkpoint_bytes=slot_bytes,
                 write_probe_bytes=64 * 1024**2, site_device=capsule.stat().st_dev)
    if quota_root is not None:
        proof["quota_root"] = str(root)
    return proof


def award(binding, capsule, allocation_value):
    from hopper_pbs import native
    from hopper_project_admission import CLUSTER, allocation

    admission = read(capsule / "ADMITTED.json")
    grant = read(pin(binding["site_grant"]["path"], binding["site_grant"]["sha256"]))
    require(sha(capsule / "BINDING.json") == read(capsule / "INTENT.json")["binding_sha256"],
            "PBS science binding changed")
    if grant["source"] == "SITE_PBS_PROJECT_AWARD_REQUIRED":
        basis = grant["basis"]
        claim = read(pin(basis["path"], basis["sha256"]))
        require(claim["uid"] == os.getuid() and claim["line"] == grant["lineage"]
                and claim["cluster"] == CLUSTER and claim["project"] == grant["project"]
                and claim["project_observation"]["account"] == grant["account"]
                and time.time() < min(claim["deadline"], grant["expiry_epoch"]),
                "The pinned project period does not cover this executing node")
    require(admission["job"] == allocation_value["job"], "PBS award differs from running process")
    rate = allocation(native(admission["job"]), grant["account"], admission["reserved_s"], "R")
    require(rate == admission["rate"], "Running PBS charge differs from the admitted job")
    return dict(job=admission["job"], queue_rate=rate, project=grant["project"],
                billing_cache_verified=True, authority=grant["source"])
