"""Seal the early Recovery CUDA draw bank on the original Corelab sampling path."""

import argparse
from contextlib import closing
import hashlib
import os
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require
from scripts.pi0292.durable import OwnerLock, write
from v7_d0_roster import PLAN, QUERIES


ROSTER = "4e10cfdd865ade7fe6e360664d2601530ba5737e9160e14d1c9abdde4edfe761"
FOLLOWUP_ROSTER = "3a719a4f644cf2b5513eb1b0b5dcaaff6c040cd5cdfad9ebcab1e11cdae3310d"
FIRST_NOISE = ("v7-d0-noise.tITolp/ACTUAL.private.json",
               "1073abde8dae989e6534fe298b30063a6c770767808add685ce565ea4e7f48e6")
WORKER = "361709e35c2bf427d5644e7af37a925014ab52861c075d4782749afd371d6bfc"
OLD_NOISE = (
    ("V64-CPU-READBACK-02/RECOVERY-TRAIN-CUDA-NOISE.json",
     "8ddc137196a1b035d9b835b9397bef51eff86d6037ee55de47c06131dce6e89e"),
    ("v64-recovery-wave2-code-01/RECOVERY-TRAIN-CUDA-NOISE-02.json",
     "ccd74ac999aa16fc29cad8a7e57d6fc097207c8cbcc1274bf77d736dba0f3cf6"),
    ("v64-recovery-open-code-01/RECOVERY-TRAIN-CUDA-NOISE-03.json",
     "fe1dcf4529d77a822da1ef220bab0c04d41f1d306863b8ca2182f7dcbeccb30a"),
    ("v64-recovery-wave4-code-01/RECOVERY-TRAIN-CUDA-NOISE-04.json",
     "4fca10f106f45bfa09bb62710d39403d0fe95347c42ab9efbd54674ee18d0b31"),
    ("v64-recovery-wave5-code-01/RECOVERY-TRAIN-CUDA-NOISE-05.json",
     "baecc922957ea2766e94e1d6ac621895e1c08337740f26298b73d5136c65c8a2"),
)


def seal(site, roster_path, worker_path, output, followup=False):
    import jax
    import numpy as np

    roster_sha = FOLLOWUP_ROSTER if followup else ROSTER
    roster = read(pin(roster_path, roster_sha))
    worker = read(pin(worker_path, WORKER))
    require(roster["plan"] == PLAN and roster["split"] == "TRAIN"
            and not roster["authorized_native_runtime"] and not roster["actual_starts"]
            and os.environ["CUDA_VISIBLE_DEVICES"] == worker["GPU_uuid"], "Require the pinned Corelab sampling device")
    owners = ("RECOVERY3000-NATIVE-HOLD-01", "RECOVERY3000-NATIVE-PLACE-01",
              "RECOVERY3000-NATIVE-INDEPENDENT_HOLD-01", "RECOVERY-TRAIN-LIVE-05")
    if followup:
        owners += ("V7-D0-EARLY-NATIVE-01",)
        for phase in ("HOLD", "PLACE", "INDEPENDENT_HOLD"):
            for i in range(2):
                ongoing = site / "V7-A500-CORE-01" / f"NATIVE-{i}-{phase}"
                if (ongoing / "control/LEASE.json").is_file():
                    require(all(read(ongoing / leaf)["status"] == "CLOSED" for leaf in
                                ("control/LEASE.json", "worker/control/LEASE.json")),
                            "V7 evaluation still owns its current phase")
    for run in owners:
        for leaf in ("control/LEASE.json", "worker/control/LEASE.json"):
            require(read(site/run/leaf)["status"] == "CLOSED", "Do not overlap an existing model or simulation owner")
    original = worker["sample_noise"]
    valid = read(pin(original["path"], original["sha256"]))["tensors"]
    cohorts = (*OLD_NOISE, FIRST_NOISE) if followup else OLD_NOISE
    old = [read(pin(site/path, checksum))["tensors"] for path, checksum in cohorts]
    require(all(len(rows) == 2*QUERIES for rows in old), "Incomplete prior Recovery noise")
    previous = valid + [row for cohort in old for row in cohort]
    ids = {r["noise_id"] for r in previous}
    hashes = {r["gpu_sha256"] for r in previous}

    def fingerprint(index):
        sample = jax.random.normal(jax.random.PRNGKey(index), (1,40,32), dtype=np.float32)
        return hashlib.sha256(np.asarray(sample).tobytes()).hexdigest()

    with closing(OwnerLock(Path(worker["neural_owner_lock"]))):
        require(not output.exists() and jax.default_backend() == "gpu" and len(jax.local_devices()) == 1,
                "Reserve one actual GPU sampling owner and a new output")
        sentinels = (valid[0], valid[len(valid)//2], valid[-1])
        require(all(fingerprint(r["noise_id"]) == r["gpu_sha256"] for r in sentinels),
                "CUDA PRNG output differs from the fixed sampler")
        rows = []
        for case in roster["cases"]:
            for query in range(QUERIES):
                index = (case["policy_seed"] + query) % 2**31
                checksum = fingerprint(index)
                require(index not in ids and checksum not in hashes, "A prospective draw overlaps original PLACE or Recovery")
                ids.add(index)
                hashes.add(checksum)
                rows.append(dict(case=case["id"],query=query,noise_id=index,gpu_sha256=checksum))
        require(len(rows) == 2*QUERIES, "Incomplete prospective CUDA range")
        write(output,dict(plan=PLAN,split="TRAIN",roster_sha256=roster_sha,worker_spec_sha256=WORKER,
                         prior_noise_manifest_sha256=original["sha256"],prior_recovery=[digest for _,digest in cohorts],
                         matched_archived_cuda_sentinels=len(sentinels),tensors=rows,model_forwards=0,
                         native_integrations=0,optimizer_updates=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("site","roster_path","worker_path","output"):
        cli.add_argument(field,type=Path)
    cli.add_argument("--followup",action="store_true")
    seal(**vars(cli.parse_args()))
