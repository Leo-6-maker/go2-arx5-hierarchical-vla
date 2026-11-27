"""Seal the actual CUDA sampling noise for the two Recovery TRAIN cases."""

import argparse
from contextlib import closing
import hashlib
from pathlib import Path

from scripts.pi0292.durable import OwnerLock, write
from v64_recovery_roster import PLAN, QUERIES, ROSTER, fixed

QUALIFIED_WORKER = "361709e35c2bf427d5644e7af37a925014ab52861c075d4782749afd371d6bfc"
TRAIN_ROSTER = "e9b7552d4c10779f4c87a04ef23c252ee2f599a69a24df5d8f94624166ad6cba"


def seal(roster, worker, previous, output):
    import jax
    import numpy as np

    registered = fixed(roster, TRAIN_ROSTER)
    runtime = fixed(worker, QUALIFIED_WORKER)
    binding = runtime["sample_noise"]
    historical = fixed(previous, binding["sha256"])
    if (registered["plan"] != PLAN or registered["split"] != "TRAIN" or
            tuple(case["id"] for case in registered["cases"]) != tuple(row[0] for row in ROSTER)
            or not all(case["maximum_queries"] == QUERIES for case in registered["cases"])
            or str(previous.resolve(strict=True)) != binding["path"]):
        raise ValueError("Require the exact new TRAIN roster and the previous qualified CUDA runtime")
    prior_ids = {row["noise_id"] for row in historical["tensors"]}
    prior_bytes = {row["gpu_sha256"] for row in historical["tensors"]}

    def value_sha(noise_id):
        value = np.asarray(jax.random.normal(jax.random.PRNGKey(noise_id), (1, 40, 32), dtype=np.float32))
        return hashlib.sha256(value.tobytes()).hexdigest()

    with closing(OwnerLock(Path(runtime["neural_owner_lock"]))):
        if output.exists() or jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
            raise ValueError("Require one actual qualified CUDA device and an unused destination")
        reference = historical["tensors"]
        checks = (reference[0], reference[len(reference)//2], reference[-1])
        if any(value_sha(row["noise_id"]) != row["gpu_sha256"] for row in checks):
            raise ValueError("Current CUDA sampling differs from the qualified real bytes")
        rows = []
        for case in registered["cases"]:
            for query in range(QUERIES):
                noise_id = (case["policy_seed"] + query) % 2**31
                fingerprint = value_sha(noise_id)
                if noise_id in prior_ids or fingerprint in prior_bytes:
                    raise ValueError("Recovery noise must remain independent from prior PLACE noise")
                prior_ids.add(noise_id)
                prior_bytes.add(fingerprint)
                rows.append(dict(case=case["id"], query=query, noise_id=noise_id, gpu_sha256=fingerprint))
        write(output, dict(plan=PLAN, split="TRAIN", roster_sha256=TRAIN_ROSTER, worker_spec_sha256=QUALIFIED_WORKER, matched_archived_cuda_sentinels=len(checks),
                 prior_noise_manifest_sha256=binding["sha256"], tensors=rows, model_forwards=0,
                 optimizer_updates=0, native_integrations=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("roster", "worker", "previous", "output"):
        parser.add_argument(name, type=Path)
    seal(**vars(parser.parse_args()))
