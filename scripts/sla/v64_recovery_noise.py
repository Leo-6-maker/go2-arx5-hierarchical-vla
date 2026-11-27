"""Seal the actual CUDA sampling noise for the two Recovery TRAIN cases."""

import argparse
from contextlib import closing
import hashlib
from pathlib import Path

from scripts.pi0292.durable import OwnerLock, write
from v64_recovery_cohorts import FIRST_NOISE, FIRST_ROSTER, SECOND_ROSTER, read_roster
from v64_recovery_roster import PLAN, QUERIES, fixed

QUALIFIED_WORKER = "361709e35c2bf427d5644e7af37a925014ab52861c075d4782749afd371d6bfc"


def seal(roster, worker, previous, output, *, prior_recovery=None):
    import jax
    import numpy as np

    registered, cohort = read_roster(roster)
    runtime = fixed(worker, QUALIFIED_WORKER)
    binding = runtime["sample_noise"]
    historical = fixed(previous, binding["sha256"])
    if (str(previous.resolve(strict=True)) != binding["path"]
            or (cohort == SECOND_ROSTER) != (prior_recovery is not None)):
        raise ValueError("The required old CUDA runtime or prior Recovery noise binding changed")
    prior_ids = {row["noise_id"] for row in historical["tensors"]}
    prior_bytes = {row["gpu_sha256"] for row in historical["tensors"]}
    history = {}
    if prior_recovery is not None:
        first = fixed(prior_recovery, FIRST_NOISE)
        if first["roster_sha256"] != FIRST_ROSTER or len(first["tensors"]) != 2 * QUERIES:
            raise ValueError("Require the complete first Recovery noise bank")
        prior_ids.update(row["noise_id"] for row in first["tensors"])
        prior_bytes.update(row["gpu_sha256"] for row in first["tensors"])
        history["prior_recovery_noise_sha256"] = FIRST_NOISE

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
        write(output, dict(plan=PLAN, split="TRAIN", roster_sha256=cohort, worker_spec_sha256=QUALIFIED_WORKER, matched_archived_cuda_sentinels=len(checks),
                 prior_noise_manifest_sha256=binding["sha256"], tensors=rows, model_forwards=0,
                 optimizer_updates=0, native_integrations=0, **history))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("roster", "worker", "previous", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--prior-recovery", type=Path)
    seal(**vars(parser.parse_args()))
