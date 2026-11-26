"""Seal the actual CUDA noise for the matched V4 long-run PLACE conditions."""

import argparse
from contextlib import closing
import hashlib
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write


REGISTRY_SHA256 = "86cde1a83f2cedefbba7321b528435304f8f2ec7b884b3981ed4527686a9f9d1"
STARTS = {701: 29_670_101, 702: 29_670_401}
QUERIES = 160


def main(registry, historical_place, historical_hold, closed_v3, worker_spec, destination):
    import jax
    import jax.numpy as jnp
    import numpy as np

    roster = [row for row in read(pin(registry, REGISTRY_SHA256))["conditions"] if row["block"] == "LONG_DEV6"]
    place = [row for row in roster if row["task"] == "ORIGINAL_PLACE"]
    hold = [row for row in roster if row["task"] == "ORIGINAL_MOBILE_HOLD"]
    wanted = [("P01", 701), ("P02", 702), ("P02", 701), ("P01", 702)]
    require(len(place) == 16 and len(hold) == 8 and all(
        [(row["slot"], row["noise_label"]) for row in place if row["model"] == model] == wanted
        and [(row["slot"], row["archived_noise_alias"]) for row in hold if row["model"] == model]
            == [("N02", "N02_A"), ("N02", "N02_B")] for model in ("PARENT", "M1000", "T1000", "M2000")),
        "The fixed original PLACE/HOLD comparison roster changed")
    reference = [read(path) for path in (historical_place, historical_hold, closed_v3)]
    lineage = {value["old_gpu_manifest_sha256"] for value in reference}
    require(len(lineage) == 1, "Existing paired noise manifests do not share the qualified CUDA source")
    prior_ids = {row["noise_id"] for value in reference for row in value["tensors"]}
    prior_hashes = {row["gpu_sha256"] for value in reference for row in value["tensors"]}
    lock = read(worker_spec)["neural_owner_lock"]

    with closing(OwnerLock(Path(lock))):
        require(not destination.exists() and jax.default_backend() == "gpu" and len(jax.local_devices()) == 1,
                "Generate the new noise once on the qualified CUDA runtime")
        rows = []
        for label, base in STARTS.items():
            for query in range(QUERIES):
                key = base + query
                require(key not in prior_ids, "New fixed seed collides with an archived noise key")
                value = np.asarray(jax.random.normal(jax.random.PRNGKey(key), (1, 40, 32), dtype=jnp.float32))
                digest = hashlib.sha256(value.tobytes()).hexdigest()
                require(digest not in prior_hashes, "New fixed noise collides with an archived input")
                rows.append(dict(noise_id=key, label=label, query=query, gpu_sha256=digest))
        require(len({row["gpu_sha256"] for row in rows}) == len(rows) == 320, "Duplicate realized CUDA noise")
        write(destination, dict(stage="LONG_DEV6_PLACE", registry_sha256=REGISTRY_SHA256,
              labels={str(k): v for k, v in STARTS.items()}, tensors=rows, hold_bank_reused=True,
              paired_models=["PARENT", "M1000", "T1000", "M2000"],
              old_gpu_manifest_sha256=lineage.pop(), historical_hold_sha256=sha(historical_hold),
              worker_spec_sha256=sha(worker_spec), model_forwards=0, training_updates=0, native_integrations=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("registry", "historical_place", "historical_hold", "closed_v3", "worker_spec", "destination"):
        cli.add_argument(key, type=Path)
    main(**vars(cli.parse_args()))
