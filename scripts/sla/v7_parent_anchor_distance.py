"""Measure shared and Go2-private distance from complete, restored B6 providers."""
import argparse
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v7_parent_anchor_endpoint import ARMS, SCOPE


def distance(root, parent, output):
    require(not output.exists(), "Keep the original parameter measurement")
    proof = read(root / "CORE-PROVIDERS/ACTUAL.json")
    require(proof["scope"] == SCOPE and proof["complete_training_endpoint"]
            and proof["step"] in (100, 250, 500), "Require the complete independent restore")
    original = read(parent / "CHECKPOINT.json")
    before = {row["key"]: row for row in original["arrays"]}
    measured = {}
    for arm in ARMS:
        source = root / "CORE-PROVIDERS" / arm
        current = read(pin(source / "CHECKPOINT.json", proof["arms"][arm]["provider_sha256"]))
        require(current["phase"] == proof["step"] and current["b6_arm"] == arm
                and {row["key"] for row in current["arrays"]} == set(before), "Changed inference array ABI")
        groups = {role: dict(squared_L2=0., parent_squared_L2=0., leaves=[]) for role in ("shared", "target")}
        for row in current["arrays"]:
            old = before[row["key"]]
            require(row["shape"] == old["shape"] and row["dtype"] == old["dtype"], "Changed array shape or dtype")
            if "lora" in row["key"].lower():
                require(row["sha256"] == old["sha256"], "Frozen LoRA changed")
                continue
            role, = (role for role in groups if row["key"].startswith(f"['parameters']['{role}']"))
            a = np.load(pin(source / row["file"], row["sha256"]), mmap_mode="r", allow_pickle=False).reshape(-1)
            b = np.load(pin(parent / old["file"], old["sha256"]), mmap_mode="r", allow_pickle=False).reshape(-1)
            squared, parent_squared = 0., 0.
            for first in range(0, a.size, 1 << 20):
                x, y = (np.asarray(v[first:first+(1 << 20)], dtype=np.float64) for v in (a, b))
                require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite restored weights")
                squared += float(np.sum(np.square(x-y)))
                parent_squared += float(np.sum(np.square(y)))
            leaf = dict(key=row["key"], squared_L2=squared, parent_squared_L2=parent_squared,
                        relative_L2=float(np.sqrt(squared/parent_squared)))
            groups[role]["leaves"].append(leaf)
            groups[role]["squared_L2"] += squared
            groups[role]["parent_squared_L2"] += parent_squared
        require([len(groups[role]["leaves"]) for role in ("shared", "target")] == [11, 8],
                "The registered trainable shared or Go2 partition changed")
        measured[arm] = groups
    control, treatment = (measured[arm]["shared"]["squared_L2"] for arm in ARMS)
    require(control > 0, "No actual control displacement")
    write(output, dict(complete_step=proof["step"], arms=measured,
          shared_squared_distance_reduction=1-treatment/control,
          provider_proof_sha256=sha(root/"CORE-PROVIDERS/ACTUAL.json"),
          optimizer_updates=0, native_integrations=0))
    print(output, sha(output), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "parent", "output"):
        cli.add_argument(name, type=Path)
    distance(**vars(cli.parse_args()))
