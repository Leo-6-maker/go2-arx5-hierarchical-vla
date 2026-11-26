"""Independently restore and audit the complete formal V6 endpoint."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import read, require, sha
from v3_main_place_learn import endpoint_files
from v4_long_export import export
from v6_dual_course import PARENT, SEED
from v6_dual_source_worker import inputs_for


def audit(run, custody, short, bank, source, target):
    import jax
    import orbax.checkpoint as ocp
    from scripts.pi0292.durable import write

    require(jax.default_backend() == "cpu", "Use a separate read-only CPU restore")
    terminal_file = run / "TERMINAL.json"
    terminal = read(terminal_file)
    step = int(terminal["updates"])
    contract = run / "RUN_CONTRACT.json"
    settings = terminal["settings"]
    require(terminal["status"] == "FORMAL_1000" and step == 1000
            and sha(contract) == terminal["contract_sha256"] and read(contract)["settings"] == settings
            and read(contract)["parent"]["parent_sha256"] == PARENT,
            "Only the complete formal V6 endpoint is eligible")
    _, readers, banks, slots, _, reproduced = inputs_for(
        custody, short, bank, source, settings["source_extension_manifest"])
    require(reproduced == settings, "Source and target visit banks changed after execution")
    counts = Counter()
    new_dual, new_single = 0, 0
    timing = []
    with (run / "UPDATES.jsonl").open() as ledger:
        for i, line in enumerate(ledger):
            row = json.loads(line)
            require(i < step and row["step"] == i + 1 and math.isfinite(row["global_gradient_norm"])
                    and row["global_gradient_norm"] > 0 and math.isfinite(row["lr"]) and row["lr"] > 0,
                    "Unsealed, duplicated or nonfinite optimizer update")
            details = row["components"]
            visits = {k: v["visits"] for k, v in details.items()}
            require(visits == {**settings["visits"], "auxiliary": 4} and details["auxiliary"]["slots"] == slots[i].tolist(),
                    "Realized learner batch or actual auxiliary slots differ")
            for domain in banks:
                require(details[domain]["visit_sha256"] == hashlib.sha256(banks[domain][i].tobytes()).hexdigest(),
                        "Real learner visit identities changed")
            n = int(np.count_nonzero(banks["robodojo"][i] >= readers["robodojo"].base_length))
            require(n == row["v6_new_dual_source_visits"]
                    and row["v6_new_single_source_visits"] == visits.get("single", 0), "New source exposure differs")
            counts.update(visits)
            new_dual += n
            new_single += row["v6_new_single_source_visits"]
            timing.append(row["seconds"])
    require(len(timing) == step and new_dual == terminal["new_dual_source_visits"] > 0
            and new_single == terminal["new_single_source_visits"], "Full course source evidence is missing")
    path = Path(terminal["checkpoint"]["path"]).resolve(strict=True)
    require(path.parent == run.resolve(strict=True) and path.name == "STEP1000"
            and endpoint_files(path) == terminal["checkpoint"]["files"], "Complete checkpoint archive changed")
    with ocp.PyTreeCheckpointer() as saver:
        metadata = saver.metadata(path).tree
        typed = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata)
        state = saver.restore(path, args=ocp.args.PyTreeRestore(item=metadata, restore_args=typed, transforms={}))
    groups = {"shared", *settings["visits"]}
    require(int(state["step"]) == step and set(state["parameters"]) == set(state["optimizer"]) == groups
            and np.array_equal(state["rng"], np.asarray(jax.random.PRNGKey(SEED))), "Full generation identity changed")
    for value in jax.tree.leaves((state["parameters"], state["optimizer"])):
        require(np.isfinite(value).all(), "Nonfinite complete recovered tensor")
    adam = [int(value) for value in jax.tree.leaves(state["optimizer"])
            if np.asarray(value).ndim == 0 and np.issubdtype(np.asarray(value).dtype, np.integer)]
    require(adam == [step] * len(groups), "Independent Adam counters disagree")
    proof = dict(step=step, formal=True, parent_sha256=PARENT, endpoint=str(path),
                 receipt_sha256=sha(terminal_file), contract_sha256=sha(contract), source_recipe=settings["recipe"],
                 actual_visits=dict(counts), actual_new_dual_visits=new_dual, actual_new_single_visits=new_single,
                 warm_p50_s=float(np.median(timing[2:])), warm_p95_s=float(np.percentile(timing[2:], 95)),
                 parameter_leaves=len(jax.tree.leaves(state["parameters"])),
                 optimizer_leaves=len(jax.tree.leaves(state["optimizer"])), adam_counters=adam, rng_restored=True,
                 checkpoint_bytes=sum((path / f).stat().st_size for f in terminal["checkpoint"]["files"]))
    del state
    proof["provider_manifest_sha256"] = export(proof, short / "incoming/reference", target,
        schema="SLA_V6_SOURCE_INFERENCE", formal_role="SOURCE1K_V6")
    write(target.parent / "SOURCE1K_ENDPOINT_READBACK.json", proof)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "custody", "short", "bank", "source", "target"):
        parser.add_argument(name, type=Path)
    print(json.dumps(audit(**vars(parser.parse_args()))))
