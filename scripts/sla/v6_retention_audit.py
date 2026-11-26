"""Independently restore bounded MAIN retention checkpoints on Corelab."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import read, require, sha
from v3_main_place_learn import MAIN, endpoint_files
from v4_long_export import export


GROUPS = ("shared", "target", "robodojo", "libero")
SEED = 38002001  # Original V4 MAIN long-course seed.


def restore(endpoint, step, target_only):
    import jax
    import orbax.checkpoint as ocp

    require(jax.default_backend() == "cpu", "Endpoint audit stays on Corelab CPU")
    with ocp.PyTreeCheckpointer() as saver:
        metadata = saver.metadata(endpoint).tree
        array = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata)
        state = saver.restore(endpoint, args=ocp.args.PyTreeRestore(item=metadata, restore_args=array, transforms={}))
    require(int(state["step"]) == step and set(state["parameters"]) == set(state["optimizer"]) == set(GROUPS)
            and np.array_equal(state["rng"], np.asarray(jax.random.PRNGKey(SEED))),
            "Saved parameters, optimizer, clock, or training RNG differ")
    param, optimizer = [jax.tree.leaves(state[key]) for key in ("parameters", "optimizer")]
    require(all(np.isfinite(x).all() for x in (*param, *optimizer)), "A recovered tensor is nonfinite")
    counts = {}
    for name in GROUPS:
        values = [int(x) for x in jax.tree.leaves(state["optimizer"][name])
                  if np.asarray(x).ndim == 0 and np.issubdtype(np.asarray(x).dtype, np.integer)]
        expected = 0 if target_only and name in ("robodojo", "libero") else step
        require(values == [expected], f"Adam counter disagrees for {name}")
        counts[name] = values[0]
    return dict(parameter_leaves=len(param), optimizer_leaves=len(optimizer), adam_counters=counts,
                rng_restored=True)


def audit(run, step, template, output):
    contract_file = run / "RUN_CONTRACT.json"
    receipt_file = run / f"STEP{step:04d}_SEALED.json"
    contract, receipt = read(contract_file), read(receipt_file)
    role = contract["role"]
    target_only = role == "R3_TARGET_GRADIENT_ONLY"
    require(contract["parent"]["parent_sha256"] == MAIN and step in (100, 200)
            and receipt["step"] == step and contract["active_auxiliary"] == 4
            and contract["source_updates_enabled"] == (not target_only),
            "The snapshot is not one of the controlled retention milestones")
    endpoint = run / f"STEP{step:04d}"
    require(Path(receipt["checkpoint_path"]).name == endpoint.name and endpoint_files(endpoint) == receipt["files"],
            "Transferred checkpoint bytes no longer match the scheduler-side receipt")
    if step == 200:
        terminal = read(run / "TERMINAL.json")
        require(terminal["status"] == "FORMAL_RETENTION_200" and terminal["updates"] == step
                and terminal["role"] == role and terminal["contract_sha256"] == sha(contract_file),
                "The complete 200-update experiment lacks its original terminal record")
    population = Counter()
    replay_visits = Counter()
    reexpressed_visits = 0
    timing = []
    target = {"target": 64, "auxiliary": 4}
    expected = target if target_only else dict(target, robodojo=32, libero=32)
    with (run / "UPDATES.jsonl").open() as logfile:
        for index in range(1, step + 1):
            row = json.loads(logfile.readline())
            visits = {k: v["visits"] for k, v in row["components"].items()}
            require(row["step"] == index and visits == expected and row["lr"] > 0
                    and math.isfinite(row["lr"]) and row["global_gradient_norm"] > 0
                    and math.isfinite(row["global_gradient_norm"]), "The actual optimizer ledger is incomplete")
            population.update(visits)
            if role == "R2_PHASE_MATCHED_REPLAY":
                require(sum(row["replay_visits"].values()) == 8, "Wrong realized blue-left visits")
                replay_visits.update(row["replay_visits"])
            if role == "R4_REEXPRESSED_COMMANDS":
                require(row["reexpressed_geometry_visits"] == 64, "Wrong realized rotated/displaced supervision visits")
                reexpressed_visits += row["reexpressed_geometry_visits"]
            timing.append(row["seconds"])
    result = dict(step=step, formal=False, complete_training_endpoint=step == 200,
                  parent_sha256=MAIN, role=role, endpoint=str(endpoint),
                  receipt_sha256=sha(receipt_file), contract_sha256=sha(contract_file), actual_visits=dict(population),
                  actual_replay_visits=dict(replay_visits),
                  actual_reexpressed_geometry_visits=reexpressed_visits,
                  warm_p50_s=float(np.median(timing[2:])), warm_p95_s=float(np.percentile(timing[2:], 95)),
                  checkpoint_bytes=sum((endpoint / p).stat().st_size for p in receipt["files"]),
                  **restore(endpoint, step, target_only))
    result["provider_manifest_sha256"] = export(result, template, output, diagnostic=True)
    with output.with_name(output.name + "_READBACK.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("step", type=int, choices=(100, 200))
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    audit(**vars(parser.parse_args()))
