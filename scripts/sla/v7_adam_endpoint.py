"""Reconstruct the final proposed AdamW direction from each sealed V7 moment tree."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require
from scripts.pi0292.durable import write
from v7_core_endpoint import ARMS, BANK, PLAN, PROJECTION, STEPS


def directional_metrics(parameters, parent_rows, parent, moments, rate, config):
    numerator = delta_sq = drift_sq = 0.0
    for key, final in parameters.items():
        if key not in parent_rows:
            raise ValueError(f"The R4 parent did not contain retained parameter {key}")
        reference = parent_rows[key]
        origin = np.load(pin(parent / reference["file"], reference["sha256"]), mmap_mode="r", allow_pickle=False)
        first, second = moments["mu"][key], moments["nu"][key]
        require(first.shape == second.shape == final.shape == origin.shape, "Adam/parent shape differs")
        vectors = [values.reshape(-1) for values in (final, first, second, origin)]
        for start in range(0, final.size, 1 << 20):
            end = start + (1 << 20)
            saved, mu, nu, base = (np.asarray(values[start:end], dtype=np.float64) for values in vectors)
            direction = (mu / (1 - config["beta1"]**STEPS)) / (
                np.sqrt(nu / (1 - config["beta2"]**STEPS)) + config["eps"])
            earlier = (saved + rate * direction) / (1 - rate * config["weight_decay"])
            proposed = -rate * (direction + config["weight_decay"] * earlier)
            drift = saved - base
            require(np.isfinite(proposed).all() and np.isfinite(drift).all(), "Nonfinite step or parameter drift")
            numerator += float(proposed @ drift)
            delta_sq += float(proposed @ proposed)
            drift_sq += float(drift @ drift)
    scale = math.sqrt(delta_sq * drift_sq)
    return dict(proposed_update_l2=math.sqrt(delta_sq), R4_parameter_drift_l2=math.sqrt(drift_sq),
                cosine_with_R4_parameter_drift=numerator / scale if scale > 0 else None)


def run(training, providers, pair_sha, parent, bank, output):
    from flax.traverse_util import flatten_dict
    from go2arx5_vla.sla_adapter.adapt5k_training import OPTIMIZER
    import jax
    import orbax.checkpoint as ocp
    from v65_recovery_endpoint import r4_provider

    require(jax.default_backend() == "cpu", "Saved moment analysis runs on Corelab CPU")
    pair = read(pin(providers / "CORE-PAIR-ACTUAL.json", pair_sha))
    require(pair["plan"] == PLAN and pair["step"] == STEPS and pair["formal_updates"] == 1000,
            "Require the complete verified V7-A pair")
    manifest = read(pin(bank / "MATCHED_INPUTS.json", BANK))
    with np.load(pin(bank / "BANK.npz", manifest["bank_sha256"]), allow_pickle=False) as planned:
        rate = float(planned["rates"][STEPS - 1])
    _, rows, _, _ = r4_provider(parent)
    arms = {}
    for arm in ARMS:
        path = training / arm
        seal = read(pin(path / "STEP0500_SEALED.json", pair["arms"][arm]["endpoint_sha256"]))
        terminal = read(pin(path / "TERMINAL.json", pair["arms"][arm]["training_terminal_sha256"]))
        recorded = pin(path / "UPDATES.jsonl", terminal["optimizer_ledger_sha256"])
        require(json.loads(recorded.read_bytes().splitlines()[-1])["lr"] == rate, "Final realized LR changed")
        groups = ("shared", "target")
        endpoint = path / "STEP0500"
        with ocp.PyTreeCheckpointer() as reader:
            metadata = reader.metadata(endpoint).tree
            wanted = {key: {name: metadata[key][name] for name in groups} for key in ("parameters", "optimizer")}
            args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), wanted)
            saved = reader.restore(endpoint, args=ocp.args.PyTreeRestore(item=wanted, restore_args=args, transforms={}))
        parts = {}
        for name in groups:
            moments = saved["optimizer"][name][0]
            require(int(moments["count"]) == seal["adam_counts"][name] == STEPS, "Wrong saved moment clock")
            parameters = flatten_dict(saved["parameters"][name])
            if name == "target":
                parameters.pop((PROJECTION,))
            originals = {key[1:]: row for key, row in rows.items() if key[0] == name and key[1:] in parameters}
            flat = {}
            for field in ("mu", "nu"):
                leaves = flatten_dict(moments[field])
                require(all(key[-1] == "value" for key in leaves), "Unexpected saved Adam state wrapper")
                flat[field] = {key[:-1]: leaf for key, leaf in leaves.items()}
            parts[name] = directional_metrics(parameters, originals, parent, flat, rate, OPTIMIZER)
        arms[arm] = parts
        del saved, parameters, flat, moments
    write(output, dict(plan=PLAN, pair_sha256=pair_sha, step=STEPS, learning_rate=rate, optimizer=OPTIMIZER,
                      arms=arms, scope="ALGEBRAIC_PROPOSED_LAST_UPDATE_FROM_EXACT_SAVED_MOMENTS",
                      model_calls=0, optimization_updates=0, physical_integrations=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "providers", "parent", "bank", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("pair_sha")
    run(**vars(parser.parse_args()))
