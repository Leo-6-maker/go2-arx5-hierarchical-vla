"""Restore a B6 arm at its sealed step without restarting Adam, samples or warmup."""
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha

SCOPE = "V7_B6_SHARED_L2SP_MATCHED_V1"
ARMS = ("LAMBDA_ZERO", "SHARED_L2SP")


def source_run(binding, arm):
    from v3_main_place_learn import endpoint_files

    require(arm in ARMS, "Unknown B6 arm")
    ref = binding["resume"]
    root = Path(ref["path"])
    terminal = read(pin(root / "TERMINAL.json", ref["terminal_sha256"]))
    prior = read(pin(root / "BINDING.json", ref["binding_sha256"]))
    start = ref["step"]
    require(start in (100, 250) and start < binding["end_step"] <= 500
            and terminal["scope"] == prior["scope"] == binding["scope"] == SCOPE
            and terminal["status"] == "PAIR_COMPLETE" and terminal["end_step"] == start
            and prior["source_pair_binding_sha256"] == binding["source_pair_binding_sha256"]
            and prior["calibration"] == binding["calibration"], "Continuation changed its matched experiment")
    run = root / arm
    result = read(pin(run / "TERMINAL.json", terminal["arms"][arm]["terminal_sha256"]))
    seal = read(run / f"STEP{start:04d}_SEALED.json")
    endpoint = run / f"STEP{start:04d}"
    require(result["arm"] == arm and result["scope"] == SCOPE and result["updates_not_in_ledger"] == 0
            and result["status"] == "FORMAL_ENDPOINT_COMPLETE" and result["checkpoint"] == seal
            and seal["step"] == seal["bank_cursor"] == start
            and seal["files"] == endpoint_files(endpoint), "Source arm is incomplete, wrong or modified")
    rows = [json.loads(line) for line in pin(run / "UPDATES.jsonl", result["ledger_sha256"]).read_text().splitlines()]
    first = prior.get("resume", {}).get("step", 0)
    require([row["step"] for row in rows] == list(range(first+1, start+1)), "Source update interval has gaps")
    return endpoint, seal, start


def restore_learner(learner, binding, arm):
    import jax
    import orbax.checkpoint as ocp
    from v4_long_probe import snapshot

    endpoint, seal, start = source_run(binding, arm)
    require(learner.step == 0 and learner.cursors == dict.fromkeys(learner.batch_by_domain, 0)
            and not learner.queued, "Restore only a fresh, unconsumed learner")
    with ocp.PyTreeCheckpointer() as reader:
        saved = reader.restore(endpoint.resolve(), item=snapshot(learner))
    require(int(saved["step"]) == start and set(saved["parameters"]) == set(learner.parameters)
            and set(saved["optimizer"]) == set(learner.optimizer), "Incomplete checkpoint state")
    for name, values in saved["parameters"].items():
        learner.parameters[name].replace_by_pure_dict(values)
    learner.optimizer, learner.rng, learner.step = saved["optimizer"], saved["rng"], start
    require({name: int(state[0].count) for name, state in learner.optimizer.items()} == seal["adam_counts"]
            == dict.fromkeys(learner.optimizer, start), "Adam clock was reset")
    require(np.array_equal(np.asarray(learner.rng), np.asarray(jax.random.PRNGKey(529202611))),
            "Original immutable draw key changed")
    actual = snapshot(learner)
    require(jax.tree.structure(actual) == jax.tree.structure(saved)
            and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
                    zip(jax.tree.leaves(actual), jax.tree.leaves(saved), strict=True)),
            "Restored parameters, Adam moments, step or RNG changed")
    return dict(arm=arm, step=start, endpoint_sha256=sha(endpoint.with_name(endpoint.name+"_SEALED.json")),
                restored_snapshot_equal=True, adam_counts=seal["adam_counts"],
                next_step=start+1, next_lr=float(learner.rates[start]),
                next_domain_cursors={key: start*size for key, size in learner.batch_by_domain.items()})


def next_input(learner, original, start):
    from v7_formal_worker import input_evidence

    row = json.loads((Path(original["source_pair"]) / "CONTROL_NEUTRAL/UPDATES.jsonl").read_text().splitlines()[start])
    evidence = input_evidence(learner, start)
    require(row["step"] == start+1 and row["lr"] == float(learner.rates[start]), "Wrong original learning-rate position")
    for domain, size in learner.batch_by_domain.items():
        actual = evidence[domain]
        require(actual["indices"] == row["components"][domain]["visit_sha256"]
                and actual["cursor"] == start*size
                and [actual["noise"], actual["tau"]] == row["noise"][domain],
                "Next batch, cursor or actual CUDA arrays differ from the original course")
    require(evidence["auxiliary"] == row["noise"]["auxiliary"][0], "Next auxiliary noise changed")
    return evidence
