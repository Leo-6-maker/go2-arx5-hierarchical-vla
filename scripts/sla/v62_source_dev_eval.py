"""Compare fixed-roster free generations from R4 initialization and SOURCE on Corelab."""

import argparse
from collections import defaultdict
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

from m38_gpu_preflight_v1 import bind_imports, check_gpu, pin, read, require, sha
from v3_main_place_learn import endpoint_files
from v61_source_course import MAIN, R4
from v62_source_admission import V62
from v62_source_dev_roster import R48, fingerprint
from v62_source_native import V65, checked_manifest, reference, source_role


DEV_ROSTER = "a6e9a33717d9938ac1870ce479947d3289045cf601ea0977a2cf80d082e80292"
R4_READBACK = "820b1ef46077355da86c85c448bd83e2b552ab199f138439d057e82f53f79741"
PAIRED_R4_DEV = "e05d16597cc0386c4c07249f68ab398a2d8986b8eee054e8c84e5f182a86ba17"


def restore_groups(endpoint, names, step):
    import jax
    import orbax.checkpoint as ocp

    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(endpoint).tree
        selected = {name: metadata["parameters"][name] for name in names}
        args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), selected)
        restored = checkpointer.restore(endpoint, args=ocp.args.PyTreeRestore(
            item=dict(parameters=selected, step=metadata["step"]),
            restore_args=dict(parameters=args, step=ocp.RestoreArgs(restore_type=int)), transforms={}))
    require(restored["step"] == step and set(restored["parameters"]) == set(names), "Source parameter restore differs")
    return restored["parameters"]


def load_inputs(dual, single, video, roster):
    from go2arx5_vla.sla_adapter.scale20k_data import RobodojoDataset
    from openpi_client.image_tools import resize_with_pad
    from v6_ville_reader import SingleArmReader

    sealed = read(pin(roster, DEV_ROSTER))
    require(sealed["plan"] == V62 and sealed["split"] == "SOURCE_DEV" and sealed["evaluation_only"],
            "Source metrics require the preregistered independent split")
    dual_ref = dict(path=str(dual / "MANIFEST.private.json"), sha256=R48)
    readers = dict(robodojo=RobodojoDataset(dual, dual_ref, split="SOURCE_DEV", resize_with_pad=resize_with_pad),
                   single=SingleArmReader(single, sealed["source_manifest_sha256"]["single"], video,
                                          split="SOURCE_DEV", resize=resize_with_pad))
    return {domain: (reader, sealed["actual_readback"][domain]["rows"]) for domain, reader in readers.items()}


def summarize(rows, candidate_key="source"):
    episodes, tasks = defaultdict(list), defaultdict(list)
    for row in rows:
        scores = [row["mae"][name] for name in ("r4_init", candidate_key)]
        episodes[row["episode"]].append(scores)
        tasks[row["task"]].append(scores)
    matched = np.asarray([np.mean(values, axis=0) for values in episodes.values()])
    parent, candidate = matched.mean(axis=0)
    return dict(windows=len(rows), episodes=len(episodes), r4_init_mae=float(parent),
                episodes_improved=int(np.count_nonzero(matched[:, 1] < matched[:, 0])),
                task_mae={task: np.mean(data, axis=0).tolist() for task, data in tasks.items()},
                **{candidate_key + "_mae": float(candidate)})


def evaluate(worker, endpoint, retention, roster, dual, single, single_video, out, step, *, adapt_baseline=None,
             recovery_baseline=None):
    spec = read(worker)
    bind_imports(spec)

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter import batch_v2, full_h40_v6, scale20k_model
    from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
    from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash
    from go2arx5_vla.sla_adapter.scale20k_training import host_batch
    from openpi.models import pi0
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.sla.core_hopper_native_scope import accepted
    from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields
    from v6_dual_course import initialize_single

    _, gate, receiver = accepted(spec)
    plan, candidate, old_rows = V62, "source", {}
    require(not (adapt_baseline and recovery_baseline), "Declare one candidate course")
    if adapt_baseline:
        from v64_adapt_endpoint import PLAN as ADAPT_PLAN

        plan, candidate = ADAPT_PLAN, "adapt"
    if recovery_baseline:
        plan, candidate = V65, "recovery"
    reused = recovery_baseline or adapt_baseline
    if reused:
        past = read(pin(reused, PAIRED_R4_DEV))
        require(past["plan"] == V62 and past["model_forwards"] == 84
                and past["roster"]["sha256"] == DEV_ROSTER
                and past["physics_ticks"] == past["optimizer_updates"] == 0,
                "The matched, previously evaluated R4 source baseline is required")
        old_rows = {name: {row["index"]: row for row in data["rows"]} for name, data in past["domains"].items()}
    parent = read(pin(Path(gate["package"]["path"]) / "CHECKPOINT.json", gate["package"]["sha256"]))
    audit = reference(endpoint / "CPU-ENDPOINT-READBACK.json")
    folder = endpoint / "PROVIDER"
    provider = dict(path=str(folder), sha256=sha(folder / "CHECKPOINT.json"))
    model = checked_manifest(provider, parent, dict(v4_model_role=source_role(step, plan), source_step=step,
                            v62_plan=plan, source_audit=audit))
    r4 = read(pin(retention / "R4_STEP0200_PROVIDER_READBACK.json", R4_READBACK))
    baseline = Path(r4["endpoint"])
    bound = read(pin(baseline.parent / "STEP0200_SEALED.json", R4))
    require(endpoint_files(baseline) == bound["files"] and r4["parent_sha256"] == MAIN,
            "The exact R4 baseline training bytes are required")
    run = endpoint / ("A250-FORMAL-R1/run" if adapt_baseline else "run")
    checkpoint = run / f"STEP{step:04d}"
    if recovery_baseline:
        verified = read(pin(audit["path"], audit["sha256"]))
        terminal = read(pin(run / "TERMINAL.json", verified["training_terminal_sha256"]))
        require(terminal["status"] == "FORMAL_V65_1K_COMPLETE" and terminal["formal_updates"] == step
                and sha(run / f"STEP{step:04d}_SEALED.json") == model["training_receipt_sha256"],
                "V6.5 restore and formal terminal disagree")
    else:
        terminal = read(pin(run / "TERMINAL.json", model["training_receipt_sha256"]))
        require(Path(terminal["checkpoint"]["path"]).name == checkpoint.name, "Wrong SOURCE checkpoint name")
    require(endpoint_files(checkpoint) == terminal["checkpoint"]["files"], "Wrong SOURCE training bytes")
    inputs = load_inputs(dual, single, single_video, roster)

    with closing(OwnerLock(Path(spec["neural_owner_lock"]))):
        require(sys.executable == spec["python"] and os.uname().nodename == spec["host"] and os.getuid() == spec["uid"]
                and all(os.environ.get(key) == spec["gpu_environment"][key] for key in
                        ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION")),
                "Source evaluation requires the qualified Corelab CUDA worker")
        gpu = check_gpu(spec["GPU_uuid"], spec["allocator_limit_bytes"], spec["minimum_host_available_bytes"])
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1, "Expected the registered Corelab GPU")
        out.mkdir(mode=0o700)
        jax.config.update("jax_compilation_cache_dir", str(out / "jit-cache"))
        maximum_forwards = 42 if reused else 84
        write(out / "START.json", dict(plan=plan, step=step, provider=provider, roster=reference(roster), gpu=gpu,
                                      maximum_forwards=maximum_forwards))
        asset = receiver["model"]
        tokenizer_path = pin(receiver["tokenizer"]["path"], model["tokenizer_sha256"])
        require(asset["parent_manifest"]["sha256"] == model["foundation_sha256"], "Frozen foundation asset changed")
        graphs, template, frozen, identity = scale20k_model.load(asset["parent_manifest"], asset["parent_params"])
        require(identity["frozen_fingerprint"] == model["frozen_model"]
                and canonical_hash(identity["inventory"]) == model["partition_sha256"], "Frozen SOURCE model changed")
        tokenizer = Tokenizer(str(tokenizer_path))
        if recovery_baseline:
            from flax.traverse_util import flatten_dict, unflatten_dict
            from v65_recovery_endpoint import r4_provider

            _, _, lora, digest = r4_provider(retention / "R4_STEP0200_PROVIDER")
            require(digest == model["parent_lora_sha256"], "R4 LoRA bytes differ from the qualified endpoint")
            values = restore_groups(checkpoint, ("shared", "robodojo"), step)
            inherited = {key[1:]: array for key, array in lora.items()}
            active_shared = flatten_dict(values["shared"])
            require(not active_shared.keys() & inherited.keys(), "Frozen LoRA must stay outside the optimizer")
            values["shared"] = unflatten_dict(active_shared | inherited)
            baseline_target = restore_groups(baseline, ("target",), 200)["target"]
            values["target"] = baseline_target
            roles = {candidate: values}
        else:
            roles = {candidate: restore_groups(checkpoint, ("shared", "single", "robodojo"), step)}
        if not reused:
            roles = dict(r4_init=restore_groups(baseline, ("shared", "target", "robodojo"), 200), **roles)
        fields, _ = extracted_fields(batch_v2, pi0)
        output = {}
        started = time.monotonic()
        for domain, (reader, rows) in inputs.items():
            graph_role = "target" if domain == "single" else domain
            sample = full_h40_v6.make_sampler(graphs[graph_role], fields, rematerialize=False, precision="highest")
            active = {}
            for role, values in roles.items():
                groups = []
                name = "target" if domain == "single" and (role == "r4_init" or recovery_baseline) else domain
                for partition in ("shared", name):
                    state = jax.tree.map(lambda value: value, template["shared" if partition == "shared" else graph_role])
                    state.replace_by_pure_dict(values[partition])
                    groups.append(state)
                private = initialize_single(groups[1]) if name == "target" else groups[1]
                active[role] = nnx.State.merge(groups[0], private)
            measured = []
            for row in rows:
                require(time.monotonic() - started < 1800 and not (out / "STOP.json").exists(), "Source evaluation lease ended")
                index, horizon, dimensions = (row[key] for key in ("index", "horizon", "dimensions"))
                item = reader[index]
                require(reader.rows[index] == (row["episode"], row["anchor"]) and fingerprint(item, horizon) == row["hashes"],
                        "Source observation, action, mask or RGB differs")
                obs, actions = host_batch(reader, tokenizer, graph_role, [index], [index], training=False)
                noise = jax.random.normal(jax.random.PRNGKey(row["evaluation_noise_seed"]), (1, horizon, 32), dtype=jnp.float32)
                target = np.asarray(actions)[0, :, :dimensions]
                record = dict(row, actual_noise_sha256=hashlib.sha256(np.asarray(noise).tobytes()).hexdigest(),
                              mae={}, prediction_sha256={})
                if reused:
                    prior = old_rows[domain][index]
                    require(all(prior[key] == val for key, val in row.items())
                            and prior["actual_noise_sha256"] == record["actual_noise_sha256"]
                            and np.isfinite(prior["mae"]["r4_init"]),
                            "The real R4 baseline does not match this input and sampled noise")
                    record["mae"]["r4_init"] = prior["mae"]["r4_init"]
                    record["prediction_sha256"]["r4_init"] = prior["prediction_sha256"]["r4_init"]
                prefix = out / f"{domain}-{index:07d}"
                write(prefix.with_suffix(".ENTERED.json"), record)
                for role, state in active.items():
                    prediction = np.asarray(sample(state, frozen, obs, noise))
                    require(prediction.shape == (1, horizon, 32) and np.isfinite(prediction).all(), "Nonfinite native-horizon model output")
                    record["mae"][role] = float(np.abs(prediction[0, :, :dimensions] - target).mean())
                    record["prediction_sha256"][role] = hashlib.sha256(prediction.tobytes()).hexdigest()
                write(prefix.with_suffix(".RETURNED.json"), record)
                measured.append(record)
            output[domain] = dict(summary=summarize(measured, candidate), rows=measured)
        require([len(output[name]["rows"]) for name in ("robodojo", "single")] == [24, 18], "Incomplete development population")
        metrics = {name: r["summary"] for name, r in output.items()}
        proof = dict(plan=plan, source_step=step, checkpoint=model["checkpoint_sha256"], audit=audit,
                     roster=reference(roster), model_forwards=maximum_forwards,
                     physics_ticks=0, optimizer_updates=0, domains=output)
        if reused:
            require(all(r["r4_init_mae"] > 0 for r in metrics.values()), "Invalid R4 baseline MAE")
            change = {name: r[candidate + "_mae"] / r["r4_init_mae"] - 1 for name, r in metrics.items()}
            proof.update(reused_r4=reference(reused), relative_mae=change,
                         retention_alert_domains=[name for name, drift in change.items() if drift > .1])
            if recovery_baseline:
                proof["single_role"] = "R4_FIXED_INITIAL_VILLE_IO_SHARED_REPRESENTATION_ONLY"
        else:
            proof["source_learned"] = all(r["source_mae"] < r["r4_init_mae"] for r in metrics.values())
        write(out / "RESULT.json", proof)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("worker", "endpoint", "retention", "roster", "dual", "single", "single_video", "out"):
        parser.add_argument(key, type=Path)
    parser.add_argument("step", type=int)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--adapt-baseline", type=Path)
    selection.add_argument("--recovery-baseline", type=Path)
    evaluate(**vars(parser.parse_args()))
