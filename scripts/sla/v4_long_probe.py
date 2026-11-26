"""Measure the fixed V4 three-domain load on one admission-checked H200."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from m38_gpu_preflight_v1 import read, require, sha
from v3_main_place_learn import MAIN, endpoint_files
from v4_long_checkpoint import load_parent
from v4_long_engine import LongLearner
from v4_long_inputs import inputs
from v4_main250_hopper import allocation


SEED = 38002001
UPDATES = 6
THRESHOLDS = dict(relative_loss=1e-4, relative_gradient_l2=1e-3)
MEMORY_RESERVE_FRACTION = .10
LADDER = {"target": (64, 32, 16, 8), "robodojo": (32, 16, 8),
          "libero": (32, 16, 8), "auxiliary": (4, 2, 1)}


def lower(profile, component):
    """Reduce the first observed pressure point, then the largest remaining batch."""
    preferred = (component, "target", "robodojo", "libero", "auxiliary")
    for key in dict.fromkeys(preferred):
        if key not in LADDER:
            continue
        levels = LADDER[key]
        i = levels.index(profile[key])
        if i + 1 < len(levels):
            return dict(profile, **{key: levels[i + 1]})
    return None


def compare_gradients(ref, new):
    import jax
    import optax

    ref_total, ref_parts, ref_metrics = ref
    candidate_total, candidate_parts, metrics = new
    comparisons = {}
    for name, reference in dict(ref_parts, total=ref_total).items():
        actual = candidate_total if name == "total" else candidate_parts[name]
        norm = float(optax.global_norm(reference))
        error = float(optax.global_norm(jax.tree.map(lambda a, b: a - b, reference, actual)))
        relative_gradient = error / max(norm, 1e-12)
        relative_loss = None
        if name != "total":
            ref_loss, loss = ref_metrics[name]["loss"], metrics[name]["loss"]
            relative_loss = abs(loss - ref_loss) / max(abs(ref_loss), 1e-12)
        passed = (np.isfinite(relative_gradient) and relative_gradient <= THRESHOLDS["relative_gradient_l2"]
                  and (relative_loss is None or np.isfinite(relative_loss)
                       and relative_loss <= THRESHOLDS["relative_loss"]))
        comparisons[name] = dict(ref_gradient_l2=norm, absolute_gradient_error_l2=error,
                                 relative_gradient_l2=relative_gradient, relative_loss=relative_loss,
                                 historical_threshold_pass=bool(passed))
    return comparisons


def check_finite(probe):
    import optax

    total, components, metrics = probe
    return all(np.isfinite(float(optax.global_norm(g))) for g in (total, *components.values())) and all(
        np.isfinite(r["loss"]) for r in metrics.values())


def snapshot(learner):
    return dict(parameters={key: value.to_pure_dict() for key, value in learner.parameters.items()},
                optimizer=learner.optimizer, rng=learner.rng, step=learner.step)


def fixed_prediction(learner, graph):
    import jax
    from flax.nnx import State
    from openpi.models import pi0
    from go2arx5_vla.sla_adapter import batch_v2, full_h40_v6
    from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields

    batch = learner.prepare(UPDATES, "target")
    observation, _ = batch_v2.device_batch(jax.tree.map(lambda a: a[:1], batch.host))
    fields, _ = extracted_fields(batch_v2, pi0)
    infer = full_h40_v6.make_sampler(graph, fields, rematerialize=False, precision="highest")
    active = State.merge(learner.parameters["shared"], learner.parameters["target"])
    noise = jax.random.normal(jax.random.fold_in(learner.rng, 197), (1, 40, 32))
    value = np.asarray(infer(active, learner.frozen, observation, noise))
    require(value.shape == (1, 40, 32) and np.isfinite(value).all(), "Fixed target prediction failed")
    return value


def recover(learner, factory, path, graph, micro, guard):
    import jax
    import orbax.checkpoint as ocp

    before = time.monotonic()
    state = snapshot(learner)
    with ocp.PyTreeCheckpointer() as saver:
        saver.save(path.resolve(), state)
        saved = time.monotonic() - before
        restored = saver.restore(path.resolve(), item=state)
    seconds = time.monotonic() - before - saved
    require(all(np.array_equal(a, b) for a,b in zip(jax.tree.leaves(state),
                jax.tree.leaves(restored), strict=True)), "Optimizer/parameter/clock restore changed bytes")
    with closing(factory()) as cold:
        for domain, arrays in restored["parameters"].items():
            # Rebuild an independent NNX state before applying the archive.
            fresh = jax.tree.map(lambda x: x, cold.parameters[domain])
            fresh.replace_by_pure_dict(arrays)
            cold.parameters[domain] = fresh
        cold.optimizer, cold.rng, cold.step = restored["optimizer"], restored["rng"], int(restored["step"])
        next_ref = learner.components(UPDATES, learner.prepared(UPDATES, UPDATES + 1), *micro, guard)
        next_new = cold.components(UPDATES, cold.prepared(UPDATES, UPDATES + 1), *micro, guard)
        digest = compare_gradients(next_ref, next_new)
        a = fixed_prediction(learner, graph)
        b = fixed_prediction(cold, graph)
        require(all(np.isfinite(x["relative_gradient_l2"]) and
                    (x["relative_loss"] is None or np.isfinite(x["relative_loss"]))
                    for x in digest.values()), "Nonfinite fixed real gradient after cold restore")
    return dict(save_s=saved, restore_s=seconds, component_diagnostic=digest,
                prediction_absolute_max=float(np.abs(a-b).max()), prediction_equal=bool(np.array_equal(a,b)),
                prediction_sha256=hashlib.sha256(a.tobytes()).hexdigest(), files=len(endpoint_files(path)))


def validate_profile(capsule, custody, short_run, bank_root, profile):
    import jax
    import optax
    from scripts.pi0292.durable import write

    widths = {d: profile[d] for d in ("target", "robodojo", "libero")}
    micro = widths, profile["auxiliary"]
    name = profile_name(profile)
    result = capsule / "run"
    receipt = result / (name + ".json")
    site = read(result / "START.json")
    row = dict(profile=name, fm_micro=widths, aux_micro=micro[1], engineering_updates=0)
    write(receipt, row)
    require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
            and "H200" in jax.local_devices()[0].device_kind, "H200 runtime identity changed")
    jax.config.update("jax_compilation_cache_dir", str(capsule / "jit-cache"))
    charter = read(capsule / "incoming/DATA_BINDING.json")
    for input_name, path in (("source_target_bank", bank_root / "prepared/ACTUAL.json"),
                       ("auxiliary", bank_root / "native_aux-01/MANIFEST.json"),
                       ("short_terminal", short_run / "run/TRAIN_TERMINAL.json"),
                       ("short_data", short_run / "incoming/V3-MAIN-CONSUMED-DATA-02/MANIFEST.json")):
        require(sha(path) == charter[input_name], "Declared input changed: " + input_name)
    archive = custody / "HOPPER-ADAPT20K-V1/MAIN/checkpoints/phase03800"
    reference = short_run / "incoming/reference"
    model_start = time.monotonic()
    graphs, base, frozen, parent = load_parent(custody, archive, MAIN, reference)
    write(result / (name + ".parent.json"), dict(parent, load_s=time.monotonic()-model_start))
    datasets, tokenizer, banks, slots, auxiliary, cursors, rates = inputs(custody, short_run, bank_root, archive)
    rng = jax.random.PRNGKey(SEED)

    def factory():
        return LongLearner(graphs=graphs, parameters=dict(base), frozen=frozen, datasets=datasets,
                           tokenizer=tokenizer, banks=banks, auxiliary_slots=slots, auxiliary=auxiliary,
                           cursors=cursors, rates=rates, rng=rng)

    def guard():
        if (capsule / "STOP.json").exists() or time.time() >= site["deadline"] - 600:
            raise TimeoutError("Finite probe time or project STOP")

    with closing(factory()) as learner:
        try:
            probe = learner.components(0, learner.prepared(0, 1), *micro, guard)
            if not check_finite(probe):
                raise FloatingPointError("Nonfinite first-batch gradient")
            total, components, metrics = probe
            row["first_batch"] = {
                key: dict(loss=metrics.get(key, {}).get("loss"), gradient_l2=float(optax.global_norm(gradient)))
                for key, gradient in dict(components, total=total).items()
            }
            del total, components, probe
            write(receipt, row)
            timings = []
            with (result / (name + ".jsonl")).open("x", buffering=1) as out:
                for i in range(UPDATES):
                    update = learner.update(UPDATES, *micro, guard)
                    out.write(json.dumps(update, allow_nan=False) + "\n")
                    if i >= 2:
                        timings.append(update["seconds"])
                    row["engineering_updates"] = i + 1
                    write(receipt, row)
            row.update(p50_s=float(np.median(timings)), p95_s=float(np.percentile(timings, 95)))
            record_memory(learner, row)
            if row["headroom_fraction"] >= MEMORY_RESERVE_FRACTION:
                learner.active_component = "recovery"
                row["restore"] = recover(learner, factory, result / name, graphs["target"], micro, guard)
                record_memory(learner, row)
            if row["headroom_fraction"] < MEMORY_RESERVE_FRACTION:
                row.update(failure="INSUFFICIENT_HEADROOM", pressure_component=row["peak_component"])
        except FloatingPointError as exc:
            row.update(failure=str(exc), pressure_component=learner.active_component)
        except jax.errors.JaxRuntimeError as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc) and "out of memory" not in str(exc).lower():
                raise
            row.update(failure="GPU_MEMORY_EXHAUSTED", pressure_component=learner.active_component)
    write(receipt, row)


def record_memory(learner, row):
    import jax

    memory = jax.local_devices()[0].memory_stats()
    limit, peak = int(memory["bytes_limit"]), int(memory["peak_bytes_in_use"])
    if peak > row.get("gpu_memory", {}).get("peak_bytes_in_use", -1):
        row["peak_component"] = learner.peak_component
    row.update(gpu_memory=memory, gpu_headroom_at_process_peak_bytes=limit - peak,
               headroom_fraction=(limit - peak) / limit,
               host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def profile_name(profile):
    return "G{target}-R{robodojo}-L{libero}-A{auxiliary}".format(**profile)


def run(capsule, custody, short_run, bank_root, prior_probe=None):
    from scripts.pi0292.durable import OwnerLock, write

    with closing(OwnerLock(capsule / "OWNER.lock")):
        result = capsule / "run"
        result.mkdir(mode=0o700)
        site = allocation()
        write(result / "START.json", site)
        records = []
        profile = {domain: levels[0] for domain, levels in LADDER.items()}
        if prior_probe is not None:
            prior = read(capsule / "PREVIOUS_PROBE_READBACK.json")
            source = prior_probe / "run/PROFILES.json"
            require(sha(source) == prior["measured_sha256"]
                    and sha(prior_probe / "incoming/DATA_BINDING.json") == sha(capsule / "incoming/DATA_BINDING.json"),
                    "The preserved probe receipts must cover the same hardware and data")
            for row in read(source):
                require(row["profile"] == profile_name(profile)
                        and row == read(prior_probe / "run" / (row["profile"] + ".json")),
                        "Prior candidate receipt changed")
                records.append(dict(row, measured_job=prior["job"]))
                profile = lower(profile, row.get("pressure_component", "target"))
        stable = sum(bool(r.get("restore") and not r.get("failure")) for r in records)
        while profile is not None and stable < 2:
            name = profile_name(profile)
            print("H200 candidate", name, flush=True)
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), str(capsule),
                            str(custody), str(short_run), str(bank_root), "--profile",
                            json.dumps(profile)], check=True)
            row = read(result / (name + ".json"))
            records.append(dict(row, measured_job=site["job"]))
            write(result / "PROFILES.json", records)
            stable += int(bool(row.get("restore") and not row.get("failure")))
            profile = lower(profile, row.get("pressure_component", "target"))
        write(result / "PROFILES.json", records)
        eligible = [r for r in records if r.get("restore") and not r.get("failure")
                    and r["engineering_updates"] == UPDATES]
        require(eligible, "No finite, recoverable parallel profile with sufficient GPU headroom")
        fastest = min(eligible, key=lambda r: (r["p50_s"], r["p95_s"]))
        write(result / "ACCEPTED.json", dict(stage="V4_LONG_MAIN38_PERFORMANCE_ONLY",
              selected=dict(fm=fastest["fm_micro"], auxiliary=fastest["aux_micro"]), workers=4, prefetched_batches=2,
              formal_optimizer_updates=0, profile_sha256=sha(result / "PROFILES.json"),
              engineering_updates=sum(r["engineering_updates"] for r in records),
              selection="FASTEST_STABLE_WITH_NATIVE_BATCH_AND_RECOVERY",
              thresholds="DIAGNOSTIC_ONLY",
              cache="DISABLED_PENDING_INDEPENDENT_FROZEN_DEPENDENCY_PROOF"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capsule", "custody", "short_run", "bank_root"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--profile", type=json.loads)
    parser.add_argument("--prior-probe", type=Path)
    args = vars(parser.parse_args())
    profile = args.pop("profile")
    if profile is None:
        run(**args)
    else:
        args.pop("prior_probe")
        validate_profile(**args, profile=profile)
