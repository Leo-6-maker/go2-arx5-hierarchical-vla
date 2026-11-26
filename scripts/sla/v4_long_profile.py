"""Trace a bounded replay of the original H200 V4 learner after its formal 1K exit."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing, contextmanager
import gzip
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_place_learn import endpoint_files
from v4_long_checkpoint import load_parent
from v4_long_engine import LongLearner
from v4_long_export import COUNTS, ledger_prefix
from v4_long_inputs import inputs
from v4_long_probe import SEED, snapshot
from v4_main250_hopper import allocation


START = 250
WARM = 3
TRACED = 24
PHASES = ("prefetch_wait/", "prepare_gpu/", "flow_matching/", "h40_auxiliary",
          "join_gradient", "finite_gradient_readback", "adamw_global_clip")


def same_step(a, b):
    return (a["step"] == b["step"] and a["lr"] == b["lr"]
            and {d: v["visits"] for d, v in a["components"].items()} == COUNTS
            and all(a["components"][d]["visit_sha256"] == b["components"][d]["visit_sha256"]
                    for d in ("target", "robodojo", "libero"))
            and a["components"]["auxiliary"]["slots"] == b["components"]["auxiliary"]["slots"])


def union_seconds(intervals):
    total = 0.
    end = float("-inf")
    for start, stop in sorted(intervals):
        total += max(0., stop - max(start, end))
        end = max(end, stop)
    return total / 1e6


def inspect_trace(path):
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    gpu_pids = {e["pid"] for e in events if e.get("name") == "process_name"
                and "gpu" in str(e.get("args", {}).get("name", "")).lower()}
    gpu_streams = {(e["pid"], e["tid"]) for e in events
                   if e.get("name") == "thread_name" and e["pid"] in gpu_pids
                   and str(e.get("args", {}).get("name", "")).lower().startswith("stream")}
    kernel_time = Counter()
    tagged = Counter()
    kernels, copies = [], []
    phases = defaultdict(list)
    steps = []
    for event in events:
        if event.get("ph") != "X" or event.get("dur", 0) <= 0:
            continue
        name = event["name"]
        args = event.get("args", {})
        interval = event["ts"], event["ts"] + event["dur"]
        hardware_stream = (event["pid"], event.get("tid")) in gpu_streams
        copy = "memcpy_details" in args or hardware_stream and name.lower().startswith("memcpy")
        kernel = "kernel_details" in args or hardware_stream and "correlation_id" in args
        if copy:
            copies.append(interval)
        elif kernel and not name.lower().startswith(("memset", "memalloc", "memfree")):
            kernels.append(interval)
            kernel_time[name] += event["dur"]
            label = " ".join(str(args.get(key, "")) for key in ("hlo_op", "tf_op", "deduplicated_name"))
            if "remat" in (name + " " + label).lower():
                tagged["explicit_remat_label"] += event["dur"]
        elif event["pid"] not in gpu_pids:
            if name == "v4_train":
                steps.append(interval)
            elif name.startswith(PHASES):
                phases[name].append(interval)

    def overlap(intervals, start, end):
        return union_seconds((max(lo, start), min(hi, end)) for lo, hi in intervals
                             if lo < end and hi > start)

    return dict(step_markers=len(steps), kernel_events=len(kernels), memcpy_events=len(copies),
                gpu_kernel_union_seconds=union_seconds(kernels),
                gpu_memcpy_union_seconds=union_seconds(copies),
                total_step_marker_seconds=union_seconds(steps),
                kernel_busy_median_s=statistics.median(overlap(kernels, a, b) for a, b in steps) if steps else None,
                memcpy_busy_median_s=statistics.median(overlap(copies, a, b) for a, b in steps) if steps else None,
                timestamp_overlap_by_host_phase={phase: dict(markers=len(spans),
                    host_seconds=union_seconds(spans),
                    median_host_s=statistics.median((b-a)/1e6 for a,b in spans),
                    median_gpu_kernel_overlap_s=statistics.median(overlap(kernels,a,b) for a,b in spans),
                    median_gpu_memcpy_overlap_s=statistics.median(overlap(copies,a,b) for a,b in spans))
                    for phase, spans in phases.items()},
                top_raw_cuda_kernel_seconds=[(name, total / 1e6)
                                             for name, total in kernel_time.most_common(20)],
                tagged_kernel_duration_s={key: value / 1e6 for key, value in tagged.items()})


@contextmanager
def sample_device(path):
    """Record low-rate device utilization independently of CUDA activity tracing."""
    with path.open("xb") as output:
        sampler = subprocess.Popen(
            ["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
             "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
             "--format=csv", "-l", "1"], stdout=output, stderr=subprocess.STDOUT)
        try:
            yield
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()


def run(capsule, custody, short_run, bank_root, main_run, milestone):
    import jax
    import orbax.checkpoint as ocp
    from scripts.pi0292.durable import OwnerLock, write

    with closing(OwnerLock(capsule / "OWNER.lock")):
        charter = read(capsule / "incoming/PROFILE_BINDING.json")
        source = main_run / "run"
        terminal = read(pin(source / "TRAIN_TERMINAL.json", charter["main_terminal_sha256"]))
        require(terminal["formal_endpoint"] and terminal["completed_updates"] == 1000,
                "The original main arm must first close at its natural endpoint")
        contract = read(pin(source / "RUN_CONTRACT.json", terminal["contract_sha256"]))
        fixed = read(pin(milestone, charter["milestone_sha256"]))
        endpoint = Path(fixed["endpoint"]).resolve(strict=True)
        require(fixed["step"] == START and endpoint == source.resolve() / "STEP0250"
                and fixed["contract_sha256"] == terminal["contract_sha256"]
                and endpoint_files(endpoint) == fixed["files"], "Replay must use the exact captured milestone")
        basis = contract["charter"]
        for key, artifact in (("target_bank_sha256", bank_root/"prepared/ACTUAL.json"),
                              ("auxiliary_sha256", bank_root/"native_aux-01/MANIFEST.json"),
                              ("short_terminal_sha256", short_run/"run/TRAIN_TERMINAL.json")):
            pin(artifact, basis[key])
        reference = ledger_prefix(source, START + WARM + TRACED)[START:]
        result = capsule / "run"
        result.mkdir(mode=0o700)
        site = allocation()
        write(result / "START.json", site)
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
                and "H200" in jax.local_devices()[0].device_kind, "Profiling requires the same one-H200 backend")
        jax.config.update("jax_compilation_cache_dir", str(capsule / "jit-cache"))
        parent = Path(basis["parent_path"])
        v3 = short_run / "run" if basis.get("parent_v3_overlay") else None
        graphs, parameters, frozen, _ = load_parent(custody, parent, basis["parent_sha256"],
                                                   short_run / "incoming/reference", v3_endpoint=v3)
        data, tokenizer, banks, slots, auxiliary, cursors, rates = inputs(custody, short_run, bank_root, parent)
        rng = jax.random.PRNGKey(SEED)
        settings = contract["settings"]
        require(settings["workers"] == 4 and settings["prefetch_batches"] == 2,
                "The recorded input pipeline differs from the replay")
        stop = terminal["completed_updates"]

        def guard():
            if (capsule / "STOP.json").exists() or time.time() > site["deadline"] - 300:
                raise TimeoutError("Profiling reached its finite endpoint")

        with closing(LongLearner(graphs=graphs, parameters=parameters, frozen=frozen, datasets=data,
                                tokenizer=tokenizer, banks=banks, auxiliary_slots=slots, auxiliary=auxiliary,
                                cursors=cursors, rates=rates, rng=rng, trace=jax.profiler.TraceAnnotation)) as learner:
            with ocp.PyTreeCheckpointer() as reader:
                state = reader.restore(endpoint, item=snapshot(learner))
            require(int(state["step"]) == START and np.array_equal(state["rng"], rng), "Wrong restored step or RNG")
            for key, value in state["parameters"].items():
                learner.parameters[key].replace_by_pure_dict(value)
            learner.optimizer, learner.rng, learner.step = state["optimizer"], state["rng"], START
            del state
            rows = []
            with (result / "UPDATES.jsonl").open("x", buffering=1) as out:
                def step():
                    row = learner.update(stop, settings["fm_micro"], settings["aux_micro"], guard)
                    row["profiled"] = learner.step > START + WARM
                    out.write(json.dumps(row, allow_nan=False) + "\n")
                    require(same_step(row, reference[len(rows)]), "Replay changed actual sample visits, LR, or aux slots")
                    rows.append(row)

                for _ in range(WARM):
                    step()
                with sample_device(result / "DEVICE_SAMPLES.csv"), jax.profiler.trace(result / "xprof"):
                    for _ in range(TRACED):
                        with jax.profiler.StepTraceAnnotation("v4_train", step_num=learner.step + 1):
                            step()
        files = endpoint_files(result / "xprof")
        require(len(rows) == WARM + TRACED and any(k.endswith(".xplane.pb") for k in files),
                "Expected the complete profiler XPlane artifact")
        chrome, = (result / "xprof" / k for k in files if k.endswith(".trace.json.gz"))
        detail = inspect_trace(chrome)
        write(result / "DEVICE_TRACE.json", detail)
        require(detail["kernel_events"] > 0 and detail["step_markers"] == TRACED,
                "A host-only or incomplete trace cannot establish H200 kernel activity")
        report = dict(status="ORIGINAL_H200_PROFILE_CAPTURED", replay_start=START, warmup=WARM, traced=TRACED,
                      file_hashes=files, engineering_updates=len(rows), optimizer_updates_formal=0,
                      trace_wall_p50_s=statistics.median(r["seconds"] for r in rows[WARM:]),
                      original_matched_wall_p50_s=statistics.median(r["seconds"] for r in reference[WARM:]),
                      trace_bytes=sum((result / "xprof" / path).stat().st_size for path in files),
                      allocator=jax.local_devices()[0].memory_stats(), matched_visits=True,
                      device_trace_sha256=sha(result / "DEVICE_TRACE.json"),
                      device_samples_sha256=sha(result / "DEVICE_SAMPLES.csv"),
                      source_main_terminal_sha256=charter["main_terminal_sha256"], physics_ticks=0)
        write(result / "RESULT.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("capsule", "custody", "short_run", "bank_root", "main_run", "milestone"):
        p.add_argument(arg, type=Path)
    run(**vars(p.parse_args()))
