"""Run the fixed six-window offline checkpoint diagnostic against original H40/G7."""

import argparse
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

from m38_gpu_preflight_v1 import bind_imports, check_gpu, pin, read, require, sha


ROSTER = (
    ("V3-MAIN-PLACE250-EVAL-03/M_DEV/M_DEV_P01_301", 0, "PLACE_ENTRY"),
    ("V3-MAIN-PLACE250-EVAL-03/M_DEV/M_DEV_P02_301", 0, "PLACE_ENTRY"),
    ("V3-MAIN-PLACE250-EVAL-03/H_REG/H_REG_N02_A", 0, "HOLD_ENTRY"),
    ("V3-MAIN-PLACE250-EVAL-03/H_REG/H_REG_N02_B", 0, "HOLD_ENTRY"),
    ("V2-CONFIRM-01/CONFIRM_N02_201", 56, "HOLD_POST_GRASP"),
    ("V2-CONFIRM-01/CONFIRM_N02_202", 58, "HOLD_POST_GRASP"),
)
STEPS = (0, 100, 250, 500, 1000)
PARENT = "502d890d3ff314ac42f4c79c360942b9b626bf1fcc4b6fc488fbab15ba508361"
WALL = 900


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def pinned(cell):
    return pin(cell["path"], cell["sha256"])


def fixed_windows(root):
    from go2arx5_vla.sla_adapter.dense_g7_v6 import previous_command_context

    rows = []
    for path, index, phase in ROSTER:
        folder = root / path
        archive = folder / "native-01/g7-queries"
        query = archive / f"{index:04d}"
        lease = read(folder / "control/LEASE.json")
        result = folder / "RESULT.private.json"
        request = query / "MODEL_REQUEST.private.json"
        returned = query / "MODEL_RETURN.private.json"
        meta = read(query / "INPUT_META.private.json")
        request_data = read(request)
        response = read(returned)["response"]
        require(lease["status"] == "CLOSED" and not read(result)["technical_failure"]
                and request_data["input"] == meta["input"]
                and request_data["previous_reference"] == meta["previous_reference"]
                and not meta["teacher_or_object_truth_in_model_input"], "Use only closed, genuine student inputs")
        previous_command_context(request_data["previous_reference"])
        grasp = None
        if phase == "HOLD_POST_GRASP":
            grasp = read(result)["detail"]["score"]["events"]["target_grasp"]["native_step"]
            ordered = sorted(archive.glob("*/MODEL_REQUEST.private.json"))
            first = next(f for f in ordered if read(f)["native_step"] > grasp)
            require(first == request, "Select the first fixed query after actual grasp")
        with np.load(pinned(meta["input"]), allow_pickle=False) as packet:
            for key, shape in (("state_eef10", (10,)), ("policy_from_world", (4, 4)),
                               ("accepted_reference_TCP", (4, 4)), ("T_policy_from_mount", (4, 4))):
                require(packet[key].shape == shape and np.isfinite(packet[key]).all(), "Malformed frozen observation")
            require(all(packet[key].ndim == 3 and packet[key].shape[-1] == 3 and packet[key].dtype == np.uint8
                        for key in ("body_rgb", "wrist_rgb")), "Missing original image bytes")
        pinned(response["action"])
        noise = response["comparison_noise_sha256"]
        require(len(noise) == 64, "Expected an archived real CUDA sampler hash")
        rows.append(dict(id=folder.name, phase=phase, query=index, tick=request_data["native_step"],
                         grasp_tick=grasp, input=meta["input"], request=ref(request),
                         terminal=ref(result), returned=ref(returned), noise_sha256=noise,
                         noise_id=request_data["noise_id"]))
    return rows


def candidate(folder, original, step, audit):
    manifest = read(folder / "CHECKPOINT.json")
    require(original["checkpoint_sha256"] == PARENT, "Expected MAIN3.8K as the direct parent")
    if step == 0:
        require(manifest == original and audit is None, "Wrong parent or spurious milestone")
    else:
        report = read(audit)
        formal = step == 1000
        require(report["step"] == manifest["phase"] == step
                and report["provider_manifest_sha256"] == sha(folder / "CHECKPOINT.json")
                and report["receipt_sha256"] == manifest["training_receipt_sha256"]
                and report["contract_sha256"] == manifest["training_contract_sha256"]
                and report["parent_sha256"] == manifest["previous_checkpoint_sha256"] == PARENT
                and manifest["schema"] == "SLA_V4_MAIN_MULTISOURCE_INFERENCE"
                and manifest["formal_inference_eligible"] == report["formal"] == formal
                and manifest["role"] == ("MAIN_MULTISOURCE_1000_V4" if formal else "DIAGNOSTIC")
                and manifest["source_codec"] == "DENSE_REFERENCE_G7_V6" and manifest["frozen_wbc"]
                and manifest["original_l1"], "Unverified or wrong fixed milestone")
    require(all(manifest[key] == original[key] for key in (
        "foundation_sha256", "frozen_model", "action_sha256", "state_sha256",
        "tokenizer_sha256", "partition_sha256")), "Changed downstream ABI")
    for old, new in zip(original["arrays"], manifest["arrays"], strict=True):
        require(all(old[key] == new[key] for key in ("file", "key", "shape", "dtype")),
                "Provider tensor schema changed")
        pin(folder / new["file"], new["sha256"])
    return manifest


def score(raw, packet, previous):
    from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
    from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
    from scripts.pi0292.eval_contract import matrix10

    frame = packet["policy_from_world"]
    measured = np.linalg.inv(frame) @ matrix10(packet["state_eef10"])
    poses = np.linalg.inv(frame) @ matrix10(raw)
    xyz = (poses[:, :3, 3] - measured[:3, 3]) * 1e3
    width = 2. + 40. * raw[:, 9]
    return dict(original_envelope=analyze(raw, packet["state_eef10"], frame, packet["accepted_reference_TCP"]),
                jaw=label_margin_report(raw, previous), p10_displacement_world_mm=xyz[9].tolist(),
                h40_displacement_world_mm=xyz[-1].tolist(), p10_highest_z_mm=float(xyz[:10, 2].max()),
                h40_highest_z_mm=float(xyz[:, 2].max()), jaw_width_mm_at_1_10_40=width[[0, 9, 39]].tolist(),
                p10_tightest_width_mm=float(width[:10].min()), h40_tightest_width_mm=float(width.min()))


def watch(root, worker, roster, package, step, audit, out, *, execute, retained_roster_sha=None, reuse=None):
    from scripts.pi0292.durable import OwnerLock, write

    spec = read(worker)
    bind_imports(spec)
    from scripts.sla.core_hopper_native_scope import accepted

    _, gate, receiver = accepted(spec)
    template = read(pin(Path(gate["package"]["path"]) / "CHECKPOINT.json", gate["package"]["sha256"]))
    if retained_roster_sha is None:
        rows = fixed_windows(root)
    else:
        retained = read(pin(roster, retained_roster_sha))
        rows = retained["rows"]
        require(24 <= len(rows) <= 48 and retained["original_worker"] == ref(worker)
                and len({row["id"] for row in rows}) == len(rows), "Invalid retained diagnosis roster")
    if not roster.exists():
        require(not execute, "Seal the complete fixed input roster before any model forward")
        write(roster, dict(rows=rows, original_worker=ref(worker), sampler_method="EXACT_ARCHIVED_CUDA_HASH",
                           original_h40_envelope=True, native_integrations=0, model_forwards=0))
    else:
        binding = read(roster)
        require(binding["rows"] == rows and binding["original_worker"] == ref(worker), "Offline roster changed")
    manifest = candidate(package, template, step, audit)
    returned_before = {}
    if reuse is not None:
        before = read(reuse)
        require(before["checkpoint"] == manifest["checkpoint_sha256"] and before["step"] == step
                and before["physics_ticks"] == before["training_updates"] == 0, "Archived offline checkpoint differs")
        returned_before = {row["id"]: row for row in before["rows"]}
    if not execute:
        known_passes = {}
        for row in rows:
            request = read(pinned(row["request"]))
            response = read(pinned(row["returned"]))["response"]
            with np.load(pinned(row["input"]), allow_pickle=False) as packet, np.load(
                    pinned(response["action"]), allow_pickle=False) as raw:
                known_passes[row["id"]] = score(raw["physical_H40"], packet, request["previous_reference"])[
                    "original_envelope"]["original_full_H40_pass"]
        write(out, dict(step=step, roster=ref(roster), provider=ref(package / "CHECKPOINT.json"),
                        checkpoint=manifest["checkpoint_sha256"], fixed_windows=len(rows),
                        archived_envelope_pass=known_passes, forwards=0))
        return

    import jax
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from scripts.sla.hopper_reference import provider_from_manifest
    from scripts.sla.hopper_site import pinned as site_pinned

    with closing(OwnerLock(Path(spec["neural_owner_lock"]))):
        require(sys.executable == spec["python"] and os.uname().nodename == spec["host"]
                and os.getuid() == spec["uid"] and all(
                    os.environ.get(key) == spec["gpu_environment"][key] for key in
                    ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION")),
                "Unexpected Core host, interpreter, GPU or numerical flags")
        gpu = check_gpu(spec["GPU_uuid"], spec["allocator_limit_bytes"], spec["minimum_host_available_bytes"])
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1, "One Core CUDA device is required")
        out.mkdir(mode=0o700)
        deadline = time.monotonic() + WALL
        jax.config.update("jax_compilation_cache_dir", str(out / "jit-cache"))
        write(out / "START.json", dict(step=step, gpu=gpu, max_forwards=len(rows), roster=ref(roster),
                                      provider=ref(package / "CHECKPOINT.json")))
        acquisition = site_pinned(receiver["input"])["target_D1"]
        dataset = AdaptTargetDataset(acquisition["legacy"], acquisition["captures"], resize_with_pad=resize_with_pad)
        dataset.bind_state(acquisition["statistics"])
        model = provider_from_manifest(package, manifest, receiver["model"], dataset, receiver["tokenizer"]["path"])
        raw_infer = model.infer
        sampler = []
        active_row = None

        def verify(*args):
            realized = hashlib.sha256(np.asarray(args[-1]).tobytes()).hexdigest()
            require(active_row is not None and realized == active_row["noise_sha256"],
                    "Actual CUDA sampler differs from the archived input")
            sampler.append(realized)
            return raw_infer(*args)

        model.infer = verify
        results = []
        reused = 0
        for row in rows:
            require(time.monotonic() < deadline and not (out / "STOP.json").exists(), "Offline GPU lease ended")
            prefix = out / row["id"]
            write(prefix.with_suffix(".ENTERED.json"), row)
            previous = returned_before.get(row["id"])
            if previous is not None:
                require(all(previous[key] == value for key, value in row.items())
                        and previous["actual_sampler_sha256"] == row["noise_sha256"],
                        "An archived model row differs from the retained input")
                pinned(previous["action"])
                record = dict(previous, reused_fixed_forward=True)
                write(prefix.with_suffix(".RETURNED.json"), record)
                results.append(record)
                reused += 1
                continue
            active_row = row
            request = read(pinned(row["request"]))
            with np.load(pinned(row["input"]), allow_pickle=False) as packet:
                result = model.predict(packet["state_eef10"], [packet["body_rgb"], packet["wrist_rgb"]],
                                       request["language"], packet["T_policy_from_mount"], noise_id=row["noise_id"],
                                       previous_reference=request["previous_reference"])
                raw = result["physical_H40"]
                metrics = score(raw, packet, request["previous_reference"])
            old_action = read(pinned(row["returned"]))["response"]["action"]
            with np.load(pinned(old_action), allow_pickle=False) as old:
                metrics["archived_raw_h40_exact"] = bool(np.array_equal(old["physical_H40"], raw))
            output = prefix.with_suffix(".npz")
            with output.open("xb") as stream:
                np.savez(stream, normalized=result["normalized"], physical_H40=raw)
            record = dict(row, action=ref(output), actual_sampler_sha256=sampler[-1], metrics=metrics)
            write(prefix.with_suffix(".RETURNED.json"), record)
            results.append(record)
        require(len(sampler) + reused == len(results) == len(rows), "Missing actual fixed sampler or return")
        write(out / "RESULT.json", dict(step=step, roster=ref(roster), checkpoint=manifest["checkpoint_sha256"],
              model_forwards=len(sampler), reused_checked_forwards=reused,
              physics_ticks=0, training_updates=0, closed_loop=False, rows=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "worker", "roster", "package", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--step", required=True, type=int, choices=STEPS)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--retained-roster-sha")
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--execute", action="store_true")
    main = vars(parser.parse_args())
    require((main["step"] == 0) == (main["audit"] is None), "Specify the checkpoint audit for each fixed milestone")
    watch(**main)
