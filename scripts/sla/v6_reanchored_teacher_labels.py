"""Represent verified consumed geometry in the input coordinates of the existing receiver."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from v3_main_place_learn import input_contract
from v6_reference_anchors import metric, original_cases, receiver, transport


SCHEMA = "V6_ACTUAL_CONSUMED_INVERSE_WORLD_TRANSPORT_G7"


def bind_dataset(target, package, manifest_sha, original_manifest):
    manifest = read(pin(package / "MANIFEST.json", manifest_sha))
    require(manifest["schema"] == SCHEMA and manifest["source_manifest_sha256"] == sha(original_manifest)
            and manifest["original_runtime_transport"] and manifest["gripper_preserved"]
            and manifest["mask_preserved"] and manifest["windows"] == len(target) == 4450,
            "The label version must match the unchanged target sampler")
    originals = {row["episode"]: row for row in read(original_manifest)["episodes"]}
    require(set(originals) == {row["episode"] for row in manifest["episodes"]} == set(target.actual),
            "The label version changed the underlying training population")
    patched = {}
    for row in manifest["episodes"]:
        identity = row["episode"]
        old, original_mask = target.actual[identity]
        require(row["source_sha256"] == originals[identity]["sha256"],
                "Source commands no longer match the independently qualified dataset")
        with np.load(pin(package / row["labels"], row["sha256"]), allow_pickle=False) as stream:
            new, mask = stream["action"], stream["valid"]
            require(new.shape == old.shape and new.dtype == old.dtype and np.isfinite(new).all()
                    and np.array_equal(new[..., 6], old[..., 6]) and np.array_equal(mask, original_mask)
                    and np.array_equal(new[~mask], old[~mask]),
                    "The corrected output must preserve grip, padding, and actual support")
            patched[identity] = new, mask
    target.actual = patched


def reexpress(spec, queries, recorded, *, record_residual=None):
    from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec

    policy = np.asarray(spec["policy_from_world"])
    world = np.linalg.inv(policy)
    codec = GripLogitCodec(np.asarray(queries[0]["episode_fixed_T_policy_from_mount"]))
    fixed = recorded["action"].copy()
    require(len(queries) == len(fixed), "Mismatched Teacher queries and command rows")
    maxima, p10_maxima = np.zeros(2), np.zeros(2)
    for index, query in enumerate(queries):
        mask = recorded["valid"][index]
        anchor = np.asarray(query["state_eef10"])
        actual = np.asarray(query["actual_state"]["tcp"])
        previous = receiver(query["recent_reference"]["accepted_pose7"])
        tick = query["native_step"]
        require(tick == recorded["native_step"][index] == query["recent_reference"]["native_step"]
                and query["phase"] == recorded["phase"][index] and mask[:10].all()
                and np.allclose(matrix_from10(anchor[None])[0], policy @ actual, atol=1e-8, rtol=0.),
                "Labels and original causal observations must use the same query")
        baseline = recorded["action"][index]
        decoded = codec.decode(anchor, np.pad(baseline, ((0, 0), (0, 25))))
        original_world = world @ matrix_from10(decoded)
        inverse = transport(original_world, previous, actual)
        preimage = pose10(policy @ inverse, decoded[:, 9])
        _, encoded = codec.encode(anchor, preimage)
        fixed[index, mask, :6] = encoded[mask, :6]
        view = codec.decode(anchor, np.pad(fixed[index], ((0, 0), (0, 25))))
        replayed = transport(world @ matrix_from10(view), actual, previous)
        residual = metric(replayed[mask], original_world[mask])
        if record_residual is not None:
            record_residual(index, residual, mask)
        maxima = np.maximum(maxima, residual.max(axis=0))
        p10_maxima = np.maximum(p10_maxima, residual[:10].max(axis=0))
    require(np.array_equal(fixed[..., 6], recorded["action"][..., 6])
            and np.array_equal(fixed[~recorded["valid"]], recorded["action"][~recorded["valid"]]),
            "Gripper labels and unsupported suffix must stay bitwise unchanged")
    return fixed, maxima, p10_maxima


def export(native_root, consumed, parent_gate, diagnosis, output):
    from go2arx5_vla.sla_adapter.codec_v1 import Quantiles

    diagnostic = read(diagnosis)
    require(diagnostic["queries"] == 4450 and all(diagnostic[key] == 0 for key in
            ("native_integrations", "model_forwards", "optimizer_updates")), "Missing the actual anchor audit")
    prior = {r["episode"]: r["query_sha256"] for r in diagnostic["cases"]}
    gate = read(parent_gate)["gate"]
    _, _, stats = input_contract(gate, Path(gate["package"]["path"]))
    q = stats["actions"]
    norm = Quantiles(np.asarray(q["q01"]), np.asarray(q["q99"]))
    support = defaultdict(list)
    maxima = np.zeros(2)
    p10_maxima = np.zeros(2)
    checked = total = full = 0
    ledger = []
    output.mkdir(mode=0o700)
    for entry, spec, queries, query_sha in original_cases(native_root, consumed):
        eid = entry["episode"]
        require(query_sha == prior[eid], "Original Teacher query bytes changed after the audit")
        with np.load(pin(consumed / entry["labels"], entry["sha256"]), allow_pickle=False) as old:
            recorded = {k: old[k] for k in ("action", "valid", "phase", "native_step")}
        fixed, error, p10_error = reexpress(spec, queries, recorded)
        maxima = np.maximum(maxima, error)
        p10_maxima = np.maximum(p10_maxima, p10_error)
        checked += int(recorded["valid"].sum())
        full += int(recorded["valid"].all(axis=1).sum())
        for index, query in enumerate(queries):
            mask = recorded["valid"][index]
            baseline = recorded["action"][index]
            phase = query["phase"]
            for scope, rows in (("P10", slice(0, 10)), ("SUPPORTED_SUFFIX", slice(10, None))):
                support[phase, scope, "original"].append(np.abs(norm.normalize(baseline[mask][rows])))
                support[phase, scope, "reexpressed"].append(np.abs(norm.normalize(fixed[index, mask][rows])))
        result = output / entry["labels"]
        with result.open("xb") as stream:
            np.savez_compressed(stream, **dict(recorded, action=fixed))
        ledger.append(dict(entry, bytes=result.stat().st_size, source_sha256=entry["sha256"], sha256=sha(result)))
        total += len(queries)
    require(total == 4450 and checked > 4450*10 and maxima[0] < .001 and maxima[1] < .001,
            "Existing physical transport must reconstruct the same executed geometry on real support")
    summary = []
    for (phase, scope, label), values in sorted(support.items()):
        entries = np.concatenate(values)
        if len(entries):
            summary.append(dict(phase=phase, scope=scope, label=label, atoms=len(entries),
                                outside=np.mean(entries > 1.000001, axis=0).tolist(),
                                absolute_p95=np.percentile(entries, 95, axis=0).tolist()))
    manifest = output / "MANIFEST.json"
    with manifest.open("x") as stream:
        json.dump(dict(schema=SCHEMA, source_manifest_sha256=sha(consumed / "MANIFEST.json"),
                       diagnosis_sha256=sha(diagnosis), episodes=ledger,
                       windows=total, full_H40=full, supported_atoms=checked,
                       original_runtime_transport=True, gripper_preserved=True, mask_preserved=True),
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
    proof = dict(manifest_sha256=sha(manifest), windows=total, full_H40=full, physical_atoms=checked,
                 max_reconstructed_mm_deg=maxima.tolist(), P10_max_reconstructed_mm_deg=p10_maxima.tolist(),
                 quantile_support=summary, model_forwards=0, optimizer_updates=0, native_integrations=0)
    with (output / "ACTUAL.json").open("x") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native_root", "consumed", "parent_gate", "diagnosis", "output"):
        parser.add_argument(name, type=Path)
    export(**vars(parser.parse_args()))
