"""Certify corrected target labels against actual receiver anchors and consumed commands."""

import argparse
from collections import defaultdict
import inspect
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v6_reanchored_teacher_labels import reexpress
from v6_reference_anchors import case_evidence


LIMITS = np.asarray([1e-4, 1e-5])


def certify(members, statistics, excluded_scenes):
    from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
    from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal

    runtime = compile_proposal.__globals__["task"].FiftyMsProgram
    source = inspect.getsource(runtime)
    require("reference[:3,3]+raw[:,:3]-actual[:3,3]" in source
            and "reference[:3,:3]@actual[:3,:3].T@rotations" in source,
            "Frozen query-to-receiver transform changed")
    q = statistics["actions"]
    norm = Quantiles(np.asarray(q["q01"]), np.asarray(q["q99"]))
    ledger, residuals, support = [], defaultdict(list), defaultdict(list)
    for native_root, row, raw, checked in members:
        require(row["scene"] not in excluded_scenes, "Evaluation scene entered TRAIN")
        spec, queries, query_sha = case_evidence(native_root, row)
        with np.load(raw, allow_pickle=False) as source_data:
            recorded = {k: source_data[k] for k in ("action", "native_step", "phase", "valid")}

        def collect(index, residual, mask):
            phase = str(recorded["phase"][index])
            residuals[phase, "P10"].append(residual[:10])
            if mask[10:].any():
                residuals[phase, "H40_SUFFIX"].append(residual[10:])

        actions, maxima, p10_maxima = reexpress(spec, queries, recorded, record_residual=collect)
        accepted = recorded["valid"]
        require(np.all(maxima <= LIMITS), "Corrected label exceeds maximum transport tolerance")
        with np.load(checked, allow_pickle=False) as existing:
            require(np.array_equal(actions, existing["action"])
                    and all(np.array_equal(recorded[k], existing[k]) for k in ("valid", "phase", "native_step")),
                    "Corrected label differs from the sealed bytes")
        for index, phase in enumerate(recorded["phase"]):
            scaled = norm.normalize(actions[index][accepted[index]])
            require(np.isfinite(scaled).all(), "A corrected target is nonfinite under the original statistics")
            support[str(phase), "P10"].append(scaled[:10])
            if len(scaled) > 10:
                support[str(phase), "H40_SUFFIX"].append(scaled[10:])
        ledger.append(dict(episode=row["episode"], scene_sha256=row["scene"], raw_sha256=sha(raw),
                           corrected_sha256=sha(checked), receiver_queries_sha256=query_sha,
                           windows=len(queries), full_H40=int(accepted.all(axis=1).sum()),
                           supported_points=int(accepted.sum()), max_error_mm_deg=maxima.tolist(),
                           p10_max_error_mm_deg=p10_maxima.tolist()))
    phase_report = []
    for (phase, part), values in sorted(residuals.items()):
        residual = np.concatenate(values)
        normalized = np.abs(np.concatenate(support[phase, part]))
        phase_report.append(dict(phase=phase, part=part, points=len(residual),
                                 max_error_mm_deg=residual.max(axis=0).tolist(),
                                 p95_error_mm_deg=np.percentile(residual, 95, axis=0).tolist(),
                                 normalized_abs_max=normalized.max(axis=0).tolist(),
                                 outside_q01_q99=np.mean(normalized > 1.000001, axis=0).tolist()))
    return dict(episodes=ledger, stage_metrics=phase_report, limits_mm_deg=LIMITS.tolist(),
                frozen_runtime_sha256=sha(Path(inspect.getsourcefile(runtime))),
                native_integrations=0, model_forwards=0, optimizer_updates=0)


def legacy(native_root, factory, retention, parent_gate, output):
    from scripts.pi0292.durable import write
    from v3_main_place_learn import input_contract

    fixed = factory / "SNAPSHOTS/TRAIN-LEGACY-064"
    received = read(fixed / "MANIFEST.json")
    inherited = retention / "REEXPRESSED-LEGACY-DELTA-68-01"
    increment = retention / "REEXPRESSED-LEGACY-69-INCREMENT-01"
    previous = {r["episode"]: r for r in read(inherited / "MANIFEST.json")["episodes"]}
    signed = read(increment / "ACTUAL.json")
    corrected = {r["episode"]: pin(inherited / r["label"], r["sha256"]) for r in previous.values()}
    corrected[signed["episode"]] = pin(increment / (signed["episode"] + ".npz"), signed["corrected_labels_sha256"])
    members = [(native_root, r, pin(fixed / r["labels"], r["sha256"]), corrected[r["episode"]])
               for r in received["episodes"] if r["origin"] == "NEW_LEGACY_READMISSION"]
    for name, root in (("LEGACY_RECOVERY", native_root), ("LEGACY-RED-PLACE-REREAD-01", native_root),
                       ("LEGACY-N1-069-REAL-READBACK-01", factory)):
        location = factory / name
        for row in read(location / "ACTUAL.json")["admissions"]:
            identity = row["episode"]["id"]
            require(row["split"] == "TRAIN", "Expected observed TRAIN Teacher progress")
            members.append((root, dict(row, episode=identity), pin(location / row["label"], row["label_sha256"]),
                            corrected[identity]))
    require({m[1]["episode"] for m in members} == set(corrected) and len(members) == 17,
            "Expected all 17 new corrected Teacher episodes exactly once")
    gate = read(parent_gate)["gate"]
    _, _, stats = input_contract(gate, Path(gate["package"]["path"]))
    result = certify(members, stats, set(received["excluded_dev_scenes"]))
    result["legacy_69_manifest_sha256"] = sha(factory / "SNAPSHOTS/TRAIN-LEGACY-069/MANIFEST.json")
    result["qualified_r4_geometry_episodes"] = 69
    write(output, result)
    print(sha(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("native_root", "factory", "retention", "parent_gate", "output"):
        parser.add_argument(field, type=Path)
    legacy(**vars(parser.parse_args()))
