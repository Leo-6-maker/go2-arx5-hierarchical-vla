"""Bind the V1.6 population to the actual frozen Corelab artifacts. CPU only."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "sla_official_pt_transfer")
)
from audit_sealed import check, checkpoint, digest, pin, read, source, target


def collect(root, repo):
    prior = root / "official-transfer-20261124"
    public = repo / "reports"
    checks = {}
    bindings = {
        "O0": (
            prior / "results/O0_verified.json",
            "SLA_OFFICIAL_GO2_TRANSFER/O0_ACTUAL.json",
        ),
        "O1": (prior / "replay05/O1.json", "SLA_OFFICIAL_GO2_TRANSFER/O1_ACTUAL.json"),
        "O3": (
            prior / "results/O3_RECEIVER.json",
            "SLA_OFFICIAL_GO2_TRANSFER/O3_RECEIVER.json",
        ),
        "B2": (
            root / "v14/b2-technical-20261123-01/B2_PUBLIC_READOUT.json",
            "SLA_MEMORY_V14/B2_NATIVE_READOUT.json",
        ),
        "D1": (
            root / "v15/matched-verified01/D1_VERIFIED.json",
            "SLA_MEMORY_D1_MATCHED/D1_VERIFIED.json",
        ),
        "D3": (
            root / "v15/d3-verified01/D3_VERIFIED.json",
            "SLA_MEMORY_D3_KNOWN_PATH/D3_VERIFIED.json",
        ),
        "F1": (
            root / "feedback/F1_VERIFIED.json",
            "SLA_FROZEN_FEEDBACK/F1_ACTUAL.json",
        ),
        "F2": (
            root / "feedback/F2_VERIFIED.json",
            "SLA_FROZEN_FEEDBACK/F2_ACTUAL.json",
        ),
        "decision": (
            root / "feedback/final-report01/DECISION_ACTUAL.json",
            "SLA_FROZEN_FEEDBACK/DECISION_ACTUAL.json",
        ),
    }
    docs = {}
    for k, (path, ref) in bindings.items():
        checks[k] = pin(path, digest(public / ref))
        docs[k] = read(path)
    sys.path.insert(0, str(root / "v12"))
    ckpt = checkpoint(root)
    check(
        ckpt == docs["O0"]["checkpoint"]
        and source(root) == docs["O0"]["source"]
        and target(root) == docs["O0"]["target"],
        "O0 cold byte replay",
    )
    manifest = root / "v13/run01/S0_ACTUAL.json"
    pin(manifest, "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9")
    s0 = read(manifest)
    for file, expected in s0["inputs"].items():
        pin(Path(file), expected)
    windows = []
    for w in docs["O1"]["windows"]:
        x = w["OFFICIAL_WITH_CONTINUOUS_REF"]
        path = prior / "replay05" / w["id"] / "FULL40_ENVELOPE.private.json"
        pin(path, x["shadow"]["rejection_evidence_sha256"])
        reasons = read(path)
        check(
            len(reasons) == x["shadow"]["original_full40_envelope_rejected_windows"],
            "rejection count",
        )
        windows.append(
            dict(
                id=w["id"],
                episode=w["episode"],
                raw_sha256=w["raw_sha256"],
                previews=x["shadow"]["installs"],
                reject=len(reasons),
                before_source=x["rejected_starts_before_source"],
                on_source=x["rejected_starts_on_source"],
                events=len(x["source_events"]),
            )
        )
    totals = {
        k: sum(r[k] for r in windows)
        for k in ("previews", "reject", "before_source", "on_source", "events")
    }
    check(list(totals.values()) == [1718, 420, 103, 317, 8], "wrong O1 population")
    native = {}
    for label, folder in [
        ("D1", "v15/sla-d1-matched-b2-20261123-01"),
        ("D3", "v15/sla-d3-known-paths-20261123-01"),
        ("F1", "feedback/sla-feedback-f1-a0-20261123-01"),
        ("F2", "feedback/sla-feedback-f2-a1-20261123-01"),
    ]:
        rows = docs[label]["records"]
        for r in rows:
            case = root / folder / r.get("slot", r.get("card", r.get("id")))
            pin(case / "RESULT.private.json", r["result_sha256"])
        native[label] = {
            k: v
            for k, v in docs[label].items()
            if k
            in (
                "started",
                "screened",
                "unstarted",
                "complete",
                "native_ticks",
                "physics_ticks",
                "starts",
                "attempted",
                "status_counts",
                "scientific_comparisons",
            )
        }
    spec = read(
        root / "feedback/sla-feedback-f2-a1-20261123-01/F2-01/EXECUTION_SPEC.json"
    )
    hardware = {}
    for k in (
        "native_binary",
        "solver_binary",
        "controller_binary",
        "outer_library",
        "trace_library",
        "observer_library",
        "model_path",
        "mobile_safety_library",
    ):
        f = Path(spec[k])
        hardware[k] = pin(f, spec["frozen_files"][str(f)])
    return dict(
        stage="V0_BASELINE_ACTUAL",
        status="VERIFIED",
        source_hashes=checks,
        frozen={
            "upstream": ckpt["source_commit"],
            "revision": ckpt["revision"],
            "manifest": ckpt["manifest_sha256"],
            "checkpoint_files": len(ckpt["checkpoint_files"]),
            "bytes": ckpt["bytes"],
            "norm": ckpt["norm_sha256"],
            "config": ckpt["config"],
            "binary_sha256": hardware,
        },
        sealed_inputs_checked=len(s0["inputs"]),
        source_calls=docs["O0"]["source"],
        target_calls=docs["O0"]["target"],
        windows=windows,
        totals=totals,
        old_direct_legal=docs["O1"]["labels"]["DIRECT_OFFICIAL_RAW"]["TASK_LEGAL"],
        native=native,
        B2={
            k: docs["B2"][k]
            for k in ("actual_requests", "completed_cards", "actual_ticks")
        },
        source_clock_s=1,
        virtual_inputs=["left_wrist", "left_arm"],
        limits=dict(s0["limits"], finger_velocity=0.04),
        student_owner=docs["O3"]["archived_student_owner_receipts"],
        student_native_attempts=0,
        task_successes=0,
        task_success_rate=None,
        blocked_student_windows=16,
        technical_failure_windows=0,
        new_gpu_forwards=0,
        new_native_requests=0,
        new_physics_ticks=0,
        training_updates=0,
        native_stops=0,
        after_stop_ticks=0,
        lease="none acquired",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    doc = collect(args.root.resolve(), Path(__file__).resolve().parents[2])
    with args.output.open("x") as f:
        json.dump(doc, f, indent=2, allow_nan=False)
        f.write("\n")
