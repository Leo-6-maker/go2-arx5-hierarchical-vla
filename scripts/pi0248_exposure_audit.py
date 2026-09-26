#!/usr/bin/env python3
"""Compare sealed M64's planned 5000-row exposure with actual consumption."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    card = read(a.run_root / "run_card.json")
    training = read(a.run_root / "training_manifest.json")
    cohort = read(a.cohort)["rows"]
    schedule = read(a.schedule)["rows"]
    prefix_path = a.run_root / "training/pi0241_sample_consumption.jsonl"
    prefix = [json.loads(line) for line in prefix_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    recovery_root = a.run_root / "recovery_attempts"
    winners = []
    for resume_path in recovery_root.glob("*/resume_record.json"):
        resume = read(resume_path)
        training_root = resume_path.parent / "training"
        audit_path = training_root / "pi0241_update_audit.json"
        if (resume.get("status") == "SEALED_RESUME_SOURCE" and audit_path.is_file()
                and read(audit_path).get("status") == "COMPLETE"
                and int(read(audit_path).get("global_step", -1)) == 5000):
            winners.append((resume_path, resume, training_root / "pi0241_sample_consumption.jsonl"))
    if len(winners) != 1:
        raise RuntimeError(f"expected one complete exact-resume training path, found {len(winners)}")
    resume_path, resume_record, resume_consumption_path = winners[0]
    resume_rows = [json.loads(line) for line in resume_consumption_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    resume_after = int(resume_record["resume_after_step"])
    if (resume_after != len(prefix) or len(resume_rows) != 5000 - resume_after
            or resume_record.get("schedule_sha256") != card["schedule"]["sha256"]):
        raise RuntimeError("exact-resume evidence does not join the original sample prefix")
    actual = prefix + [row for row in resume_rows if int(row.get("optimizer_step", -1)) > resume_after]
    source_rows = {str(row["source_episode_id"]): row for row in cohort}
    old16 = set(card["support"]["old16_source_ids"])
    assert len(source_rows) == 64 and len(old16) == 16
    assert len(schedule) == len(actual) == 5000
    fields = ("optimizer_step", "fixed_index", "episode_index", "source_episode_id")
    for index, (want, got) in enumerate(zip(schedule, actual), 1):
        if any(want.get(k) != got.get(k) for k in fields):
            raise RuntimeError(f"planned/actual sample identity differs at update {index}")
        if (int(got["optimizer_step"]) != index or int(got["yield_ordinal"]) != index
                or int(got.get("observation_index", -1)) != 0):
            raise RuntimeError(f"actual consumption cursor is not contiguous at update {index}")
    counts = Counter(str(row["source_episode_id"]) for row in actual)
    if set(counts) != set(source_rows):
        raise RuntimeError("actual training did not consume the sealed 64-source support exactly")
    by_group: dict[str, Counter] = defaultdict(Counter)
    for source, count in counts.items():
        row = source_rows[source]
        group = f"{row.get('target_family')}/phase-{row.get('phase')}/{'old16' if source in old16 else 'new48'}"
        by_group[group]["source_count"] += 1
        by_group[group]["update_count"] += count
    bases: dict[str, dict] = {}
    for source, row in source_rows.items():
        base = source.split(":obs-", 1)[0]
        item = bases.setdefault(base, {"support": "old16" if source in old16 else "new48", "source_ids": [], "updates": 0})
        item["source_ids"].append(source)
        item["updates"] += counts[source]
    report = {
        "schema": "PI0248_M64_EXPOSURE_AUDIT_V1", "status": "PASS_EXACT_PLANNED_ACTUAL_5000",
        "run_card_sha256": sha(a.run_root / "run_card.json"),
        "training_manifest_sha256": sha(a.run_root / "training_manifest.json"),
        "cohort_sha256": sha(a.cohort), "schedule_sha256": sha(a.schedule),
        "training_manifest_schedule_sha256": training.get("schedule_sha256"),
        "actual_consumption_sha256": {"prefix": sha(prefix_path), "exact_resume": sha(resume_consumption_path)},
        "exact_resume_record_sha256": sha(resume_path), "exact_resume_after_step": resume_after,
        "planned_rows": len(schedule), "actual_rows": len(actual), "row_by_row_identity_equal": True,
        "unique_sources": len(counts), "unique_rollout_bases": len(bases),
        "old16_updates": sum(counts[s] for s in old16),
        "new48_updates": sum(counts[s] for s in set(counts) - old16),
        "updates_by_source": [{"source_episode_id": source, "support": "old16" if source in old16 else "new48",
                               "target_family": source_rows[source].get("target_family"),
                               "phase": source_rows[source].get("phase"), "updates": counts[source]}
                              for source in sorted(counts)],
        "updates_by_group": {key: dict(value) for key, value in sorted(by_group.items())},
        "rollout_bases": [{"rollout_base": base, **item} for base, item in sorted(bases.items())],
    }
    if report["training_manifest_schedule_sha256"] != card["schedule"]["sha256"]:
        raise RuntimeError("training manifest schedule identity differs from run card")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if a.output.exists() and a.output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to replace conflicting exposure audit: {a.output}")
    a.output.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": report["status"], "sha256": sha(a.output),
        "old16_updates": report["old16_updates"], "new48_updates": report["new48_updates"],
        "unique_sources": report["unique_sources"], "unique_rollout_bases": report["unique_rollout_bases"]}))


if __name__ == "__main__":
    main()
