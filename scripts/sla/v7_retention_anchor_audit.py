"""Audit direct controller-selection evidence in the fixed closed R4 TRAIN archives."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v7_d0_roster import FIRST_CLOSE, FIRST_EARLY, HISTORY, PLAN, PROTECTED


def audit(site, early_roster, output):
    protected = read(pin(site / "v63-place-core-code-01/PROTECTED_POPULATION.json", PROTECTED))["clusters"]
    held_out = {cluster["scene"]["sha256"] for cluster in protected if cluster.get("split") != "TRAIN"}
    batches = [(site / f"RECOVERY-TRAIN-LIVE-{i:02d}", site / f"RECOVERY-TRAIN-ROSTER-{i:02d}.json", roster,
                site / f"RECOVERY-TRAIN-SEALED-{i:02d}/RESULT.json", result)
               for i, (roster, result) in enumerate(HISTORY, 1)]
    batches.append((site / "V7-D0-EARLY-NATIVE-01", early_roster, FIRST_EARLY,
                    site / "V7-D0-EARLY-SEALED-01/RESULT.json", FIRST_CLOSE))
    cases = []
    members = hashlib.sha256()
    for root, roster, roster_sha, terminal, terminal_sha in batches:
        assignments = read(pin(roster, roster_sha))["cases"]
        settled = read(pin(terminal, terminal_sha))
        require(settled["complete_reconciliation"] and settled["denominator"] == settled["terminal_cases"] == 2
                and settled["actual_training_updates"] == 0 and settled["forward_ledger_reconciled"],
                "Only the original atomically closed TRAIN cohorts may be audited")
        for expected, recorded in zip(assignments, settled["cases"], strict=True):
            case_id = expected["id"]
            require(case_id == recorded["id"] and recorded["state"] == "CLOSED", "The registered TRAIN case changed")
            case = root / case_id
            lease = read(case / "control/LEASE.json")
            spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
            outcome = read(pin(case / "RESULT.private.json", recorded["result"]["sha256"]))
            scene = spec["frozen_files"][spec["model_path"]]
            require(lease["status"] == "CLOSED" and spec["collection_split"] == "TRAIN"
                    and spec["evaluation_split"] == "TRAIN_RECOVERY" and not spec["expert_training_eligible"]
                    and outcome["model_generations"] == recorded["forwards"]
                    and scene == expected["scene_sha256"] and scene not in held_out,
                    "Overlapping split, changed stage or non-TRAIN student identity")
            selected = case / "native-01/SELECTED_CONTROLLER.private.jsonl"
            require(not selected.is_file(), "A direct trace exists and requires action-support verification")
            query_files = sorted((case / "native-01/g7-queries").glob("*/QUERY_ACTUAL.private.json"))
            selection = Counter()
            full = 0
            for path in query_files:
                row = read(path)
                sel = row.get("controller_selected_reference")
                require(sel in (None, "UNKNOWN"), "A directly selected candidate requires review")
                selection["UNKNOWN" if sel else "MISSING"] += 1
                if row.get("full_P10") and not row.get("mock_consumption"):
                    require(len(row["confirmed_intervals"]) == row["confirmed_atoms"] == 10,
                            "Malformed archived full prefix")
                    full += 1
                digest = sha(path)
                members.update(f"{case_id}/{path.parent.name}/{digest}\n".encode())
            require(len(query_files) == recorded["forwards"], "The student-query ledger changed")
            cases.append(dict(id=case_id, scene_sha256=scene, query_records=len(query_files), full_p10_confirmed=full,
                              controller_selected_reference=dict(selection), direct_selection_trace=False,
                              technical=recorded["technical_failure"], eligible_retention_anchors=0))
    require(len(cases) == 12, "The fixed R4 TRAIN audit lost a case")
    write(output, dict(plan=PLAN, scope="FIXED_R4_TRAIN_RECOVERY_SEARCH_FOR_DIRECT_ACTION_SUPPORT",
                       cases=cases, query_records=sum(row["query_records"] for row in cases),
                       confirmed_full_p10=sum(row["full_p10_confirmed"] for row in cases), eligible_retention_anchors=0,
                       audited_query_manifest_sha256=members.hexdigest(), optimizer_updates=0, native_integrations=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("site", "early_roster", "output"):
        p.add_argument(name, type=Path)
    audit(**vars(p.parse_args()))
