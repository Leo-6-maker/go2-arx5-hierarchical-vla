"""Replay the fixed A93/R3 population against hash-bound original command receipts."""

from collections import Counter
import io
import ast
import json
from pathlib import Path
import sys
import tarfile
from tempfile import mkdtemp

import numpy as np

from m38_gpu_preflight_v1 import pin, read, sha
from scripts.sla.core_hopper_native_scope import accepted
from scripts.sla.hopper_site import pinned
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
from v64_adapt_pregrasp_attribution import journal
from v65_go2_hub_dataset import LEGACY, R4, INCREMENTS
from v65_recovery_data import ROSTER
from v7_physical_label_replay import replay_case
import v7_physical_label_replay


INDEX = "9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56"


def source_catalog(work, episodes):
    native = Path("/var/tmp/sla-v6-core-uid2009-owned-20261118")
    legacy = native / "sla-core-multienv-wave-a-catalog-cutover-20261122-02/snapshots/TRAIN-069/MANIFEST.private.json"
    corrected = work.parent / "v6-retention-20261126-01/REEXPRESSED-G7-V1/MANIFEST.json"
    authorities = {}
    inputs = []

    def load(path, digest):
        value = read(pin(path, digest))
        inputs.append(dict(path=str(path), sha256=digest))
        return value

    def add(name, result_hash, scene):
        if name in authorities:
            assert authorities[name] == (result_hash, scene), "Conflicting original episode authority"
        authorities[name] = (result_hash, scene)

    original = load(corrected, R4)["episodes"]
    labels = {r["episode"]: r["sha256"] for r in original}
    raw = {r["id"]: r for r in load(legacy, LEGACY)["episodes"]}
    # The source snapshot also contains five excluded raw episodes. Its 69
    # names are not the corrected curriculum's final legacy membership.
    for row in original:
        assert raw[row["episode"]]["native_result_sha256"] == row["result"]
        add(row["episode"], row["result"], row["scene"])
    locations = read(work / "V7-D1-NATIVE-PREP-02/PHYSICAL-REPLAY-INC-LOCATIONS.json")
    for digest in INCREMENTS:
        for row in load(Path(locations[digest]), digest)["episodes"]:
            ep = row["episode"]
            add(ep["id"], row["native_result_sha256"], ep["scene_sha256"])
            if ep["id"] in labels:
                assert labels[ep["id"]] == row["corrected"]["sha256"]
            labels[ep["id"]] = row["corrected"]["sha256"]
    direct = []
    for wave, (suffix, windows, digest) in enumerate(ROSTER, start=1):
        name = f"V64_RECOVERY_TRAIN_{suffix}"
        seal = load(work / f"RECOVERY-TRAIN-SEALED-{wave:02d}" / name / "ACTUAL.json", digest)
        add(name, seal["native_result"]["sha256"], seal["scene"])
        labels[name] = seal["labels"]["sha256"]
        direct.append(Path(seal["native_result"]["path"]))
    expected = {e["episode"] for e in episodes}
    assert set(authorities) == set(labels) == expected and len(expected) == 96
    for ep in episodes:
        assert labels[ep["episode"]] == ep["label_source_sha256"]
        assert authorities[ep["episode"]][1] == ep["scene_group_sha256"]
    candidates = direct[:]
    # This admitted legacy completion was produced in the separate V5 factory,
    # rather than either of the two earlier collection roots.
    candidates.append(work.parent / "core-factory-v5-20261126-01/LEGACY-N1-069-01/cases/CHV1-TRAIN-F1-HOLD-red-19/RESULT.private.json")
    for root in (native, work):
        for pattern in ("*/cases/*/RESULT.private.json", "*/*/RESULT.private.json"):
            candidates.extend(p for p in root.glob(pattern) if p.parent.name in expected)
    found = {name: [] for name in expected}
    visited = set()
    for path in candidates:
        resolved = path.resolve(strict=True)
        if resolved in visited:
            continue
        visited.add(resolved)
        if sha(path) == authorities[path.parent.name][0]:
            found[path.parent.name].append(path.parent)
    return authorities, found, inputs


def check_widths(case, ep, package):
    with tarfile.open(package / ep["path"], "r") as archive:
        with archive.extractfile("labels.npz") as member, np.load(io.BytesIO(member.read()), allow_pickle=False) as f:
            ticks, mask = f["native_step"], f["valid"]
            prior, widths = f["prior_actual_width_m"], f["actual_width_m"]
    samples = {r["native_end_step"]: r for r in journal(case / "native-01/MOBILE_SAMPLES.private.jsonl")}
    before_error, issued_error = 0., 0.
    for i, tick in enumerate(ticks):
        previous = samples[int(tick)]
        assert previous["ticks"] == 25 and previous["safety_valid"]
        assert abs(previous["gripper_command"]["source_native_s"] - (tick-25)*.002) < 1e-8
        before_error = max(before_error, abs(float(prior[i])-.044*previous["installed_EEF10"][9]))
        for j in np.flatnonzero(mask[i]):
            actual = samples[int(tick+25*(j+1))]
            issued_error = max(issued_error, abs(float(widths[i,j])-.044*actual["installed_EEF10"][9]))
    assert before_error < 1e-12 and issued_error < 1e-12
    return dict(prior_width_max_error_m=before_error, issued_width_max_error_m=issued_error)


def run(work, previous=None):
    output = Path(mkdtemp(prefix="v7-full-physical-label-replay.", dir=work))
    package = Path("/mnt/local/hdd1/UserData/liuyu-nfs/Projects/GR00T-WBC/WM02/slago2-embodiment-a93-r3-hub-20261127-01")
    index = read(pin(package / "index.json", INDEX))
    episodes = index["episodes"]
    assert len(episodes) == 96 and sum(ep["windows"] for ep in episodes) == 8031
    assert Counter(e["split"] for e in episodes) == {"TRAIN": 93, "TRAIN_RECOVERY": 3}
    authorities, found, inputs = source_catalog(work, episodes)
    worker = pin(work / "ADAPT0250-NATIVE-PLACE-01/worker/EXECUTION_SPEC.json", "361709e35c2bf427d5644e7af37a925014ab52861c075d4782749afd371d6bfc")
    _, _, receiver = accepted(read(worker))
    stats = pinned(receiver["action"])["actions"]
    quant = Quantiles(stats["q01"], stats["q99"])
    reused = {}
    previous_value = None
    if previous is not None:
        previous_value = read(pin(previous, "d89564efb97d7f97ad6a57b51da78a0b749d8cc446b43a636e86920e37157359"))
        assert previous_value["public_index_sha256"] == INDEX
        assert previous_value["case_replay_script_sha256"] == sha(Path(v7_physical_label_replay.__file__))
        original = work / "v7-physical-full-code-v2.ONDd8N/v7_full_physical_label_replay.py"
        pin(original, previous_value["replay_script_sha256"])
        trees = [ast.parse(p.read_text()) for p in (original, Path(__file__))]
        functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "check_widths") for tree in trees]
        assert ast.dump(functions[0]) == ast.dump(functions[1]), "The cached width check must be identical"
        reused = {row["episode"]: row for row in previous_value["episodes"]}
        assert len(reused) == len(previous_value["episodes"]) == 95
    rows, errors = [], []
    for ep in episodes:
        name = ep["episode"]
        try:
            assert found[name], "Original native result not found in bounded project catalog"
            # Resolve identity by the original seal, never by replay outcome.
            case = sorted(found[name], key=str)[0]
            lease = read(case / "control/LEASE.json")
            spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
            result = read(pin(case / "RESULT.private.json", authorities[name][0]))
            assert result["completed"] and spec["collection_split"] == "TRAIN"
            assert spec["frozen_files"][spec["model_path"]] == ep["scene_group_sha256"]
            if ep["split"] == "TRAIN_RECOVERY":
                assert spec["evaluation_split"] == "TRAIN_RECOVERY" and result["detail"]["recovery_task_success"]
            if name in reused:
                row = dict(reused[name])
                assert row["result_sha256"] == authorities[name][0] and row["windows"] == ep["windows"]
                for file, key in (("MOBILE_SAMPLES.private.jsonl", "native_samples_sha256"),
                                  ("COMMANDS.private.jsonl", "native_commands_sha256")):
                    pin(case / "native-01" / file, row[key])
                pin(package / ep["path"], ep["archive_sha256"])
            else:
                row = replay_case(case, ep, package, quant)
                row.update(check_widths(case, ep, package))
            row["canonical_original_case"] = str(case)
            row["identical_result_copies"] = [str(p) for p in sorted(found[name], key=str)]
            # Diagnostic roundtrip tolerances, not changes to any physical gate.
            assert np.all(np.asarray(row["max_position_mm_SO3_deg_width_mm"]) <= [.001, .001, .01])
            assert row["action_boundary_clock_error_s"] < 1e-8
            rows.append(row)
            print(json.dumps(dict(episode=name, status="EXACT_CACHED_PASS" if name in reused else "PASS", windows=row["windows"], done=len(rows))), flush=True)
        except Exception as exc:
            errors.append(dict(episode=name, type=type(exc).__name__, message=str(exc)))
            print(json.dumps(errors[-1]), flush=True)
    phases = Counter()
    for row in rows:
        phases.update(row["phases"])
    value = dict(status="FULL_96_PHYSICAL_LABEL_REPLAY_PASS" if len(rows) == 96 and not errors else "INCOMPLETE_OR_FAILED",
        public_index_sha256=INDEX, input_authorities=inputs, replay_script_sha256=sha(Path(__file__)),
        case_replay_script_sha256=sha(Path(v7_physical_label_replay.__file__)),
        cached_previous_receipt=dict(path=str(previous), sha256=sha(previous), reused_episodes=len(reused),
                                    original_errors=previous_value["errors"]) if previous is not None else None,
        expected_episodes=96, expected_windows=8031, checked_episodes=len(rows), checked_windows=sum(r["windows"] for r in rows),
        valid_action_points=sum(r["valid_action_points"] for r in rows), p10_points=sum(r["p10_points"] for r in rows),
        diagnostic_roundtrip_tolerance_mm_deg_mm=[.001, .001, .01],
        maximum_roundtrip_mm_deg_mm=np.max([r["max_position_mm_SO3_deg_width_mm"] for r in rows],axis=0).tolist() if rows else None,
        phase_windows=dict(phases), episodes=rows, errors=errors,
        supervision_contract="FLOAT32_NORMALIZATION_AND_R4_CODEC_THEN_RUNTIME_REANCHOR_MATCH_ORIGINAL_INTERVAL_START_COMMAND",
        controller_selected_reference_directly_observed=False,
        scope_limit="Verifies actual installed nominal commands and prior gripper context, not hidden WBC selected policy or task competence",
        new_native_integrations=0, new_model_forwards=0, new_optimizer_updates=0)
    out = output / "ACTUAL.json"
    out.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(out, sha(out), flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
