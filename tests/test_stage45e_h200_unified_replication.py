from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45e_h200_unified_diagnostics as diagnostics
import stage45e_h200_unified_replication as campaign


def test_static_surfaces_are_cpu_only() -> None:
    result = campaign.static_audit()
    assert result["status"] == "PASS"
    assert result["gpu"] == 0
    assert result["model_forward"] == 0
    assert result["training"] == 0
    assert diagnostics.static_audit()["status"] == "PASS"


def test_manifest_has_exact_pi0134_budget_queue_and_continuation_ceiling() -> None:
    manifest = json.loads((ROOT / "manifests/STAGE45E_H200_UNIFIED_REPLICATION_PI0143_V1.json").read_text())
    assert manifest["authority"]["pi_gate"] == "PI_GATE_STATE_0143"
    assert manifest["authority"]["issue_comment"] == 5644782159
    assert manifest["gpu_policy"]["max_science_allocations"] == 8
    assert manifest["gpu_policy"]["previous_science_allocations_consumed"] == 3
    assert manifest["gpu_policy"]["additional_science_allocations_authorized"] == 5
    assert manifest["gpu_policy"]["target_additional_allocations"] <= 2
    assert manifest["training_contract"]["training_lineages"] == 7
    assert len(manifest["training_contract"]["queue"]) == 7
    assert manifest["diagnostic_contract"]["e4"]["total_model_side_calls"] == 1680
    assert manifest["diagnostic_contract"]["d1"]["total_model_side_calls"] == 3840
    assert manifest["diagnostic_contract"]["d2a"]["total_model_side_calls"] == 3456
    assert manifest["diagnostic_contract"]["total_model_side_calls"] == 8976
    assert manifest["diagnostic_contract"]["unit_key"] == "(phase,route)"
    assert len(manifest["diagnostic_contract"]["units"]) == 9
    assert manifest["evidence_contract"]["c10_arm_count"] == 18


def test_pi0143_allocation_manifest_is_append_only_and_fail_closed() -> None:
    manifest = json.loads((ROOT / "manifests/STAGE45E_H200_ALLOCATION_LEDGER_PI0143_V1.json").read_text())
    assert manifest["seeded_consumed_ordinals"] == [1, 2, 3]
    assert manifest["max_science_allocations"] == 8
    assert manifest["future_ordinal_9"] == "FAIL_CLOSED_UNAUTHORIZED"
    validation = campaign.validate_committed_allocation_manifest()
    assert validation["validation"]["science_records"] == 3
    assert validation["validation"]["next_science_ordinal"] == 4


def test_phase_budget_partition_is_exact() -> None:
    assert campaign._expected_phase_totals("E4") == {
        "legacy": {"arms": 4, "conditioning_passes": 96, "action_head_calls": 384, "raw_rows": 96},
        "l1": {"arms": 4, "conditioning_passes": 96, "action_head_calls": 384, "raw_rows": 96},
        "n2": {"arms": 6, "conditioning_passes": 144, "action_head_calls": 576, "raw_rows": 144},
    }
    for phase, expected in (("D1", {"conditioning_passes": 384, "action_head_calls": 3456, "raw_rows": 3456}), ("D2A", {"conditioning_passes": 384, "action_head_calls": 3072, "raw_rows": 1536})):
        observed = {key: sum(value[key] for value in campaign._expected_phase_totals(phase).values()) for key in expected}
        assert observed == expected


def test_no_top_level_model_imports_or_scheduler_mutation() -> None:
    for path in (ROOT / "scripts/stage45e_h200_unified_replication.py", ROOT / "scripts/stage45e_h200_unified_diagnostics.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        assert not imports.intersection({"torch", "gr00t", "transformers", "jax", "tensorflow"})
    source = (ROOT / "scripts/stage45e_h200_unified_replication.py").read_text()
    assert "subprocess.run" in source
    assert '"HF_HUB_OFFLINE": "1"' in source
    assert '"TRANSFORMERS_OFFLINE": "1"' in source
    assert '"HF_DATASETS_OFFLINE": "1"' in source
    assert "qsub(" not in source
    assert "git merge" not in source


def test_high_water_calculation_keeps_margin() -> None:
    expected = campaign.ESTIMATED_TRAINING_ROOT_BYTES * len(campaign.TRAINING_QUEUE)
    assert int(expected * 1.15) + campaign.MIN_FREE_BYTES > expected
    continuation_expected = campaign.ESTIMATED_TRAINING_ROOT_BYTES * 5
    assert int(continuation_expected * 1.15) + campaign.MIN_FREE_BYTES > continuation_expected


def test_only_authorized_l1_pre_model_failure_can_be_retried(tmp_path: Path) -> None:
    root = tmp_path / "l1_seed2223_failed"
    root.mkdir()
    (root / "stage45e_l1_failure.json").write_text(json.dumps({
        "status": "FAIL",
        "training_seed": 2223,
        "model_imported": False,
        "model_construction_started": False,
        "model_construction_completed": False,
        "training_started": False,
        "backward_started": False,
        "optimizer_steps_observed": 0,
        "checkpoint_writes_observed": 0,
        "action_head_forwards": 0,
    }))
    evidence = campaign._authorized_pre_model_retry("l1", 2223, root)
    assert evidence is not None
    assert evidence["model_imported"] is False
    assert campaign._authorized_pre_model_retry("legacy", 2223, root) is None
    (root / "stage45e_l1_failure.json").write_text(json.dumps({"status": "FAIL", "training_seed": 2223, "model_imported": True}))
    assert campaign._authorized_pre_model_retry("l1", 2223, root) is None


def test_checkpoint_acceptance_reconciles_state_and_tree(tmp_path: Path) -> None:
    root = tmp_path / "legacy_seed2223"
    checkpoint = root / "checkpoint-1000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1000}))
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(name.encode())
    entries = campaign.regular_tree_entries(checkpoint, "stage45e_legacy/checkpoint-1000")
    record = {
        "status": "PASS",
        "expected_step": 1000,
        "global_step": 1000,
        "required_state_files_present": True,
        "missing_state_files": [],
        "tree_sha256": campaign.tree_digest(entries),
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
    }
    accepted = campaign._verify_checkpoint_artifact("legacy", root, 1000, record)
    assert accepted["tree_sha256"] == record["tree_sha256"]
    recovery_record = {
        "path": str(checkpoint),
        "global_step": 1000,
        "required_state_files": list(campaign.CHECKPOINT_STATE_FILES),
        "training_steps_verified": 1000,
        "optimizer_steps_verified": 1000,
        "backward_steps_verified": 1000,
        "optimizer_step_log_count": 1000,
        "tree_sha256": record["tree_sha256"],
        "file_count": record["file_count"],
        "total_bytes": record["total_bytes"],
    }
    recovered = campaign._verify_checkpoint_artifact("legacy", root, 1000, recovery_record)
    assert recovered["tree_sha256"] == recovery_record["tree_sha256"]
    (checkpoint / "scheduler.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checkpoint tree hash drift|checkpoint byte-count drift"):
        campaign._verify_checkpoint_artifact("legacy", root, 1000, record)


def test_diagnostic_wrapper_leaves_route_output_fresh_for_canonical_helper() -> None:
    source = (ROOT / "scripts/stage45e_h200_unified_diagnostics.py").read_text()
    run_phase = source[source.index("def run_phase"):source.index("def main")]
    assert "pre_model_sidecar" in run_phase
    assert "write_json(pre_model_sidecar, pre_model_record)" in run_phase
    assert "write_json(output / \"pre_model_admission.json\", pre_model_record)" in run_phase
    assert "sys.path.insert(0, str(paths[\"groot\"]))" in run_phase
    # The wrapper must not create the route output before calling the helper;
    # canonical E4/D1/D2A helpers enforce their own fresh-root contract.
    helper_call = run_phase.index("result = module.e4(route_args)")
    assert "output.mkdir(parents=True, exist_ok=False)" not in run_phase[:helper_call]


def test_v1_parser_default_and_run_phase_stub_cover_every_route_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert campaign.parser().parse_args([]).diagnostic_evidence_root is None
    helper = tmp_path / "synthetic_helper.py"
    helper.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "def prepare_env(*args): pass\n"
        "def require(condition, message):\n"
        "    if not condition: raise RuntimeError(message)\n"
        "def _run(args):\n"
        "    args.output_root.mkdir(parents=True, exist_ok=False)\n"
        "    phase = args.output_root.name.rsplit('_', 1)[-1]\n"
        "    budget = {\"E4\": (4, 1), \"D1\": (9, 9), \"D2A\": (8, 4)}[phase]\n"
        "    counters = {\"conditioning_passes\": 24, \"action_head_calls\": 24 * budget[0], \"raw_rows\": 24 * budget[1], \"training\": False, \"optimizer_steps\": 0, \"Validation-E\": 0, \"Holdout-F\": 0, \"WBC\": 0, \"simulator\": 0, \"hardware\": 0}\n"
        "    (args.output_root / 'seal.json').write_text(json.dumps({'status': 'PASS', 'counters': counters}))\n"
        "    return {'synthetic_test_only': True}\n"
        "def e4(args): return _run(args)\n"
        "def diagnostic(args, phase): return _run(args)\n",
        encoding="utf-8",
    )
    route_repo = tmp_path / "route_repo"
    route_repo.mkdir()
    admission_path = tmp_path / "admission.json"
    admission_path.write_text("{}", encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    source_paths = {}
    for name in ("base_model", "dataset", "groot", "cosmos", "modality", "config", "sealed_baseline", "snapshot_preflight"):
        path = tmp_path / name
        path.mkdir() if name in {"groot", "cosmos"} else path.write_text(name, encoding="utf-8")
        source_paths[name] = path

    def fake_admission(_path: Path, route: str, phase: str, _output: Path):
        return ({"phase_specs": {phase: {route: str(spec_path)}}, "source_identities": {"synthetic_test_only": True}}, {"checkpoints": [{"path": str(tmp_path / "checkpoint"), "step": 1000}], "evaluation_identity": {}, "orchestration_identity": {}})

    def fake_git(_repo: Path, *git_args: str) -> str:
        if git_args[:1] == ("branch",):
            return diagnostics.ROUTE_BRANCHES[current["route"]]
        if git_args[:2] == ("rev-parse", "HEAD"):
            return diagnostics.ROUTE_HEADS[current["route"]]
        return ""

    monkeypatch.setattr(diagnostics, "_validate_admission", fake_admission)
    monkeypatch.setattr(diagnostics, "_gpu_observation", lambda: {"rows": [{"index": 0, "name": "NVIDIA H200", "uuid": "GPU-synthetic", "memory_total_mib": diagnostics.H200_VRAM_MIB, "memory_used_mib": 0}]})
    monkeypatch.setattr(diagnostics, "git", fake_git)
    original_sha256_file = diagnostics.sha256_file

    def fake_sha256_file(path: Path) -> str:
        if Path(path).name.startswith("stats_"):
            return diagnostics.N2_STATS_SHA if current["route"] == "n2" else diagnostics.RUNTIME_STATS_SHA
        return original_sha256_file(path)

    monkeypatch.setattr(diagnostics, "sha256_file", fake_sha256_file)
    current = {}
    for route in diagnostics.ROUTE_HEADS:
        for phase in ("E4", "D1", "D2A"):
            current.update(route=route, phase=phase)
            stats_path = tmp_path / f"stats_{route}"
            stats_path.write_text(json.dumps({"statistics_payload_sha256": diagnostics.N2_STATS_PAYLOAD_SHA if route == "n2" else diagnostics.RUNTIME_STATS_PAYLOAD_SHA}), encoding="utf-8")
            namespace = {
                "route": route,
                "phase": phase,
                "route_repo": route_repo,
                "helper_script": helper,
                "admission": admission_path,
                "output_root": tmp_path / f"{route}_{phase}",
                **source_paths,
                "stats": stats_path,
            }
            result = diagnostics.run_phase(diagnostics.argparse.Namespace(**namespace))
            assert result["status"] == "PASS"
            assert result["route"] == route and result["phase"] == phase


def test_pbs_uses_cluster_routing_queue_and_single_h200() -> None:
    pbs = (ROOT / "hpc/pbs/stage45e_h200_unified_replication.pbs").read_text()
    assert "#PBS -q auto" in pbs
    assert "#PBS -l select=1:ngpus=1:ncpus=12:mem=225gb" in pbs
    assert "#PBS -q gpu" not in pbs
    assert "/hpctmp/e1192548/stage45e_h200_unified_replication_20260912" in pbs


def test_pbs_binds_qualified_native_video_runtime_before_python() -> None:
    pbs = (ROOT / "hpc/pbs/stage45e_h200_unified_replication.pbs").read_text()
    assert "FFMPEG_RUNTIME_ROOT=/scratch/e1192548/STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1__" in pbs
    assert 'TORCHCODEC_PACKAGE_ROOT="${TORCH_SITE}/torchcodec"' in pbs
    assert 'TORCH_LIBRARY_ROOT="${TORCH_SITE}/torch/lib"' in pbs
    assert 'export LD_LIBRARY_PATH="${FFMPEG_RUNTIME_ROOT}/lib:${FFMPEG_RUNTIME_ROOT}/compat:${TORCHCODEC_PACKAGE_ROOT}:${TORCH_LIBRARY_ROOT}:${AMBIENT_LD_LIBRARY_PATH}"' in pbs
    assert pbs.index("export LD_LIBRARY_PATH=") < pbs.index("exec \"$ORCH_PY\"")


def test_pbs_binds_recovered_primary_and_has_no_stale_l1_history() -> None:
    pbs = (ROOT / "hpc/pbs/stage45e_h200_unified_replication.pbs").read_text()
    assert "--l1-history" not in pbs
    assert "--recovered-l1-root" in pbs
    assert "--recovered-l1-acceptance" in pbs
    assert "--recovered-l1-input-v2" in pbs
    assert "--duplicate-l1-root" in pbs


def test_pi0143_p0_checklist_is_closed() -> None:
    checklist = campaign.p0_checklist()
    assert checklist
    assert set(checklist.values()) == {"PASS_ALREADY_PRESENT"}


def test_l1_612342_cannot_be_used_as_primary(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not admissible as primary"):
        campaign.verify_historical(tmp_path, "l1")


def test_l1_recovered_checkpoint_document_tamper_fails_before_tree_read() -> None:
    try:
        mounted = campaign.RECOVERED_L1_ROOT.exists()
    except PermissionError:
        mounted = False
    if not mounted:
        pytest.skip("Hopper recovery evidence is not mounted on this machine")
    with pytest.raises(RuntimeError, match="SHA drift"):
        campaign.verify_recovered_l1(expected_document_hashes={"acceptance": "0" * 64}, synthetic_test_override=True)


def test_legacy_exact_recipe_binds_immutable_config() -> None:
    config = campaign.DEFAULTS["config"]
    try:
        mounted = config.exists()
    except PermissionError:
        mounted = False
    if not mounted:
        pytest.skip("Hopper frozen training config is not mounted on this machine")
    result = campaign.verify_legacy_exact_recipe(campaign.DEFAULTS["legacy_history"], config)
    assert result["config_sha256"] == campaign.CONFIG_SHA
    assert result["trainable_modules"] == ["projector", "diffusion_model"]
    assert result["frozen_modules"] == ["visual", "llm", "vlln"]


def test_legacy_provenance_failure_emits_fresh_h200_decision(tmp_path: Path) -> None:
    decision = campaign.legacy_provenance_decision(tmp_path / "missing-legacy-root", tmp_path / "missing-config")
    assert decision["status"] == "BLOCKED_PROVENANCE"
    assert decision["token"] == "LEGACY_SEED2222_FRESH_H200_REQUIRED_BY_PROVENANCE"
    assert decision["lineage"] is None


def _ledger_record(job: str, ordinal: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "allocation_kind": "SCIENCE",
        "pbs_jobid": job,
        "root": "/tmp/evidence",
        "status": "PRE_MODEL_ADMITTED",
        "model_imported": False,
        "model_forward": 0,
        "training": False,
        "attempted_model_calls": 0,
        "failure_seal_sha256": None,
        "consumed": True,
    }
    if ordinal is not None:
        record["ordinal"] = ordinal
    return record


def test_allocation_ledger_admits_fourth_and_blocks_ninth(tmp_path: Path) -> None:
    ledger = tmp_path / "allocation.jsonl"
    campaign.seed_allocation_ledger(ledger, campaign.HISTORICAL_ALLOCATION_RECORDS)
    appended = campaign.append_allocation_record(ledger, _ledger_record("612920.hopper-m-02"))
    assert appended["record"]["ordinal"] == 4
    assert campaign.read_allocation_ledger(ledger)["records"][3]["pbs_jobid"] == "612920.hopper-m-02"
    full = tmp_path / "full.jsonl"
    campaign.seed_allocation_ledger(full, [_ledger_record(f"job-{i}", i) for i in range(1, 9)])
    with pytest.raises(RuntimeError, match="ALLOCATION_9_BLOCKED"):
        campaign.append_allocation_record(full, _ledger_record("job-9"))


def test_allocation_duplicate_observation_is_idempotent_and_conflict_blocks(tmp_path: Path) -> None:
    ledger = tmp_path / "allocation.jsonl"
    campaign.seed_allocation_ledger(ledger, campaign.HISTORICAL_ALLOCATION_RECORDS)
    first = campaign.append_allocation_record(ledger, _ledger_record("612921.hopper-m-02"))
    same = campaign.append_allocation_record(ledger, first["record"])
    assert same["status"] == "IDEMPOTENT_OBSERVATION"
    changed = dict(first["record"])
    changed["status"] = "CONFLICT"
    with pytest.raises(RuntimeError, match="conflicting ledger record"):
        campaign.append_allocation_record(ledger, changed)


def test_allocation_malformed_ledger_fails_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "allocation.jsonl"
    ledger.write_text("not-json\n")
    with pytest.raises(RuntimeError, match="malformed"):
        campaign.read_allocation_ledger(ledger)


def test_diagnostic_accounting_separates_unique_and_failed_attempts() -> None:
    accepted = {"counts": {"model_side_calls": 1680}}
    attempts = {
        "E4:legacy": [
            {"phase": "E4", "route": "legacy", "root": "/e4-pass", "status": "PASS_ACCEPTED", "counts": {"model_side_calls": 1680}},
            {"phase": "E4", "route": "legacy", "root": "/e4-fail", "status": "FAILED_ATTEMPT", "counts": {"model_side_calls": 12}},
        ]
    }
    report = campaign.diagnostic_accounting(attempts, {"E4:legacy": accepted})
    assert report["accepted_unique_scientific_calls"] == 1680
    assert report["all_attempted_model_calls"] == 1692
    assert report["failed_attempt_model_calls"] == 12
    assert report["reused_readonly_model_calls"] == 0


def test_diagnostic_accounting_counts_readonly_reuse_separately() -> None:
    report = campaign.diagnostic_accounting(
        {"E4:legacy": [{"phase": "E4", "route": "legacy", "root": "/sealed", "status": "PASS_ACCEPTED", "counts": {"model_side_calls": 1680}}]},
        {"E4:legacy": {"counts": {"model_side_calls": 1680}, "reused_readonly": True}},
    )
    assert report["accepted_unique_scientific_calls"] == 1680
    assert report["all_attempted_model_calls"] == 1680
    assert report["reused_readonly_model_calls"] == 0
    assert report["reused_readonly_model_calls_by_unit"]["E4:legacy"] == 0
    assert report["reused_readonly_evidence_model_calls"] == 1680
    assert report["reused_readonly_evidence_model_calls_by_unit"]["E4:legacy"] == 1680


def test_duplicate_diagnostic_passes_block_continuation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostics_root = tmp_path / "diagnostics"
    diagnostics_root.mkdir()
    for name in ("attempt_a_E4_legacy", "attempt_b_E4_legacy"):
        root = diagnostics_root / name
        root.mkdir()
        (root / "unified_phase_seal.json").write_text(json.dumps({"status": "PASS"}))
    monkeypatch.setattr(campaign, "_validate_diagnostic_pass", lambda candidate, **_: dict(candidate, status="PASS_ACCEPTED"))
    with pytest.raises(RuntimeError, match="duplicate complete PASS"):
        campaign.select_diagnostic_attempt(diagnostics_root, "E4", "legacy", spec_path=tmp_path / "spec.json")


def test_campaign_failure_seal_preserves_actual_unknown_state(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    (root / "logs").mkdir(parents=True)
    command_record = {
        "schema": "VANDA_STAGE45E_UNIFIED_SUBPROCESS_RECORD_V1",
        "name": "diagnostic_E4_legacy",
        "returncode": 2,
    }
    (root / "logs/diagnostic_E4_legacy.command.json").write_text(json.dumps(command_record))
    result = campaign.write_campaign_failure_seal(
        root,
        state={
            "failure_stage": "DIAGNOSTICS",
            "last_completed_lineage": {"route": "legacy", "seed": 2224},
            "last_completed_unit": "E4:n2",
            "pending_queue": [{"route": "D1", "seed": 2222}],
            "execution_state": {"model_imported": "UNKNOWN", "model_forward": "UNKNOWN", "training": "UNKNOWN"},
            "execution_lifecycle": {"admission": "PASS", "model_import": "UNKNOWN_AFTER_FAILURE", "forward": "NOT_STARTED"},
        },
    )
    document = result["document"]
    assert document["immutable_children_referenced"] is True
    assert document["last_completed_unit"] == "E4:n2"
    assert document["execution_state"]["model_forward"] == "UNKNOWN"
    assert document["model_execution_started"] == "UNKNOWN"
    assert document["execution_lifecycle"]["model_import"] == "UNKNOWN_AFTER_FAILURE"
    assert document["attempted_model_calls"] is None
    assert document["attempted_model_calls_complete"] is False
    assert document["unknown_counter_attempts"] == 1


def test_v2_scientific_identity_ignores_allocation_timestamp_but_binds_inputs() -> None:
    base = {"schema": "VANDA_STAGE45E_UNIFIED_PHASE_SPEC_V2", "phase": "E4", "route": "legacy", "timestamp_utc": "2026-01-01T00:00:00Z", "checkpoints": [{"tree_sha256": "a" * 64}], "science_contract": {"noise_seed": 4242, "stats_sha256": campaign.RUNTIME_STATS_SHA}}
    later = dict(base, timestamp_utc="2026-01-02T00:00:00Z")
    changed_stats = dict(base, science_contract={"noise_seed": 4242, "stats_sha256": "b" * 64})
    assert campaign.scientific_spec_identity_sha256(base) == campaign.scientific_spec_identity_sha256(later)
    assert campaign.scientific_spec_identity_sha256(base) != campaign.scientific_spec_identity_sha256(changed_stats)


def test_v3_missing_seal_counter_is_unknown_not_zero(tmp_path: Path) -> None:
    counts = campaign._attempt_call_counts(tmp_path / "missing")
    assert counts["model_side_calls"] is None
    assert counts["counters_complete"] is False
    assert campaign._sum_known_call_counts([{"counts": counts}]) is None


def test_allocation_correction_is_source_bound_and_append_only(tmp_path: Path) -> None:
    ledger = tmp_path / "allocation.jsonl"
    campaign.seed_allocation_ledger(ledger, campaign.HISTORICAL_ALLOCATION_RECORDS)
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"status": "PRE_MODEL_FAILURE", "job": "612918.hopper-m-02"}), encoding="utf-8")
    correction = {"correction_id": "PI0145-TEST-001", "status": "SOURCE_BOUND_CORRECTION", "attempted_model_calls": None}
    first = campaign.append_allocation_correction(ledger, pbs_jobid="612918.hopper-m-02", correction=correction, source_path=source, source_sha256=campaign.sha256_file(source))
    assert first["status"] == "APPENDED"
    second = campaign.append_allocation_correction(ledger, pbs_jobid="612918.hopper-m-02", correction=correction, source_path=source, source_sha256=campaign.sha256_file(source))
    assert second["status"] == "IDEMPOTENT_OBSERVATION"
    with pytest.raises(RuntimeError, match="conflicting allocation correction"):
        campaign.append_allocation_correction(ledger, pbs_jobid="612918.hopper-m-02", correction=dict(correction, status="CONFLICT"), source_path=source, source_sha256=campaign.sha256_file(source))


def test_v5_catalog_has_six_legacy_and_two_l1_required_references() -> None:
    catalog = campaign.historical_diagnostic_evidence_catalog()
    assert catalog["required_topology"] == {"legacy2222": 6, "l1_seed2222_e4": 2, "total_required_references": 8, "source_rows_are_not_copied": True}
    assert set(catalog["references"]) == {"legacy", "l1"}
    assert sum(len(steps) for steps in catalog["references"]["legacy"].values()) == 6
    assert len(catalog["references"]["l1"]["E4"]) == 2
    assert all(reference["raw_before_derived"] is True for phases in catalog["references"].values() for steps in phases.values() for reference in steps.values())


def test_historical_contract_flag_locations_are_schema_explicit() -> None:
    d1 = {"evidence": {"raw_before_derived": True, "target_xyz_in_model_input": False}}
    campaign._require_historical_contract_flag(d1, container="evidence", field="raw_before_derived", expected=True, label="raw")
    campaign._require_historical_contract_flag(d1, container="evidence", field="target_xyz_in_model_input", expected=False, label="target")
    d2 = {"experiment_contract": {"raw_before_derived": True, "target_xyz_in_model_input": False}}
    campaign._require_historical_contract_flag(d2, container="experiment_contract", field="raw_before_derived", expected=True, label="raw")
    campaign._require_historical_contract_flag(d2, container="experiment_contract", field="target_xyz_in_model_input", expected=False, label="target")
    with pytest.raises(RuntimeError, match="namespace missing"):
        campaign._require_historical_contract_flag(d1, container="experiment_contract", field="raw_before_derived", expected=True, label="raw")
    with pytest.raises(RuntimeError, match="drift"):
        campaign._require_historical_contract_flag({"evidence": {"target_xyz_in_model_input": True}}, container="evidence", field="target_xyz_in_model_input", expected=False, label="target")


def test_l1_raw_manifest_order_and_selected_binding_are_fail_closed(tmp_path: Path) -> None:
    source_root = tmp_path / "l1"
    raw_file = source_root / "raw/L1_at_1000/window-00.npy"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(b"synthetic raw bytes")
    raw_sha = campaign.sha256_file(raw_file)
    array_sha = "b" * 64
    checkpoint_sha = "a" * 64
    window_id = "stage22_expert_red_target_train_a:expert-0000"
    raw_record = {
        "schema": "VANDA_STAGE45E_L1_RAW_PREDICTION_ARTIFACT_V1",
        "path": "raw/L1_at_1000/window-00.npy",
        "file_sha256": raw_sha,
        "array_sha256": array_sha,
        "shape": [1, 40, 132],
        "checkpoint": {"label": "L1@1000", "tree_sha256": checkpoint_sha},
        "window_id": window_id,
    }
    derived_record = {
        "schema": "VANDA_STAGE45E_L1_DERIVED_EVIDENCE_REFERENCE_V1",
        "raw_path": raw_record["path"],
    }
    manifest = source_root / "raw_predictions.jsonl"
    manifest.write_text(json.dumps(raw_record) + "\n" + json.dumps(derived_record) + "\n", encoding="utf-8")
    selected = [{
        "schema": "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1",
        "checkpoint": {"label": "L1@1000", "sha256": checkpoint_sha},
        "window_id": window_id,
        "raw_prediction_artifact": {"path": raw_record["path"], "sha256": raw_sha, "array_sha256": array_sha, "shape": [1, 40, 132]},
    }]
    assert campaign._historical_row_identity(selected[0])["target_id"] == "red_target_train_a"
    proof = campaign._audit_l1_raw_prediction_order(manifest, selected, source_root.resolve())
    assert proof["raw_before_derived"] is True
    assert proof["selected_raw_record_count"] == 1
    manifest.write_text(json.dumps(derived_record) + "\n" + json.dumps(raw_record) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="precedes raw"):
        campaign._audit_l1_raw_prediction_order(manifest, selected, source_root.resolve())


def test_v6_fixture_is_consumed_by_pinned_corelab_verifier(tmp_path: Path) -> None:
    fixture = campaign.build_corelab_producer_fixture(tmp_path)
    assert fixture["synthetic_test_only"] is True
    source = campaign.git(campaign.REPO, "show", f"{campaign.CORELAB_PR65_VERIFIER_HEAD}:{campaign.CORELAB_PR65_VERIFIER_PATH}")
    namespace = {"__name__": "corelab_v6_fixture_verifier", "__file__": str(campaign.REPO / campaign.CORELAB_PR65_VERIFIER_PATH)}
    exec(compile(source, campaign.CORELAB_PR65_VERIFIER_PATH, "exec"), namespace)
    verified = namespace["verify_campaign"](Path(fixture["corelab_campaign"]))
    assert verified["status"] == "PASS_READONLY_PROVENANCE_VERIFIED"
    assert verified["checked_phase_seals"] == 9
    assert fixture["corelab_verifier_source_head"] == campaign.CORELAB_PR65_VERIFIER_HEAD
