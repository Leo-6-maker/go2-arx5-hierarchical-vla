"""CPU/static regression tests for the complete PI0123 L1 execution surface."""

from __future__ import annotations

import ast
from pathlib import Path

from scripts import stage45e_l1_h200_launcher as launcher
from scripts import stage45e_l1_runtime_admission as admission
from scripts import stage45e_l1_train24_eval as evaluation


ROOT = Path(__file__).resolve().parents[1]


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module.split(".", 1)[0])
    return result


def test_static_admission_and_execution_budget_are_frozen() -> None:
    static = admission.validate_static_contract()
    assert static["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert admission.CHECKPOINT_ORDER == ("FULL24@2000", "L1@1000", "L1@2000")
    assert admission.EVAL_ACTION_HEAD_CALLS == 288
    assert admission.EVAL_CONDITIONING_PASSES == 72
    assert admission.EVAL_TOTAL_MODEL_SIDE_CALLS == 360
    assert admission.EVAL_SAMPLER_CONTRACT["seed"] == 4242
    assert admission.EVAL_SAMPLER_CONTRACT["t_values"] == [0.0, 0.25, 0.5, 0.75]
    assert admission.L1_CONTRACT["stats_sha256"] == admission.FULL24_STATS_SHA256
    assert admission.L1_CONTRACT["active_scalar_count"] == 190
    assert len(admission.DECODER_SOURCE_SHA256) == 64
    assert len(admission.METRICS_SOURCE_SHA256) == 64


def test_launcher_and_evaluator_plan_only_are_cpu_static() -> None:
    launch_plan = launcher.plan()
    eval_plan = evaluation.plan()
    assert launch_plan["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert eval_plan["status"] == "PASS_STATIC_NOT_EXECUTED"
    assert launch_plan["pbs"] == 0 and launch_plan["h200"] == 0
    assert eval_plan["pbs"] == 0 and eval_plan["h200"] == 0
    assert eval_plan["raw_before_derived"] is True
    assert eval_plan["return_token"] == admission.FULL_PASS_TOKEN


def test_committed_entrypoints_have_no_top_level_model_import() -> None:
    forbidden = {"torch", "transformers", "gr00t", "cuda"}
    for relative in (
        "scripts/stage45e_l1_h200_launcher.py",
        "scripts/stage45e_l1_h200_train.py",
        "scripts/stage45e_l1_train24_eval.py",
        "scripts/stage45e_l1_runtime_admission.py",
    ):
        assert not _top_level_imports(ROOT / relative).intersection(forbidden), relative


def test_evaluation_seals_raw_before_decode_and_uses_all_three_arms() -> None:
    source = (ROOT / "scripts/stage45e_l1_train24_eval.py").read_text(encoding="utf-8")
    raw_write = source.index("append_jsonl_durable(raw_manifest, {")
    raw_false = source.index('"decoded": False', raw_write)
    decode = source.index("physical, decoder_details = stage45c._decode_v2(")
    assert raw_false < decode
    assert "FULL24@2000" in source and "L1@1000" in source and "L1@2000" in source
    assert "install_l1_loss_patch" not in source


def test_launcher_is_foreign_cwd_empty_pythonpath_and_no_retry() -> None:
    guard = launcher.static_guard()
    assert guard["status"] == "PASS"
    assert guard["foreign_cwd_supported"] is True
    assert guard["empty_pythonpath_supported"] is True
    assert guard["automatic_retry"] is False
    source = (ROOT / "scripts/stage45e_l1_h200_launcher.py").read_text(encoding="utf-8")
    assert '"PYTHONPATH": ""' in source
    assert "--execute" in source
    assert "str(TRAIN_SCRIPT)" in source and "str(EVAL_SCRIPT)" in source


def test_pbs_surface_uses_absolute_launcher_and_never_submits_a_retry() -> None:
    source = (ROOT / "hpc/pbs/stage45e_l1_h200.pbs").read_text(encoding="utf-8")
    assert "#PBS -l select=1:ngpus=1" in source
    assert "unset PYTHONPATH" in source
    assert "stage45e_l1_h200_launcher.py" in source
    assert "qsub" not in source
    assert "retry" in source.lower()
    assert "Validation-E" in source and "Holdout-F" in source


def test_training_binds_nested_offline_model_to_hub_cache_namespace() -> None:
    source = (ROOT / "scripts/stage45e_l1_h200_train.py").read_text(encoding="utf-8")
    assert '"HF_HUB_CACHE": str(hub)' in source
    assert '"HUGGINGFACE_HUB_CACHE": str(hub)' in source
    assert 'config.training.transformers_cache_dir = str(Path(os.environ["HF_HUB_CACHE"]))' in source
    assert 'config.training.transformers_cache_dir = str(Path(os.environ["HF_HOME"]))' not in source


def test_backward_canary_materializes_gr00t_config_directory_before_setup() -> None:
    source = (ROOT / "scripts/stage45e_l1_h200_train.py").read_text(encoding="utf-8")
    directory = 'canary_config_dir = canary_root / "experiment_cfg"'
    assert directory in source
    assert source.index('canary_config_dir.mkdir(parents=True, exist_ok=False)') < source.index(
        'pipeline = Gr00tN1d7Pipeline(config, canary_config_dir)'
    )
