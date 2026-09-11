"""Synthetic regression tests for Stage45F sampling/resource/dataset-union prep."""

from scripts import stage45f_dataset_union_audit as union_audit
from scripts import stage45f_resource_scaling_audit as resource_audit
from scripts import stage45f_sampling_mix_audit as sampling_audit


def _base_full24_windows() -> list[dict]:
    rows = []
    for target in union_audit.TARGET_ORDER:
        for index in range(6):
            rows.append(
                {
                    "window_id": f"stage22_expert_{target}:expert-{index:04d}",
                    "source_episode_id": f"stage22_expert_{target}:expert-{index:04d}",
                    "target_id": target,
                }
            )
    return rows


def test_dataset_union_binds_exact_full24_lineage() -> None:
    result = union_audit.analyze(
        {
            "schema": "STAGE45F_DATASET_UNION_PROPOSAL_V1",
            "source_split": "TRAIN",
            "validation_e_read": False,
            "holdout_f_read": False,
            "base_dataset_tree_sha256": union_audit.ACCEPTED_FULL24_TREE,
            "base_windows": _base_full24_windows(),
            "added_windows": [
                {
                    "window_id": "stage45f_new:expert-0000",
                    "source_episode_id": "stage45f_new:expert-0000",
                    "target_id": "red_target_train_new",
                    "label_source": "SIMULATOR_EXPERT",
                    "validation_or_holdout_origin": False,
                    "label_artifact_sha256": "1" * 64,
                }
            ],
        }
    )
    assert result["base_full24_tree_exact"] is True
    assert result["base_window_count"] == 24
    assert result["added_window_count"] == 1
    assert result["union_window_count"] == 25
    assert result["training_authorized"] is False


def test_uniform_window_sampling_changes_base_group_share() -> None:
    result = sampling_audit.analyze(
        {
            "schema": "STAGE45F_SAMPLING_MIX_PROPOSAL_V1",
            "source_split": "TRAIN",
            "validation_e_read": False,
            "holdout_f_read": False,
            "policy": "UNIFORM_WINDOW",
            "groups": [
                {"name": "base", "role": "BASE_FULL24", "window_count": 24, "group_weight": None},
                {"name": "new", "role": "NEW_STATIC_EXPERT", "window_count": 24, "group_weight": None},
            ],
        }
    )
    assert result["base_full24_nominal_group_probability"] == 0.5
    assert result["base_full24_nominal_per_window_probability"] == 1.0 / 48.0
    assert result["exact_realized_exposure_requires_runtime_trace"] is True
    assert result["training_authorized"] is False


def test_explicit_group_weight_can_preserve_declared_base_share() -> None:
    result = sampling_audit.analyze(
        {
            "schema": "STAGE45F_SAMPLING_MIX_PROPOSAL_V1",
            "source_split": "TRAIN",
            "validation_e_read": False,
            "holdout_f_read": False,
            "policy": "EXPLICIT_GROUP_WEIGHT",
            "groups": [
                {"name": "base", "role": "BASE_FULL24", "window_count": 24, "group_weight": 0.75},
                {"name": "rollout", "role": "ROLLOUT_ORACLE_RELABEL", "window_count": 120, "group_weight": 0.25},
            ],
        }
    )
    assert result["base_full24_nominal_group_probability"] == 0.75
    assert result["training_authorized"] is False


def test_resource_port_with_batch_change_is_science_confound() -> None:
    result = resource_audit.analyze(
        {
            "schema": "STAGE45F_RESOURCE_SCALING_PROPOSAL_V1",
            "validation_e_read": False,
            "holdout_f_read": False,
            "baseline": {
                "device_role": "CORELAB_4080S",
                "device_count": 1,
                "per_device_batch": 1,
                "gradient_accumulation_steps": 1,
                "precision": "BF16",
                "distributed_strategy": "SINGLE_GPU",
                "optimizer_identity": "adamw-v1",
                "scheduler_identity": "cosine-v1",
            },
            "proposal": {
                "device_role": "VANDA_H200",
                "device_count": 1,
                "per_device_batch": 2,
                "gradient_accumulation_steps": 1,
                "precision": "BF16",
                "distributed_strategy": "SINGLE_GPU",
                "optimizer_identity": "adamw-v1",
                "scheduler_identity": "cosine-v1",
            },
        }
    )
    assert result["classification"] == "RESOURCE_CHANGE_CONFOUNDS_TRAINING_SEMANTICS_OR_DISTRIBUTION"
    assert "effective_global_batch" in result["science_affecting_differences"]
    assert result["training_authorized"] is False


def test_same_single_gpu_semantics_can_only_be_resource_port_candidate() -> None:
    result = resource_audit.analyze(
        {
            "schema": "STAGE45F_RESOURCE_SCALING_PROPOSAL_V1",
            "validation_e_read": False,
            "holdout_f_read": False,
            "baseline": {
                "device_role": "CORELAB_4080S",
                "device_count": 1,
                "per_device_batch": 1,
                "gradient_accumulation_steps": 1,
                "precision": "BF16",
                "distributed_strategy": "SINGLE_GPU",
                "optimizer_identity": "adamw-v1",
                "scheduler_identity": "cosine-v1",
            },
            "proposal": {
                "device_role": "VANDA_H200",
                "device_count": 1,
                "per_device_batch": 1,
                "gradient_accumulation_steps": 1,
                "precision": "BF16",
                "distributed_strategy": "SINGLE_GPU",
                "optimizer_identity": "adamw-v1",
                "scheduler_identity": "cosine-v1",
            },
        }
    )
    assert result["classification"] == "RESOURCE_PORT_ONLY_CANDIDATE_REQUIRES_NUMERICAL_EQUIVALENCE_CHECK"
    assert result["resource_selected"] is False
