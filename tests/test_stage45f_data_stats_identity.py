"""Regression for provenance-only vs numerical normalization changes."""

from scripts import stage45f_data_expansion_audit as audit


def test_provenance_only_stats_identity_change_is_not_data_plus_normalization() -> None:
    proposal = {
        "schema": "STAGE45F_DATA_EXPANSION_PROPOSAL_V1",
        "source_split": "TRAIN",
        "validation_e_read": False,
        "holdout_f_read": False,
        "contains_validation_e_ids": False,
        "contains_holdout_f_ids": False,
        "current_stats": {
            "identity": "stats-old-seal",
            "xyz_center_m": [0.0, 0.0, 0.0],
            "xyz_half_range_m": [0.1, 0.1, 0.1],
        },
        "proposed_stats": {
            "identity": "stats-new-seal-same-values",
            "xyz_center_m": [0.0, 0.0, 0.0],
            "xyz_half_range_m": [0.1, 0.1, 0.1],
        },
        "examples": [
            {
                "window_id": "w0",
                "source_episode_id": "episode-0",
                "target_id": "target-0",
                "label_source": "SIMULATOR_EXPERT",
                "future_xyz_m": [[0.0, 0.0, 0.0]],
            }
        ],
    }
    result = audit.audit(audit.validate(proposal))
    assert result["stats_identity_changed"] is True
    assert result["stats_numeric_normalization_changed"] is False
    assert result["stats_change_detected"] is False
    assert result["intervention_class"].startswith("DATA_ONLY_CANDIDATE")
