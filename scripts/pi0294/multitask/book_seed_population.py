"""Declared source book positions for the Core maintenance experiment.

These are positions of the same book and caddy assets, not independent asset
instances or layouts. Confirmation is sealed before collecting new TRAIN.
"""

VERSION = 'BOOK_M32_POSITION_AND_EARLY_RECOVERY_V1'
ORIGINAL_TRAIN_OFFSETS_M = ((0., 0.), (.004, 0.), (-.004, 0.), (0., .004), (0., -.004))
ADDITIONAL_TRAIN_OFFSETS_M = (
    (.012, 0.), (-.012, 0.), (0., .012), (0., -.012),
    (.010, .010), (.010, -.010), (-.010, .010), (-.010, -.010),
    (.016, .004), (-.016, -.004),
)
ORIGINAL_DEV_OFFSETS_M = (
    (.006, 0.), (-.006, 0.), (0., .006), (0., -.006),
    (.004, .004), (.004, -.004), (-.004, .004), (-.004, -.004),
    (.008, .002), (-.008, -.002),
)
CONFIRM_OFFSETS_M = tuple((x, y) for x in (-.018, -.009, .009, .018)
                        for y in (-.018, -.009, .009, .018)) + (
    (.022, 0.), (-.022, 0.), (0., .022), (0., -.022),
)


def plan():
    return dict(schema=VERSION,
        original_TRAIN=[dict(id=f'BOOK_SOURCE_SEED{i:03}', xy_offset_m=list(p))
                        for i, p in enumerate(ORIGINAL_TRAIN_OFFSETS_M)],
        additional_TRAIN=[dict(id=f'BOOK_SOURCE_SEED{i:03}', xy_offset_m=list(p))
                          for i, p in enumerate(ADDITIONAL_TRAIN_OFFSETS_M, 5)],
        original_DEV=[dict(id=f'BOOK_DEV{i:03}', xy_offset_m=list(p))
                      for i, p in enumerate(ORIGINAL_DEV_OFFSETS_M)],
        confirmation=[dict(id=f'BOOK_M32_CONFIRM{i:03}', xy_offset_m=list(p),
                           noise_seed=2026110200+i) for i, p in enumerate(CONFIRM_OFFSETS_M)],
        confirmation_for_TRAIN_statistics_selection=False,
        confirmation_replaces_original_TEST=False,
        position_only_same_book_caddy_camera_home_and_goal=True,
        task_seconds=32, warmup_seconds=5, H40=40, prefix=8,
        original_scoring_safety_and_WBC_unchanged=True,
        collection_requires_full_static_path_and_native_episode_audit=True,
        failed_generation_attempts_retained=True,
        recovery=dict(TRAIN_parents=list(range(5)), takeover_indices=[8, 16],
                      checkpoint_manifest_sha256='b248513ae76f0bb1ec29978a7c7242eabb5155063d8e732f61e182035a676b8c',
                      total_attempts=10, student_prefix_is_expert=False,
                      only_actual_completed_expert_suffix_eligible=True,
                      actual_safe_state_and_full_task_audit_required=True),
        capacity_comparison_requires_same_new_data_statistics_parent_and_exposure=True,
        optimizer_updates=0, physics_integrations=0, training_admitted=False)
