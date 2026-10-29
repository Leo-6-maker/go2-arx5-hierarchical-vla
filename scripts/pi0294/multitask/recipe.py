"""Proposed shared-policy training arguments, without admission or model loads.

This is a new learning identity. It does not alter matched E1/E2, turn their
512 endpoint into a new parent, or authorize the proposed 10,000 updates.
"""
from .sampling import TASKS, sha

RAW_REVISION = '2fc962b973bccdd5d8ce4f67cc63b264d6886495'
RAW_FILES_ID = '0e493187af087d156f6352d46ce14e049ff18e7c1cb1eabc8671357ab4b96e71'


def proposed_recipe(*, manifest_sha256, statistics_sha256, seed,
                    microbatch=2, accumulation=16, recovery_round=False):
    if (not sha(manifest_sha256) or not sha(statistics_sha256)
            or type(seed) is not int or seed < 0
            or type(microbatch) is not int or type(accumulation) is not int
            or microbatch <= 0 or accumulation <= 0 or microbatch*accumulation != 32
            or type(recovery_round) is not bool):
        raise ValueError('PINNED_SHARED_DATA_AND_ACTUAL_32_WINDOW_OPTIMIZER_BATCH')
    return dict(schema='SOURCE_SHARED_FOUR_TASK_EEF10_RECIPE_PROPOSAL_V1',
        status='CPU_PREPARATION_NOT_NATIVE_TRAINING_AUTHORITY',
        pretrained_repo='nvidia/GR00T-N1.7-3B', pretrained_revision=RAW_REVISION,
        initial_parent_files_sha256=RAW_FILES_ID,
        initialization='RAW_PRETRAINED_NOT_SEQUENTIAL_SINGLE_TASK_CHECKPOINTS',
        tasks=list(TASKS), target_category=10, shared_weights=True,
        action_dim=10, action_horizon=40, inference_steps=4, execution_prefix=8,
        relative_EEF_processor='EXISTING_PI0292_RELATIVE_EEF_ABSOLUTE_GRIPPER',
        manifest_sha256=manifest_sha256, statistics_sha256=statistics_sha256,
        trainable='TARGET_INTERFACE_AND_DECLARED_E2_SHARED_ACTION_COMPONENTS',
        tune_llm=False, tune_visual=False, visual_adaptation='SEPARATE_DEV_DIAGNOSIS_IF_NEEDED',
        loss='EXISTING_NATIVE_MASKED_FLOW_MATCHING', seed=seed,
        optimizer='adamw_torch', learning_rate=1e-4, weight_decay=1e-5,
        max_steps=10000, warmup_updates=500, effective_batch=32, world_size=1,
        per_device_train_batch_size=microbatch, gradient_accumulation_steps=accumulation,
        development_checkpoints=[2000, 5000, 10000],
        final_TEST='FROZEN_POLICY_ONLY_NO_SELECTION',
        sampling='SHARED_FOUR_TASK_EPISODE_EVENT_RECOVERY_V1', recovery_round=recovery_round,
        new_training_budget_adopted=False)


def trainer_arguments(recipe):
    """Numerical arguments for a future qualified owner, not a launch function.

    Loader, parameter membership, loss, full-state saving and scope/lease checks
    still come from the original training implementation and its new release.
    Recovery-round optimizer origin/horizon needs a separate explicit recipe;
    do not accidentally start another 10k raw-parent round with this template.
    """
    expected = proposed_recipe(manifest_sha256=recipe['manifest_sha256'],
        statistics_sha256=recipe['statistics_sha256'], seed=recipe['seed'],
        microbatch=recipe['per_device_train_batch_size'],
        accumulation=recipe['gradient_accumulation_steps'], recovery_round=recipe['recovery_round'])
    if recipe != expected or recipe['recovery_round']:
        raise ValueError('EXACT_INITIAL_SHARED_RECIPE_RECOVERY_TRAINING_SCHEDULE_NOT_DEFINED')
    return dict(max_steps=10000, warmup_steps=500, learning_rate=1e-4, weight_decay=1e-5,
        optim='adamw_torch', per_device_train_batch_size=recipe['per_device_train_batch_size'],
        gradient_accumulation_steps=recipe['gradient_accumulation_steps'],
        seed=recipe['seed'], data_seed=recipe['seed'], remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_drop_last=True, report_to=[])
