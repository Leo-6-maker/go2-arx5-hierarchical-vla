"""Replay a closed, fully retained VLA proposal through original native safety.

This is execution adaptation diagnostics, not a claim that the proposal solves
the short skill. Both reference tracking and the unchanged task score survive.
"""


def run(adapter):
    from .fusion_live_proposal_qualification_contract import episode_config, make_program
    from .fusion_trajectory_qualification_native import run_program
    from .fusion_short_skill import document_hash, score_contract
    episode, inputs = episode_config(adapter.spec)
    contract = score_contract()
    if inputs['task']['score_contract_sha256'] != document_hash(contract):
        raise ValueError('LIVE_PROPOSAL_ORIGINAL_SHORT_SKILL_SCORE_REQUIRED')
    context = dict(schema='WM02_NATIVE_LIVE_PROPOSAL_TRAJECTORY_QUALIFICATION_V3',
        experiment_role='VLA_PLUS_EXPLICIT_TRAJECTORY_ADAPTATION',
        desired_outcome_provenance='DECODED_VLA_INTENT_POSE_REFERENCE_TASK_INTENT_UNQUALIFIED',
        source_query_uuid=inputs['archive']['external_pins']['query_uuid'],
        source_checkpoint_manifest_sha256=inputs['archive']['external_pins']['checkpoint_manifest_sha256'],
        geometry_intervention_scale=episode['geometry_scale'],
        VLA_reference_replays=1, actual_VLA_commands=1, VLA_commands=1,
        VLA_inferences=0, original_Processor_decodes=0,
        task_intent_qualification='NOT_ESTABLISHED_BY_QUERY_OR_TRAJECTORY_TRACKING',
        online_query=False, online_VLA=False, full_task_success=False,
        target_task_qualification=False)
    return run_program(adapter, episode, inputs, program_factory=make_program,
                       report_context=context, task_score_contract=contract)
