"""One presealed finite control envelope; no action/watchdog/safety changes."""
from .decision import AUTHORITY

PROFILE='PV1_FINITE_POLL250MS_CALL150000_V1'
POLL_S=.25
MAX_CALLS=150000


def validate(manifest,stage):
    from scripts.pi0294.generalization import minimal_core_accounting as first512
    if first512.typed(manifest):
        if (manifest.get('scope')not in(first512.ROLE,first512.PHYSICAL_ROLE)
                or manifest['authority'].get('scope_grant_sha256')!=first512.GRANT_SHA
                or stage.get('task')!=first512.plan.STAGE or stage.get('scope_grant_sha256')!=first512.GRANT_SHA
                or manifest['authority'].get('protocol_ref')!=first512.campaign.TASK_SOURCE
                or stage.get('authority_release')!=first512.campaign.TASK_SOURCE
                or manifest.get('execution',{}).get('control_budget_profile')!=PROFILE
                or stage.get('control_budget_profile')!=PROFILE):
            raise PermissionError('RAW512_EXACT_FINITE_CONTROL_AUTHORITY_AND_PROFILE')
        return dict(profile=PROFILE,poll_s=POLL_S,max_calls=MAX_CALLS,
            CPU_limit_s=7200,wall_limit_s=28800,conditions_per_panel_max=24,
            one_shot_mutations=True,action_query_and_progress_watchdogs_unchanged=True)
    from . import mix_core_identity as cd
    from .successor_core_context import EVAL_ROLE
    from . import mix_vanda_eval_identity as colocated
    cohost=colocated.typed(manifest)
    successor=cohost or cd.typed(manifest) or manifest.get('scope')==EVAL_ROLE
    authority=cd.s.TASK_SOURCE if successor else AUTHORITY
    if successor:
        roles=(colocated.NEURAL_ROLE,colocated.PHYSICAL_ROLE) if cohost else (cd.c.ROLE,EVAL_ROLE)
        if (manifest.get('scope')not in roles
                or manifest['authority'].get('scope_grant_sha256')!=cd.s.GRANT_SHA
                or stage.get('task')!=cd.s.TASK or stage.get('scope_grant_sha256')!=cd.s.GRANT_SHA):
            raise PermissionError('V5_ACTUAL_FINITE_CONTROL_AUTHORITY_NOT_OLD_B24_PERMISSION')
    if (manifest.get('authority',{}).get('protocol_ref')!=authority
            or stage.get('authority_release')!=authority
            or manifest.get('execution',{}).get('control_budget_profile')!=PROFILE
            or stage.get('control_budget_profile')!=PROFILE):
        raise PermissionError('PV1_PRESEALED_FINITE_CONTROL_PROFILE_BOTH_MANIFEST_AND_STAGE')
    return dict(profile=PROFILE,poll_s=POLL_S,max_calls=MAX_CALLS,
        CPU_limit_s=7200,wall_limit_s=28800,conditions_per_panel_max=24,
        one_shot_mutations=True,action_query_and_progress_watchdogs_unchanged=True)


def worst_case_calls(*,wall_s=28800,conditions=24):
    if type(wall_s) is not int or not 0<wall_s<=28800 or type(conditions) is not int or not 0<conditions<=24:
        raise PermissionError('PV1_ORIGINAL_EIGHT_HOUR_AND_24_CONDITION_CONTROL_BOUND')
    # Every None read sleeps; successful prefixes have finitely many non-None
    # reads and one-shot writes. Count BOTH endpoint operations conservatively.
    return int(wall_s/POLL_S)+conditions*80*5
