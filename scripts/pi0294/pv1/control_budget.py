"""One presealed finite control envelope; no action/watchdog/safety changes."""
from .decision import AUTHORITY

PROFILE='PV1_FINITE_POLL250MS_CALL150000_V1'
POLL_S=.25
MAX_CALLS=150000


def validate(manifest,stage):
    if (manifest.get('authority',{}).get('protocol_ref')!=AUTHORITY
            or stage.get('authority_release')!=AUTHORITY
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
