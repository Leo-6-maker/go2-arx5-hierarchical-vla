"""Bounded adopted series admission; historical fixed pilot remains terminal."""
from pathlib import Path
from .task_capture_runtime import read,sha,within
from .component_teacher import VERSION,cards

AUTHORITY_SHA256='aaf2c430f986f733e5cc7c4c08fbd94799aec32e23a79169ca86b3a8c7d02cd6'
OLD_TERMINAL_SHA256='f6264df2becc0f6178feffd32a720b9e4834604b2b033db3d65f4f6e582831a5'


def require_adoption(root,manifest):
    root=Path(root);folder=root/'teacher-qualification-v2'
    if sha(folder/'HUMAN_ADOPTION.json')!=AUTHORITY_SHA256:raise PermissionError('EXPLICIT_NEW_TEACHER_HUMAN_ADOPTION_REQUIRED')
    a=read(folder/'HUMAN_ADOPTION.json');s=read(folder/'SERIES_LEDGER.json')
    if (folder/'SERIES_CLOSED.json').exists():
        raise PermissionError('COMPONENT_TEACHER_V2_SERIES_TERMINAL_PRESERVED')
    if (a['version']!=VERSION or not a['authorizes_execution'] or a['new_T0'] or a['budget_reset']
            or s['authority_sha256']!=AUTHORITY_SHA256 or sha(root/'TASK_PILOT_CLOSED.json')!=OLD_TERMINAL_SHA256):
        raise PermissionError('TEACHER_SERIES_AUTHORITY_OR_OLD_TERMINAL_CHANGED')
    if sha(root/'teacher-qualification-v1/SERIES_CLOSED.json')!=s['predecessor_terminal_sha256']:
        raise PermissionError('PREDECESSOR_V1_TERMINAL_CHANGED')
    if (folder/'NATIVE_SAFETY_FAIL.json').exists() or (root/'NATIVE_SAFETY_FAIL_V2.json').exists():
        raise PermissionError('TEACHER_VERSION_NATIVE_SAFETY_FAIL_PRESERVED')
    # Revalidate original V2 engineering admission against its actual immutable
    # runtime source. The old terminal alone is superseded FOR THIS new series.
    baseline=dict(manifest,code=s['baseline_code'])
    if sha(root/baseline['code']['source_seal_relative'])!=baseline['code']['source_seal_sha256']:
        raise PermissionError('BASELINE_SOURCE_SEAL_CHANGED')
    from .task_capture_runtime import require_task_asset
    require_task_asset(root,baseline,'FULLBODY_TASK_OVERLAY_V2_WORKING_STROKE',_adopted_teacher=True)
    return a,s


def require_card(root,manifest,episode):
    a,s=require_adoption(root,manifest)
    card=next((c for c in cards() if c['id']==episode),None)
    if card is None:raise PermissionError('UNKNOWN_NEW_TEACHER_FIXED_CONDITION')
    if episode.startswith('V2_FULL_'):
        for dev in ('V2_DEV_W1_T1','V2_DEV_W1_T2'):
            receipt=read(Path(root)/'teacher-qualification-v2/episodes'/dev/'INDEPENDENT_CPU_AUDIT.json')
            if receipt['status']!='PASS_SHORT_MILESTONES_AND_INTEGRITY' or receipt['version']!=VERSION:
                raise PermissionError('SAME_VERSION_SHORT_QUALIFICATION_REQUIRED')
    proof=read(Path(root)/'teacher-qualification-v2/CPU_ADMISSION.json')
    if proof['status']!='PASS_ZERO_PHYSICS_RUNTIME_CONTRACT' or proof['source_sha']!=manifest['code']['vla_commit']:
        raise PermissionError('ACTUAL_CURRENT_SOURCE_CPU_ADMISSION_REQUIRED')
    return card


def prefix_boundary_pause(*,now,deadline,pause,packet,expected_step):
    """Expiry exemption ONLY at an exactly stopped physical prefix boundary.

    No TTL extension. No further step may execute before the next transaction;
    off-boundary, late, running or identity-mismatched state keeps native HOLD.
    """
    return bool(abs(now-deadline)<=1e-8 and pause.get('running') is False and
        pause.get('step')==packet['step']==expected_step and
        abs(packet['time_s']-deadline)<=1e-8)
