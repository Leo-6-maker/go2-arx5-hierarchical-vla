"""Exact two-task recovery authority, retaining the closed finite S1 account.

Pure checks and fixed cards. This module creates no run, lease or allowance.
"""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .task_path_entry import APPROVAL_SHA256, authority
from .task_path_teacher import VERSION

SCOPE = 'S1_TWO_TASK_RECOVERY01'
APPROVAL_PIN = 'ef274cd6812dba773d25b8e9027ae16832830eaf6f50c6a24d7c6aac9cc718be'
CLOSED_ACCOUNT_PIN = '1918cbf2e96e158d7cb13681b55c607797a5f582260e3cf1090212d61d29d198'
CLOSED_LEASE_PIN = 'b4605910ff760f6862f231095b3de39f5a22a1dd687ecf52897b27ce7aba492e'
OLD_RUN = '20261029T113139Z-standing-task-path-s1'
LIMITS = dict(starts=5, integrations=42012, RGB_pairs=1282)
CAPS = dict(new_starts_max=2, cumulative_starts_max=5, warmup_s_each_max=5,
    task_s_each_max=32, new_integrations_max=37006, cumulative_integrations_max=42012,
    new_RGB_pairs_max=1282, cumulative_RGB_pairs_max=1282,
    total_Core_and_PC_CPU_s=14400, Core_CPU_s=13800, PC_CPU_s=600, max_cores=4,
    Core_output_and_staging_bytes=8*1024**3, PC_evidence_bytes=1024**3,
    transfer_bytes=8*1024**3, GR00T_queries=0, optimizer_updates=0, PBS=0)


def need(ok, reason):
    if not ok:
        raise PermissionError('S1_RECOVERY_'+reason)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def recovery_authority(raw, original, closed_account, closed_lease, *, run_root, now_epoch):
    """Bind direct new approval to an immutable new run and exact old closure."""
    original_value = authority(original, now_epoch=now_epoch)
    need(type(raw) is bytes and digest(raw) == APPROVAL_PIN, 'EXACT_NEW_HUMAN_APPROVAL')
    value = json.loads(raw)
    need(value['schema'] == 'S1_TWO_TASK_RECOVERY_DIRECT_HUMAN_APPROVAL_V1'
         and value['scope'] == SCOPE and value['answer'] == '批准这两次 S1 恢复运行'
         and value['original_approval_sha256'] == APPROVAL_SHA256
         and value['window_end_UTC'] == original_value['window_end_UTC']
         and value['old_closed_scope_reopened'] is False
         and value['old_failures_and_costs_retained'] is True
         and value['no_automatic_retry'] is True and value['no_seed_replacement'] is True
         and value['ten_day_extension_adopted'] is False and value['WBC_frozen'] is True
         and value['approved'] == CAPS, 'UNCHANGED_SCOPE_CLOCK_AND_CAPS')
    need(type(now_epoch) in (int, float) and math.isfinite(now_epoch)
         and datetime.fromisoformat(value['adoption_recorded_at_UTC']).timestamp() <= now_epoch
         < datetime.fromisoformat(value['window_end_UTC']).timestamp(), 'NEW_APPROVAL_WINDOW')
    root = Path(run_root)
    need(root.is_absolute() and '..' not in root.parts and root.parent.name == 'PI0294'
         and root.name != OLD_RUN and root.name.endswith('-standing-task-path-s1'), 'NEW_RUN_ONLY')
    need(type(closed_account) is bytes and digest(closed_account) == CLOSED_ACCOUNT_PIN
         and type(closed_lease) is bytes and digest(closed_lease) == CLOSED_LEASE_PIN,
         'EXACT_OLD_CLOSED_BYTES')
    account, lease = json.loads(closed_account), json.loads(closed_lease)
    need(account['status'] == 'CLOSED_ENGINEERING_INCOMPLETE_NO_QUALIFIED_SHORT'
         and account['native_attempts_admitted'] is False and not account.get('active_native_ticket')
         and (account['starts'], account['integrations'], account['RGB_pairs']) == (3, 5006, 0)
         and lease['status'] == 'CLOSED' and lease['run_uuid'] == OLD_RUN,
         'OLD_SCOPE_STAYS_CLOSED')
    return value


def recovery_cards():
    return [dict(id='S1_RECOVERY01_'+task, role='RECOVERY', task=task,
        split='S1_ENGINEERING_RECOVERY', source_xy=[.45, -.08], goal_xy=[.45, .08],
        cube_yaw=0., table_top_m=.44, warmup_sim_s=5., duration_s=32., teacher=VERSION,
        scene_class='W0_FREE_STANDING_NEAR', base_translation_preference_m=0.,
        group_id='S1_RECOVERY01', instruction=('Pick up the red cube, place it in the blue target, release and withdraw.'
            if task == 'T1' else 'Push the red cube along the table into the blue target without lifting, then withdraw.'))
        for task in ('T1', 'T2')]


def validate_recovery(root, manifest, now_epoch):
    root = Path(root)
    need(manifest.get('recovery_scope') == SCOPE
         and manifest.get('recovery_approval_sha256') == APPROVAL_PIN
         and manifest.get('carry_forward_account_sha256') == CLOSED_ACCOUNT_PIN,
         'EXACT_MANIFEST_BINDING')
    recovery_authority((root/'S1_RECOVERY_HUMAN_APPROVAL.json').read_bytes(),
        (root/'S1_HUMAN_APPROVAL.json').read_bytes(),
        (root/'CLOSED_S1_ACCOUNT_CARRY_FORWARD.json').read_bytes(),
        (root/'CLOSED_S1_NATIVE_LEASE.json').read_bytes(), run_root=str(root), now_epoch=now_epoch)
    old = json.loads((root/'CLOSED_S1_ACCOUNT_CARRY_FORWARD.json').read_bytes())
    current = json.loads((root/'control/S1_PREPARATION_ACCOUNT.json').read_bytes())
    counters = ('starts', 'integrations', 'RGB_pairs', 'Core_CPU_upper_charged_s',
        'PC_CPU_upper_charged_s', 'transfer_upper_charged_bytes',
        'Core_staging_upper_charged_bytes', 'PC_evidence_upper_charged_bytes')
    need(all(current[k] >= old[k] for k in counters)
         and all(current.get('native_attempts', {}).get(k) == v for k, v in old['native_attempts'].items())
         and current.get('recovery_scope') == SCOPE
         and current.get('carry_forward_account_sha256') == CLOSED_ACCOUNT_PIN,
         'ALL_OLD_ATTEMPTS_AND_COSTS_RETAINED')
    return LIMITS.copy()
