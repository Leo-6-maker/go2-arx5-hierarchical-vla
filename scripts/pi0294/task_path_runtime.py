"""One finite approved S1 child, reusing original registry and once-only CAS.

No scheduler, cross-site queue dependency, shared-cap rewrite or legacy series
reopening. All writes remain in the existing newly registered S1 run.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

from .task_path_entry import authority, node_local_paths, registered_snapshot, ENTRY_FILES, APPROVAL_SHA256
from .task_path_teacher import VERSION
from .dec1.source_cpu_once_io import commit


def read(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def raw(value): return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def cards():
    result = []
    for role, source, goal, yaw, duration in (
            ('SHORT', [.45, -.08], [.45, .08], 0., 20.),
            ('CONFIRM', [.455, -.07], [.455, .09], .03, 32.)):
        for task in ('T1', 'T2'):
            result.append(dict(id='S1_'+role+'_'+task, role=role, task=task,
                split='S1_DEVELOPMENT' if role == 'SHORT' else 'S1_CONFIRMATION',
                source_xy=source, goal_xy=goal, cube_yaw=yaw, table_top_m=.44,
                warmup_sim_s=5., duration_s=duration, teacher=VERSION,
                scene_class='W0_FREE_STANDING_NEAR', base_translation_preference_m=0.,
                group_id='S1_'+role, instruction=('Pick up the red cube, place it in the blue target, release and withdraw.'
                    if task == 'T1' else 'Push the red cube along the table into the blue target without lifting, then withdraw.')))
    return result


def cards_for_manifest(manifest):
    if 'recovery_scope' in manifest:
        from .task_path_recovery import SCOPE, APPROVAL_PIN, recovery_cards
        if manifest['recovery_scope'] != SCOPE or manifest.get('recovery_approval_sha256') != APPROVAL_PIN:
            raise PermissionError('S1_EXACT_TWO_CARD_RECOVERY_SCOPE')
        return recovery_cards()
    return cards()


def validate_entry(root, physical_asset=None):
    root = Path(root).resolve(strict=True)
    manifest = read(root/'NATIVE_RUN_MANIFEST.json'); inputs = read(root/'NATIVE_INPUTS.private.json')
    approved = authority((root/'S1_HUMAN_APPROVAL.json').read_bytes(), now_epoch=time.time())
    if 'recovery_scope' in manifest:
        from .task_path_recovery import validate_recovery
        validate_recovery(root, manifest, time.time())
    deadline = datetime.fromisoformat(approved['window_end_UTC']).timestamp()
    if (manifest.get('schema') != 'APPROVED_S1_NATIVE_TASK_MANIFEST_V1'
            or manifest['run_uuid'] != root.name or manifest['owner'] != 'EXISTING_LOCAL_PC'
            or os.uname().nodename != 'corelab-4080s' or os.getuid() != 2009
            or manifest['deadline_epoch'] != deadline
            or sys.executable != inputs['runtime_python'] or manifest['teacher_version'] != VERSION
            or manifest['approval_sha256'] != APPROVAL_SHA256
            or sha(root/'NATIVE_INPUTS.private.json') != manifest['inputs_sha256']
            or sha(root/'NATIVE_SOURCE_SEAL.json') != manifest['source_seal_sha256']):
        raise PermissionError('S1_EXACT_CURRENT_NATIVE_CHILD_IDENTITY')
    code = root/manifest['code']['capsule_relative']; seal = read(root/'NATIVE_SOURCE_SEAL.json')
    if (not Path(__file__).resolve().is_relative_to(code)
            or seal['source_sha'] != manifest['code']['vla_commit']
            or any(sha(code/name) != pin for name, pin in seal['files'].items())):
        raise PermissionError('S1_COMPLETE_IMMUTABLE_NATIVE_SOURCE')
    entry = root/manifest['entry_relative']
    if any(sha(entry/name) != pin for name, pin in ENTRY_FILES.items()):
        raise PermissionError('S1_ORIGINAL_STANDARD_ENTRY_CHANGED')
    lease = read(root/'control/NATIVE_S1_LEASE.json')
    if (lease['status'] != 'ACTIVE' or lease['source_sha'] != manifest['code']['vla_commit']
            or lease['run_uuid'] != root.name or lease['deadline_epoch'] != deadline):
        raise PermissionError('S1_OWN_CURRENT_NATIVE_LEASE_NOT_OLD_CLOSED_QUALIFICATION')
    binding = read(root/'NATIVE_OUTPUT_BINDING.private.json')
    registered = registered_snapshot(Path(binding['snapshot']).read_bytes(), expected_sha256=binding['snapshot_sha256'],
                        run_root=str(root), role=binding['role'])
    if (registered['scope'] != 'APPROVED_S1_NATIVE_TASK' or registered['source_sha'] != manifest['code']['vla_commit']
            or registered['local_lease'] != str(root/'control/NATIVE_S1_LEASE.json')):
        raise PermissionError('S1_ACTUAL_CANONICAL_NATIVE_ROLE_BINDING')
    node_local_paths(inputs['path_roles'])
    adoption=read(root/'NATIVE_SCOPE_ADOPTION.json')
    if (adoption.get('status')!='FULL14_AND_NODE_LOCAL_CPU_READY_FOR_FIRST_RESERVED_NATIVE_START'
            or adoption['source_sha']!=manifest['code']['vla_commit']
            or adoption['approval_sha256']!=manifest['approval_sha256']
            or adoption['forecast_sha256']!=sha(root/'NATIVE_FULL14_FORECAST.json')
            or adoption['entry_probe_sha256']!=sha(root/'NATIVE_ENTRY_AND_IMPORT_PROBE.json')
            or read(root/'NATIVE_FULL14_FORECAST.json')['shortfalls']!={}):
        raise PermissionError('S1_ACTUAL_FULL14_AND_COMPLETE_NATIVE_ENTRY_ADOPTION')
    if (root/'native-s1/NATIVE_SAFETY_FAIL.json').exists():
        raise PermissionError('S1_NATIVE_SAFETY_QUARANTINE_PRESERVED')
    return manifest, inputs


def require_card(root, manifest, episode):
    root = Path(root); table = read(root/'native-s1/FIXED_SERIES_CARDS.json')
    expected = cards_for_manifest(manifest)
    if table['cards'] != expected or table['source_sha'] != manifest['code']['vla_commit']:
        raise PermissionError('S1_FIXED_APPROVED_CARDS_AND_SOURCE')
    selected = next((c for c in expected if c['id'] == episode), None)
    if selected is None: raise PermissionError('S1_ONLY_FIXED_APPROVED_ATTEMPTS')
    if selected['role'] == 'CONFIRM':
        proof = read(root/'native-s1/episodes'/('S1_SHORT_'+selected['task'])/'INDEPENDENT_CPU_AUDIT.json')
        if (proof['status'] != 'PASS_S1_SHORT_MANIPULATION_AND_INTEGRITY'
                or proof['source_sha'] != manifest['code']['vla_commit'] or proof['teacher_version'] != VERSION):
            raise PermissionError('S1_SAME_TASK_SHORT_MUST_PASS_BEFORE_CONFIRMATION')
    return selected


def account_cas(root, before, after, key):
    path = Path(root)/'control/S1_PREPARATION_ACCOUNT.json'
    return commit(path, expected_before_sha256=hashlib.sha256(before).hexdigest(), proposed_bytes=raw(after), request_id=key,
        intent_path=path.with_name(key+'.INTENT.json'), receipt_path=path.with_name(key+'.RECEIPT.json'),
        lock_path=path.with_name('S1_SCOPE_ACCOUNT.owner.lock'))


def reserve(root, card):
    """Conservative whole-attempt effects before native process launch."""
    root = Path(root); manifest, inputs = validate_entry(root)
    if require_card(root,manifest,card['id']) != card:
        raise PermissionError('S1_ONLY_THE_EXACT_ADMITTED_CARD')
    unit = inputs['native_units'][card['id']]
    if unit not in Path('/proc/self/cgroup').read_text() or not os.environ.get('INVOCATION_ID'):
        raise PermissionError('S1_ORIGINAL_OWN_BOUNDED_NATIVE_UNIT_REQUIRED')
    output = subprocess.check_output(['systemctl', '--user', 'show', unit,
        '--property=RuntimeMaxUSec,TimeoutStopUSec,KillMode,CPUQuotaPerSecUSec'],
        env=dict(os.environ,LC_ALL='C'),text=True, timeout=5)
    properties = dict(line.split('=', 1) for line in output.splitlines())
    if (properties != dict(RuntimeMaxUSec='13min 20s', TimeoutStopUSec='10s',
                           KillMode='control-group', CPUQuotaPerSecUSec='4s')):
        raise PermissionError('S1_OWN_NATIVE_UNIT_800S_10S_WHOLE_GROUP_AND_FOUR_CORE_LIMIT')
    path = root/'control/S1_PREPARATION_ACCOUNT.json'; before = path.read_bytes(); value = json.loads(before)
    cpu = 3300; steps = round((5+card['duration_s'])/.002)+3; pairs = round(card['duration_s']/.05)+1
    limits = dict(starts=4, integrations=62012, RGB_pairs=2084)
    if 'recovery_scope' in manifest:
        from .task_path_recovery import validate_recovery
        limits = validate_recovery(root, manifest, time.time())
    if (value.get('native_attempts_admitted') is not True
            or value.get('native_scope_source_sha')!=manifest['code']['vla_commit']
            or value.get('active_native_ticket') or card['id'] in value.get('native_attempts', {})
            or value['starts']+1 > limits['starts'] or value['integrations']+steps > limits['integrations']
            or value['RGB_pairs']+pairs > limits['RGB_pairs']
            or value['Core_CPU_upper_charged_s']+cpu > 13800
            or value['PC_CPU_upper_charged_s'] > 600
            or value['Core_CPU_upper_charged_s']+value['PC_CPU_upper_charged_s']+cpu > 14400):
        raise PermissionError('S1_NO_PENDING_REPLAY_AND_EXACT_APPROVED_CUMULATIVE_LIMITS')
    ticket = dict(stage=card['id'], pid=os.getpid(), proc_start=Path('/proc/self/stat').read_text().split()[21],
        started_epoch=time.time(), cpus=sorted(os.sched_getaffinity(0))[-4:], cores=4,
        cpu_s_cap=cpu, physics_steps_cap=steps, rgb_pairs_cap=pairs,
        source_sha=manifest['code']['vla_commit'])
    if len(ticket['cpus']) != 4: raise PermissionError('S1_FOUR_CPU_AFFINITY_AVAILABLE')
    value['starts'] += 1; value['integrations'] += steps; value['RGB_pairs'] += pairs
    value['Core_CPU_upper_charged_s'] += cpu; value['active_native_ticket'] = ticket
    value.setdefault('native_attempts', {})[card['id']] = dict(status='RESERVED_FIRST_ATTEMPT_NO_REPLAY', ticket=ticket)
    account_cas(root, before, value, 'S1-RESERVE-'+card['id'].replace('_', '-'))
    return ticket


def settle(root, ticket, result):
    root = Path(root); path = root/'control/S1_PREPARATION_ACCOUNT.json'; before = path.read_bytes(); value = json.loads(before)
    if value.get('active_native_ticket') != ticket: raise PermissionError('S1_OWN_EXACT_TERMINAL_TICKET')
    for actual, cap in (('physics_charged', 'physics_steps_cap'), ('RGB_pairs_charged', 'rgb_pairs_cap'),
                        ('conservative_CPU_charged_s', 'cpu_s_cap')):
        if not 0 <= result[actual] <= ticket[cap]: raise PermissionError('S1_ACTUAL_COST_OVERFLOW_REQUIRES_RECONCILIATION')
    value['integrations'] -= ticket['physics_steps_cap']-result['physics_charged']
    value['RGB_pairs'] -= ticket['rgb_pairs_cap']-result['RGB_pairs_charged']
    value['Core_CPU_upper_charged_s'] -= ticket['cpu_s_cap']-result['conservative_CPU_charged_s']
    value['native_attempts'][ticket['stage']] = dict(status='NATIVE_TERMINAL_ONCE_SETTLED', ticket=ticket, result=result)
    value.pop('active_native_ticket')
    return account_cas(root, before, value, 'S1-SETTLE-'+ticket['stage'].replace('_', '-'))
