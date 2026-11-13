"""Continue the one never-started S1 row after a proven resource refusal.

All completed policy/technical outcomes remain in the fixed denominator.
Waiting is allowed only before any pair directory or Journal effect exists;
the original allocator, host, owner, scope and native gates still decide.
"""
import json
from pathlib import Path
import subprocess
import time

from . import task_learning_grasp_v1 as task
from .stable_task_scene_v1 import pinned
from .storage import digest

SCHEMA = 'SLA_NATIVE_RESOURCE_STOP_SUFFIX_V3'
WAIT_VERSION = 'SLA_UNSTARTED_PAIR_RESOURCE_WAIT_V3'
CAPACITY_REASON = 'TASK_LEARNING_GRASP_FRESH_SHARED_GPU_AND_HOST_HEADROOM_BEFORE_DEV_PAIR'
SOURCE = 'b558a3d9813cfd9a0a973bcd1cccb7de7622db81'
RESULT = '1bbfe0d96fe585800d3af344875db4d081e7b1b779a8efdc3eff36073636bfe5'
OUTCOMES = 'ae6a458c584408c66646ae5a686cccf1e9b8831326f22410e95131491ce515a0'
PANEL = '0cef9eda77f6e6f293c8e3e12b5eb9a32f6dd2f202b594ade696098d169d9918'


def configuration(panel):
    value = panel.get('registration', {}).get('capacity_wait_before_pair_v3')
    if value is None:
        return None
    task.need(panel.get('task_learning_role') == task.SLA_ROLE
              and value == dict(version=WAIT_VERSION, poll_s=60),
              'SLA_EXPLICIT_READONLY_RESOURCE_WAIT_ONLY')
    return value


def joined_prefix(panel, retained, result, rows, lease, inp, prior):
    """Validate the specific completed prefix; never select successful rows."""
    task.need(panel['population'] == prior['population'] == 'TRAIN4'
              and prior['source_sha'] == inp['source_sha'] == result['source_sha'] == SOURCE
              and all(panel[k] == prior[k] for k in ('schema', 'task_learning_role', 'protocol',
                  'population', 'method', 'policies', 'preregistration', 'assignments'))
              and {k: v for k, v in panel['registration'].items()
                   if k not in ('source_root', 'source_files', 'capacity_wait_before_pair_v3')}
                  == {k: v for k, v in prior['registration'].items()
                      if k not in ('source_root', 'source_files', 'capacity_wait_before_pair_v3')},
              'SLA_RESOURCE_SUFFIX_SAME_FIXED_POLICY_POPULATION_AND_NATIVE_INPUTS')
    task.need(lease['status'] == 'CLOSED' and lease['worker_exit_code'] == result['exit_code'] == 1
              and result['failure'] == 'ValueError: ' + CAPACITY_REASON
              and inp['assignment_slice'] == result['assignment_slice'] == [2, 4]
              and result['retained_prefix_conditions'] == 2 and result['completed_in_group'] == 1
              and result['cumulative_completed'] == len(rows) == 3
              and rows[:2] == retained and len(retained) == 2
              and result['summary']['completed'] == 3 and not result['summary']['complete'],
              'SLA_ACTUAL_THREE_ROW_RESOURCE_STOP_PREFIX')
    task.need(rows[2]['layout'] == 'TRAIN13' and rows[2]['attempted']
              and rows[2]['technical_valid'] and rows[2]['policy_proposal_rejected']
              and not rows[2]['task_success'] and rows[2]['actual_SLA_queries'] == 1
              and rows[2]['actual_native_integrations'] == 5703
              and not any(r['hard_safety_stop'] for r in rows),
              'SLA_RETAIN_ACTUAL_TRAIN13_POLICY_REJECTION_NOT_RETRY')
    return rows


def extend_prefix(panel, retained):
    proof = panel['resource_continuation']
    task.need(proof['schema'] == SCHEMA and proof['result']['sha256'] == RESULT
              and proof['outcomes']['sha256'] == OUTCOMES and proof['panel']['sha256'] == PANEL,
              'SLA_EXACT_RESOURCE_STOP_ARTIFACTS')
    result, rows, lease, inp, prior = [pinned(proof[k]) for k in ('result', 'outcomes', 'lease', 'input', 'panel')]
    group = Path(proof['result']['path']).parent
    task.need(proof['result']['path'] == str(group/'RESULT.private.json')
              and proof['outcomes']['path'] == str(group/'OUTCOMES.private.json')
              and proof['lease']['path'] == str(group/'control/LEASE.json')
              and proof['input']['path'] == str(group/'INPUT.private.json')
              and lease['input_sha256'] == proof['input']['sha256'] == digest(group/'INPUT.private.json')
              and inp['panel'] == proof['panel']['path'] and inp['panel_sha256'] == PANEL,
              'SLA_CLOSED_RESOURCE_GROUP_ARTIFACT_JOIN')
    joined_prefix(panel, retained, result, rows, lease, inp, prior)
    wanted = group/'runs'/prior['assignments'][3]['native_run']
    task.need(result['unresolved_pair'] == dict(native_root=str(wanted), stage='REGISTRATION_NOT_COMPLETED_NO_REPLAY')
              and not wanted.exists() and not wanted.with_name(wanted.name+'-gpu').exists()
              and 'MainPID=0' in subprocess.check_output(['systemctl', '--user', 'show',
                  'wm02-'+group.name, '--property=MainPID'], text=True),
              'SLA_OLD_TRAIN25_NEVER_REGISTERED_OR_DISPATCHED')
    native = pinned(inp['native_base'])
    registry = json.loads(Path(native['registry']).read_bytes())
    task.need(not any(e.get('path') in (str(wanted), str(wanted)+'-gpu') for e in registry['entries']),
              'SLA_NO_OLD_TRAIN25_REGISTRY_EFFECT')
    # TRAIN13's actual pair exited; do not substitute the coordinator's state.
    for suffix in ('', '-gpu'):
        root = group/'runs'/(rows[2]['native_run']+suffix)
        terminal = json.loads((root/'control/LEASE.json').read_bytes())
        worker = json.loads((root/'WORKER_EXIT.private.json').read_bytes())
        task.need(terminal['status'] == 'CLOSED' and type(worker['exit_code']) is int
                  and worker['exit_code'] >= 0 and terminal['worker_exit_code'] == worker['exit_code']
                  and terminal['spec_sha256'] == digest(root/'EXECUTION_SPEC.json'),
                  'SLA_TRAIN13_BOTH_REAL_TERMINALS')
    task.need(configuration(panel) is not None, 'SLA_RESOURCE_SUFFIX_REQUIRES_READONLY_WAIT')
    return rows


def wait_for_pair(attempt, untouched, save, deadline, *, stopped=lambda: False,
                  now=time.time, sleep=time.sleep):
    """Retry only a proven effect-free admission check; never a worker/query."""
    while now() < deadline:
        if stopped():
            raise PermissionError('SLA_UNSTARTED_PAIR_STOP_RETAINED')
        try:
            return attempt()
        except (BlockingIOError, ValueError) as error:
            if isinstance(error, ValueError) and str(error) != CAPACITY_REASON:
                raise
            if not untouched():
                raise RuntimeError('SLA_PAIR_EFFECT_OR_UNKNOWN_NO_ADMISSION_REPLAY') from error
            save(dict(status='WAIT_ORIGINAL_PAIR_OWNER_OR_CAPACITY', epoch=now(),
                      effect_started=False, model_calls=0, physical_requests=0))
        sleep(min(60, max(0, deadline-now())))
    raise TimeoutError('SLA_FINITE_UNSTARTED_PAIR_WAIT_EXPIRED_NO_DISPATCH')
