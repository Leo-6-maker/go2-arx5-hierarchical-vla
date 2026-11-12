"""Cold continuation of an immutable SLA population without replaying a row.

The first native group and its technical failure remain immutable. Only the
two never-started rows may execute, under a separately sealed source/lease.
The wall read wait changes; physics quotas, exact acknowledgements and the
single ENTER publication remain unchanged.
"""
import json
from pathlib import Path
import subprocess
from . import task_learning_grasp_v1 as task
from .stable_task_scene_v1 import pinned
from .storage import digest

INPUT_SCHEMA = 'SLA_NATIVE_COLD_UNEXECUTED_SUFFIX_INPUT_V2'
REPAIR_SHA = '413d96e953a11d790136cd64e3a6f28a2fb13d89fe73ea4b6f55b896c50e6eb8'
ORIGINAL_RESULT_SHA = '2e102f5b7ddc845ea0282e43ca3c1c818a47bdffd4f146e104a5bd06800000bd'
ORIGINAL_OUTCOMES_SHA = 'faa2852890767130a3dd0890f32b65128d6c858902f748615ba040ed3e51b157'


def wait_seconds(spec):
    pin = spec.get('native_pause_read_wait')
    if pin is None:
        return 35
    repair = pinned(pin)
    task.need(spec.get('task_learning_role') == task.SLA_ROLE
        and pin['sha256'] == REPAIR_SHA
        and repair['schema'] == 'SLA_NATIVE_EXACT_PAUSE_READ_WAIT_V1'
        and repair['old_wall_read_wait_s'] == 35 and repair['new_wall_read_wait_s'] == 90
        and repair['ENTER_publications_per_request'] == 1 and repair['ENTER_retries'] == 0
        and repair['exact_step_and_snapshot_equality'] and repair['physical_dt_s'] == .002
        and not any(repair[k] for k in ('physics_quota_changed','WBC_changed','native_safety_changed',
                                      'model_weights_changed','score_changed')),
        'SLA_ONLY_SEALED_READ_WAIT_WITHOUT_CLOCK_RETRY_OR_PHYSICS_CHANGE')
    return 90


def prefix(panel):
    proof = panel.get('cold_continuation')
    if proof is None:
        return []
    task.need(panel['population'] == 'TRAIN4' and proof['schema'] == INPUT_SCHEMA
        and proof['result']['sha256'] == ORIGINAL_RESULT_SHA
        and proof['outcomes']['sha256'] == ORIGINAL_OUTCOMES_SHA,
        'SLA_EXACT_RETAINED_FAILURE_PREFIX_NOT_RESULT_SELECTION')
    result, rows, lease, inp, old = [pinned(proof[k]) for k in ('result','outcomes','lease','input','panel')]
    group = Path(proof['result']['path']).parent
    task.need(lease['status'] == 'CLOSED' and lease['worker_exit_code'] == result['exit_code'] == 1
        and lease['input_sha256'] == proof['input']['sha256'] == digest(group/'INPUT.private.json')
        and result['unresolved_pair'] is None and result['completed_in_group'] == len(rows) == 2
        and result['summary']['requested'] == 4 and not result['summary']['complete']
        and inp['panel_sha256'] == proof['panel']['sha256']
        and inp['panel'] == proof['panel']['path']
        and old['source_sha'] == inp['source_sha'] == result['source_sha']
        and all(panel[k] == old[k] for k in ('schema','task_learning_role','protocol','population',
                                          'method','policies','preregistration','assignments')),
        'SLA_ALL_ORIGINAL_POPULATION_POLICY_AND_COLD_PREFIX_BYTES_RETAINED')
    task.need('MainPID=0' in subprocess.check_output(
        ['systemctl','--user','show','wm02-'+group.name,'--property=MainPID'], text=True),
        'SLA_ORIGINAL_GROUP_ACTUALLY_COLD')
    task.need(rows[0]['layout'] == 'TRAIN01' and rows[0]['technical_valid']
        and rows[0]['policy_proposal_rejected'] and not rows[0]['task_success']
        and rows[1]['layout'] == 'TRAIN07' and not rows[1]['technical_valid']
        and rows[1]['actual_SLA_queries'] == 0
        and rows[1]['first_failure'] == 'RuntimeError: EXACT_NATIVE_PAUSE_NOT_REACHED'
        and not any(r['hard_safety_stop'] for r in rows), 'SLA_RETAIN_POLICY_AND_TECHNICAL_FAILURES')
    for row in rows:
        for suffix in ('','-gpu'):
            root = group/'runs'/(row['native_run']+suffix)
            terminal = json.loads((root/'control/LEASE.json').read_bytes())
            worker = json.loads((root/'WORKER_EXIT.private.json').read_bytes())
            task.need(terminal['status'] == 'CLOSED'
                and terminal['worker_exit_code'] == worker['exit_code']
                and type(worker['exit_code']) is int and worker['exit_code'] >= 0
                and terminal['spec_sha256'] == digest(root/'EXECUTION_SPEC.json'),
                'SLA_EVERY_OLD_PARTNER_CLOSED_WITH_ACTUAL_EXIT')
    wait_seconds(dict(task_learning_role=task.SLA_ROLE,
                      native_pause_read_wait=panel['registration']['native_pause_read_wait']))
    return rows


def register_panel(request):
    """Clone the full fixed declaration, retaining the original source prefix."""
    from contextlib import closing
    import time
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot, once
    from .task_learning_student_registration_v2 import closed_base, actual_site, descriptor
    from .sla_native_v1 import validate_panel, policy_contract
    old = pinned(request['cold_continuation']['panel'])
    n = closed_base(old['registration']['native_base'])
    campaign, root, source = Path(n['campaign_root']), Path(request['root']), Path(request['source_root'])
    actual_site(campaign)
    task.need(root.parent == Path(request['cold_continuation']['panel']['path']).parent.parent
        and all(digest(source/name) == pin for name,pin in request['source_files'].items()),
        'SLA_FRESH_SEALED_COLD_CONTINUATION_CAPSULE')
    panel = dict(old, source_sha=request['source_sha'], cold_continuation=request['cold_continuation'],
        registration=dict(old['registration'], source_root=str(source), source_files=request['source_files'],
                          native_pause_read_wait=request['native_pause_read_wait']))
    for policy in panel['policies'].values():
        policy_contract(policy)
    validate_panel(panel)
    prefix(panel)
    key = root.name.upper()+'-PANEL'
    journal = Journal(campaign/'control/journal', campaign.name)
    with closing(OwnerLock(campaign/'OWNER.lock')):
        task.need(not root.exists() and not journal.path(key).exists(), 'SLA_UNKNOWN_CONTINUATION_NEVER_REPLAY')
        def effect():
            root.mkdir()
            (root/'control').mkdir()
            write(root/'PANEL.private.json', panel)
            write(root/'control/LEASE.json', dict(status='SEALED_NONCOMPUTE_PANEL', planned_conditions=4,
                retained_terminal_prefix=2, new_physical_requests=2, created_epoch=time.time(),
                panel_sha256=digest(root/'PANEL.private.json')))
            registry, before = snapshot(n['registry'])
            task.need(not any(r.get('path') == str(root) for r in registry['entries']), 'SLA_FRESH_CONTINUATION_REGISTRY')
            registry['entries'].append(dict(path=str(root),role=key,scope=INPUT_SCHEMA,owner=n['owner'],
                source_sha=request['source_sha'],local_lease=str(root/'control/LEASE.json'),executor_node=n['host']))
            once(n['registry'],registry,key+'-REGISTRY',expected_before_sha256=before)
            return dict(status='SLA_COLD_UNEXECUTED_SUFFIX_PANEL_SEALED',panel=descriptor(root/'PANEL.private.json'))
        return journal.once(key,dict(prefix=request['cold_continuation'],source_sha=request['source_sha']),effect)
