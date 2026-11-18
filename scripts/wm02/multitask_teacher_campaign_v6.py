"""Run the sealed M Teacher population behind two observed native interfaces."""
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import traceback

from scripts.pi0292.durable import write
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close, unit_state
from go2arx5_vla.world_model.multitask_scope_v6 import PLAN, SCOPE, read_pin, validate
from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS
from go2arx5_vla.world_model.storage import digest
from .sim_only_campaign_v51 import compact_case, fatal_control_violation, interface_gate, run as run_campaign


def read(path):
    return json.loads(Path(path).read_bytes())


def wait_interfaces(spec):
    root = Path(spec['registered_root'])
    interface_pin = spec['interface_predecessor']
    interface = read_pin(interface_pin)
    prior = Path(interface['registered_root'])
    while time.time() < spec['deadline_epoch'] - spec['case_lead_s'] and not (root/'STOP.json').exists():
        lease = read(prior/'control/LEASE.json')
        state = unit_state('wm02-'+prior.name)
        if lease['spec_sha256'] != interface_pin['sha256']:
            raise ValueError('EXACT_M_INTERFACE_LEASE_REQUIRED')
        write(root/'PROGRESS.private.json',dict(phase='WAIT_ACTUAL_MULTITASK_INTERFACES',
            epoch=time.time(),waited_lease=lease['status'],waited_MainPID=state['pid'],
            new_native_requests=0,new_integrations=0))
        if lease['status']=='CLOSED' and not state['pid'] and not state['cgroup_pids']:
            return interface
        time.sleep(30)
    raise TimeoutError('FINITE_TEACHER_INTERFACE_DEPENDENCY_WINDOW')


def interface_records(interface, world_lock):
    prior = Path(interface['registered_root'])
    terminal = read(prior/'RESULT.private.json')
    if (interface['scope'] != SCOPE or interface['plan_commit'] != PLAN or
            interface['mobile_world_lock'] != world_lock or len(interface['cases']) != 2
            or terminal['exit_code'] or terminal['unresolved_case_launches']
            or terminal['new_verified_native_requests'] != 2):
        raise ValueError('TWO_COMPLETE_ACTUAL_MULTITASK_INTERFACES_REQUIRED')
    receipts = {row['id']:row for row in terminal['records']}
    rows = []
    for path in interface['cases']:
        case = read(path)
        root = Path(case['registered_root'])
        lease = read(root/'control/LEASE.json')
        file = root/'RESULT.private.json'
        result = read(file)
        detail = result['detail'] or {}
        if (lease['status'] != 'CLOSED' or lease['spec_sha256'] != digest(path)
                or receipts[root.name]['result_sha256'] != digest(file)
                or result['actual_native_integrations'] != 5953
                or result['model_generations'] != 0 or case['sim_only_role'] != 'INTERFACE'
                or detail.get('state_ABI') != [41,38,20] or detail.get('RGB_step') != 5953
                or (result['trace_audit'] or {}).get('status') != 'RECORDED_TRACE_CHECKS_PASS'):
            raise ValueError('SEALED_ACTUAL_MULTITASK_INTERFACE_EVIDENCE_REQUIRED')
        row = compact_case(case,result)
        if fatal_control_violation(row):
            raise ValueError('RECONCILE_MULTITASK_INTERFACE_TRACE_VIOLATION')
        rows.append(row)
    if not interface_gate(rows):
        raise ValueError('ACTUAL_MULTITASK_NONCONTACT_INTERFACE_GATE_FALSE')
    return rows


def teacher_row(spec, result):
    row = compact_case(spec,result)
    detail = result.get('detail') or {}
    row.update(instruction=spec['task_instruction_id'],
        cluster=spec['mobile_condition']['layout'],
        score=detail.get('score'),capture=detail.get('capture'),
        label_margin_windows=detail.get('label_margin_windows'),
        label_margin_exclusions=detail.get('label_margin_exclusions'),
        first_label_margin_exclusion=detail.get('first_label_margin_exclusion'),
        stages=detail.get('stages'),full_H40_rejection=detail.get('valid_expert_proposal_rejection'))
    return row


def teacher_coverage(rows):
    teachers = [row for row in rows if row['role']=='EXPERT']
    successes = Counter((r['instruction'],r['family']) for r in teachers if r['completed']
                        and r['task_and_mobile_success'] and not r['technical_failure'] and not r['hard_STOP'])
    cells = {(task,family) for task in INSTRUCTIONS for family in ('F1','F2')}
    qualified = all(successes[cell] >= 2 for cell in cells)
    return dict(verified=len(teachers),task_successes=sum(successes.values()),
        successes_by_cell={task:{family:successes[task,family] for family in ('F1','F2')} for task in INSTRUCTIONS},
        passes_task_counts=qualified,technical_failures=sum(r['technical_failure'] for r in teachers),
        hard_STOPs=sum(r['hard_STOP'] for r in teachers),
        observed_label_margin_windows=sum(r.get('label_margin_windows') or 0 for r in teachers),
        observed_label_margin_exclusions=sum(r.get('label_margin_exclusions') or 0 for r in teachers))


def preflight(spec):
    from .multitask_stage_cards_v6 import verified_teacher_prefix

    if (spec['plan_commit'] != PLAN or spec['scope'] != SCOPE or not spec['physics']
            or spec['new_optimizer_updates_budget'] or spec['new_model_generations_budget']
            or spec.get('resident_server_spec') is not None or spec['case_lead_s'] < 1980):
        raise PermissionError('FINITE_TEACHER_ONLY_SCOPE_REQUIRED')
    population = read_pin(spec['compiled_population'])
    candidates = population['conditions']['TEACHER']
    ids = [row['id'] for row in candidates]
    if (len(ids) != 48 or len(set(ids)) != 48 or ids[:6] != population['first_feasibility_ids']
            or Counter((r['instruction'],r['family']) for r in candidates)
            != Counter({(task,family):4 for task in INSTRUCTIONS for family in ('F1','F2')})):
        raise ValueError('EXACT_SEALED_TEACHER48_POPULATION_REQUIRED')
    panel = read_pin(spec['registered_panel'])
    start = spec['teacher_start_index']
    stop = start + spec['new_physical_requests_budget']
    if not (0 <= start < stop <= 48 and (stop == 6 or start >= 6)):
        raise ValueError('INITIAL_TEACHER6_OR_EXACT_UNSTARTED_SUFFIX_REQUIRED')
    if [r['id'] for r in panel] != ids[start:stop] or len(spec['cases']) != stop-start:
        raise ValueError('TEACHER_SLICE_MUST_MATCH_ORIGINAL_ORDER')
    previous = verified_teacher_prefix(spec['previous_teacher_results'],spec['compiled_population'],
                                       world_lock=spec['mobile_world_lock'])
    if [r['id'] for r in previous] != ids[:start]:
        raise ValueError('ONLY_EXACT_TERMINAL_TEACHER_PREFIX_CAN_PRECEDE_SUFFIX')
    clusters = {cluster['id']:cluster for cluster in population['clusters']}
    for path,item,record in zip(spec['cases'],panel,candidates[start:stop],strict=True):
        case = read(path)
        validate(case)
        cluster=clusters[record['cluster']]
        condition=item['condition']
        if (case['sim_only_role']!='EXPERT' or case['panel_entry_id']!=item['id']
                or case['mobile_group']!=spec['registered_root']
                or case['mobile_world_lock']!=spec['mobile_world_lock']
                or case['ROS_DOMAIN_ID']!=spec['ROS_DOMAIN_ID']
                or case['registered_panel']!=spec['registered_panel']
                or item['instruction']!=record['instruction']
                or item['scene']!=cluster['scene'] or item['geometry']!=cluster['scoring_geometry']
                or condition['layout']!=cluster['id'] or condition['reset']!=record['reset']
                or condition['seed']!=record['noise_seed'] or case['wall_limit_s']>1800):
            raise ValueError('TEACHER_CASE_DOES_NOT_MATCH_SEALED_GEOMETRY_AND_CLOCK')
    return previous


def run(path, spec):
    root = Path(spec['registered_root'])
    try:
        previous = preflight(spec)
        interface = wait_interfaces(spec)
        carried = interface_records(interface,spec['mobile_world_lock'])
        free = shutil.disk_usage(root)
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))) * 1024
        if available < 13*1024**3 or free.free < max(50*1024**3, int(.15*free.total))+8*1024**3:
            raise MemoryError('ACTUAL_TEACHER_HOST_OR_DISK_HEADROOM_INSUFFICIENT')
        write(root/'CAPACITY_ACTUAL.private.json',dict(epoch=time.time(),available_host_bytes=available,
            free_disk_bytes=free.free,total_disk_bytes=free.total))
    except BaseException:
        for name in spec['cases']:
            case = read(name)
            folder = Path(case['registered_root'])
            if not (folder/'SUPERVISOR.private.json').exists() and read(folder/'control/LEASE.json')['status']=='ACTIVE':
                close(case,0,not_started=True,reason='MULTITASK_TEACHER_PRECONDITION')
        write(root/'RESULT.private.json',dict(exit_code=1,source_sha=spec['source_sha'],
            status='TEACHER_NOT_DISPATCHED',new_verified_native_requests=0,native_ticks=0,
            failure=traceback.format_exc(),records=[]))
        close(spec,1)
        return 1
    return run_campaign(path, preflight=lambda _:carried,
        load_carried=lambda rows:rows, compact=teacher_row, bootstrap='v6_bootstrap.py',
        expected_count=2+spec['new_physical_requests_budget'],
        result_schema='SLA_V6_MULTITASK_TEACHER_ACTUAL',
        pilot_evaluate=lambda rows:teacher_coverage(previous+rows)['passes_task_counts'],
        case_lead_s=spec['case_lead_s'])
