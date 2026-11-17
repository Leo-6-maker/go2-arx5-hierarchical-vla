"""Prepare finite V6 M cards over the sealed scenes and existing interfaces."""
import copy
import json
from pathlib import Path
import time

from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.multitask_contract_v6 import NATIVE_METHOD
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs, unit_state
from go2arx5_vla.world_model.multitask_scope_v6 import PLAN, SCOPE, read_pin, validate
from go2arx5_vla.world_model.multitask_score_v6 import Task
from go2arx5_vla.world_model.storage import digest
from .sim_only_campaign_v51 import fatal_control_violation


def read(path):
    return json.loads(Path(path).read_bytes())


def pin(path):
    return dict(path=str(path),sha256=digest(path))


def language_for(instruction):
    task = Task.parse(instruction)
    if task.operation == 'HOLD':
        return f'Grasp the {task.color} cube, lift it, and hold it for one second while moving in the task direction.'
    return f'Grasp the {task.color} cube, lift it, carry it to the {task.destination} region, release it, and move the gripper clear.'


def verified_teacher_prefix(pins, population_pin, *, world_lock):
    """Read closed, reconciled Core results. Existing scientific failures stay in the prefix."""
    population = read_pin(population_pin)['conditions']['TEACHER']
    records = []
    for result_pin in pins:
        file = Path(result_pin['path'])
        root = file.parent
        path = root/'EXECUTION_SPEC.json'
        parent = read(path)
        result = read_pin(result_pin)
        unit = unit_state('wm02-'+root.name)
        lease = read(root/'control/LEASE.json')
        if (file.name != 'RESULT.private.json' or Path(parent['registered_root']) != root
                or lease['status'] != 'CLOSED' or lease['spec_sha256'] != digest(path)
                or unit['pid'] or unit['cgroup_pids'] or parent['owner'] != lease['owner']
                or parent['plan_commit'] != PLAN or parent['scope'] != SCOPE or parent['mobile_world_lock'] != world_lock
                or parent['compiled_population'] != population_pin or result['source_sha'] != parent['source_sha']
                or result.get('unresolved_case_launches',0)):
            raise ValueError('RECONCILE_ACTUAL_CLOSED_TEACHER_PREDECESSOR_BEFORE_ADVANCE')
        entries = [row for row in result['records'] if row['role']=='EXPERT']
        if result['new_verified_native_requests'] != len(entries):
            raise ValueError('ACTUAL_TEACHER_REQUEST_DENOMINATOR_MUST_MATCH_CLOSED_RECEIPTS')
        cases = {Path(case).parent.name:Path(case) for case in parent['cases']}
        for row in entries:
            child = cases[row['id']]
            native = read_pin(dict(path=str(child.parent/'RESULT.private.json'),sha256=row['result_sha256']))
            owned = read(child.parent/'control/LEASE.json')
            if (owned['status'] != 'CLOSED' or owned['spec_sha256'] != digest(child)
                    or row['actual_native_requests'] != 1 or native['physical_requests'] != 1
                    or native['model_generations'] or native['optimizer_updates']
                    or row['native_ticks'] != native['actual_native_integrations']
                    or fatal_control_violation(row) or row['technical_failure']):
                raise ValueError('FIRST_NATIVE_FAILURE_REQUIRES_EXPLICIT_RECONCILIATION_BEFORE_SUFFIX')
        records.extend(entries)
    if [row['id'] for row in records] != [row['id'] for row in population[:len(records)]]:
        raise ValueError('CLOSED_TEACHER_RESULTS_MUST_BE_ONE_EXACT_ORDERED_PREFIX')
    return records


def teacher_slice(population, prefix, stop):
    planned = population['conditions']['TEACHER']
    start = len(prefix)
    if ([row['id'] for row in prefix] != [row['id'] for row in planned[:start]]
            or type(stop) is not int or not 0 <= start < stop <= 48
            or (start<6 and stop!=6) or len(planned)!=48):
        raise ValueError('ONLY_NEXT_COMPLETE_TEACHER_FEASIBILITY_OR_FIXED_SUFFIX')
    return planned[start:stop]


def panel_cards(population, planned, templates, *, role):
    if role not in ('EXPERT','STUDENT') or set(templates) != {'F1','F2'}:
        raise ValueError('TWO_BOUND_INTERFACE_TEMPLATES_AND_ORIGINAL_M_ROLE_REQUIRED')
    clusters = {item['id']:item for item in population['clusters']}
    panel = []
    for row, arm in planned:
        cluster = clusters[row['cluster']]
        family = row['family']
        if cluster['family'] != family or cluster['reset'] != row['reset']:
            raise ValueError('SEALED_M_CLUSTER_AND_CONDITION_IDENTITY_REQUIRED')
        cid = row['id'] if role == 'EXPERT' else f"{row['id']}-{arm}"
        condition = dict(templates[family]['mobile_condition'],mode='M_FREE',
            method='EXPERT' if role=='EXPERT' else NATIVE_METHOD,family=family,layout=row['cluster'],
            reset=row['reset'],seed=row['noise_seed'],collection_episode=cid)
        panel.append(dict(id=cid,role=role,instruction=row['instruction'],condition=condition,
            geometry=cluster['scoring_geometry'],scene=cluster['scene'],arm=arm))
    return panel


def case_card(spec, template, item, panel_pin):
    root = Path(spec['registered_root'])
    folder = root/'cases'/item['id']
    task = Task.parse(item['instruction'])
    cap = 35703 if task.operation == 'HOLD' else 50703
    expert = item['role']=='EXPERT'
    role = f"SLA_V6_M_{item['role']}_{item['condition']['family']}"
    case = copy.deepcopy(template)
    for key in ('plan_commit','goal_plan_sha','source_sha','source_root','source_files','trusted_source_parent',
                'source_seal','registry','owner','host','uid','world_control','mobile_world_lock','ROS_DOMAIN_ID',
                'allocation_deadline_epoch','campaign_deadline_epoch','deadline_epoch','sim_only_project_authorization'):
        case[key] = spec[key]
    case.update(registered_root=str(folder),mobile_group=str(root),campaign_root=str(root),role=role,
        scope=SCOPE,sim_only_role=item['role'],registered_panel=panel_pin,panel_entry_id=item['id'],
        task_instruction_id=task.instruction,mobile_condition=item['condition'],multitask_geometry=item['geometry'],
        language=language_for(task.instruction),wall_limit_s=1800,max_ticks=cap,policy_queries=(cap-5703)//250,
        new_physical_requests_budget=1,model_service_timeout_s=30,expert_training_eligible=expert,
        model_path=item['scene']['path'],unattended_control=str(root/'control'))
    case['collection_contract'] = ('SLA_V6_MULTITASK_TEACHER_CURRENT_FULL_H40_AND_CONSUMED_MASK' if expert
                                   else 'SLA_V6_MULTITASK_G7_STUDENT_EVALUATION_ONLY')
    case['new_model_generations_budget'] = 0 if expert else case['policy_queries']
    case['frozen_files'][case['model_path']] = item['scene']['sha256']
    for kind in ('tmp','cache','logs','data'):
        case[kind] = str(folder/kind)
    # Keep the reviewed native interpreter and command isolation. Only this
    # unopened case's writable locations and source import overlay change.
    case['native_environment'].update(PYTHONPATH=spec['native_environment']['PYTHONPATH'],
        CUDA_VISIBLE_DEVICES='',ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',
        ROS_DOMAIN_ID=str(spec['ROS_DOMAIN_ID']),TMPDIR=case['tmp'],XDG_CACHE_HOME=case['cache'],ROS_LOG_DIR=str(folder/'logs/ros'))
    if not expert:
        arm = item['arm']
        case.update(arm=arm,packages=spec['packages'],G7_package=spec['packages'][arm],
            ordinary_RGB_restore=spec['ordinary_RGB_restores'][arm],resident_server_spec=spec['resident_server_spec'])
    return folder, case


def require_unstarted_cases(registry, case_ids):
    """A prior registered card is reusable only when its non-dispatch is proven."""
    for entry in read(registry)['entries']:
        root = Path(entry['path'])
        if root.name not in case_ids:
            continue
        lease = read(root/'control/LEASE.json')
        terminal_file = root/'WORKER_EXIT.private.json'
        if (lease['status'] != 'CLOSED' or not terminal_file.is_file()
                or read(terminal_file).get('not_started') is not True
                or (root/'SUPERVISOR.private.json').exists() or (root/'native-01').exists()
                or (root/'RESULT.private.json').exists()):
            raise PermissionError('EXISTING_NATIVE_CONDITION_REQUIRES_RECONCILIATION_OR_IS_ALREADY_CONSUMED:'+root.name)


def prepare_cards(spec, *, teacher_stop=None):
    """Create artifacts for a new owned stage, leaving service launch to its caller."""
    from .multitask_teacher_campaign_v6 import interface_records

    root = Path(spec['registered_root'])
    now = time.time()
    if (not now+1980 < spec['deadline_epoch'] <= min(now+43200,spec['campaign_deadline_epoch'],spec['allocation_deadline_epoch']-60)
            or spec['scope'] != SCOPE or spec['plan_commit'] != PLAN
            or (root/'EXECUTION_SPEC.json').exists() or (root/'control'/'LEASE.json').exists()
            or (root/'STOP.json').exists()):
        raise PermissionError('NEW_OWNED_CARDS_REQUIRE_LIVE_FINITE_STAGE_WINDOW')
    interface = read_pin(spec['interface_predecessor'])
    predecessor = Path(interface['registered_root'])
    unit = unit_state('wm02-'+predecessor.name)
    lease = read(predecessor/'control/LEASE.json')
    if (unit['pid'] or unit['cgroup_pids'] or lease['status']!='CLOSED'
            or lease['spec_sha256']!=spec['interface_predecessor']['sha256']):
        raise ValueError('M_INTERFACE_OWNER_MUST_HAVE_REACHED_COLD_TERMINAL')
    interface_records(interface,spec['mobile_world_lock'])
    templates = {case['mobile_condition']['family']:case for case in map(read,interface['cases'])}
    population = read_pin(spec['compiled_population'])
    if teacher_stop is not None:
        prefix = verified_teacher_prefix(spec['previous_teacher_results'],spec['compiled_population'],
                                          world_lock=spec['mobile_world_lock'])
        selected = [(row,None) for row in teacher_slice(population,prefix,teacher_stop)]
        role = 'EXPERT'
    else:
        from .multitask_student_campaign_v6 import expected_conditions

        selected = expected_conditions(population,spec['evaluation_stage'],set(spec['packages']))
        role = 'STUDENT'
    panel = panel_cards(population,selected,templates,role=role)
    require_unstarted_cases(spec['registry'],{row['id'] for row in panel})
    (root/'cases').mkdir()
    panel_file = root/'REGISTERED_PANEL.private.json'
    write(panel_file,panel)
    panel_pin = pin(panel_file)
    entries = [case_card(spec,templates[row['condition']['family']],row,panel_pin) for row in panel]
    for folder, case in entries:
        validate(case)
        dirs(folder)
    parent = dict(spec,registered_panel=panel_pin,cases=[str(folder/'EXECUTION_SPEC.json') for folder,_ in entries],
                  new_physical_requests_budget=len(entries),case_lead_s=1980)
    if teacher_stop is not None:
        parent['teacher_start_index'] = len(prefix)
    return parent, entries
