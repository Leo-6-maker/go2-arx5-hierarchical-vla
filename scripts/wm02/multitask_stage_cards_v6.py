"""Prepare finite V6 M cards over the sealed scenes and existing interfaces."""
import copy
import hashlib
import json
from pathlib import Path
import time

from scripts.pi0292.durable import write
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs, unit_state
from go2arx5_vla.world_model.multitask_scope_v6 import PLAN, SCOPE, read_pin, validate
from go2arx5_vla.world_model import multisource_d4_native as d4
from go2arx5_vla.world_model import sla_followup_native_v1 as followup
from go2arx5_vla.world_model import scale20k_native as scale
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
    method = 'EXPERT'
    if role == 'STUDENT':
        from go2arx5_vla.sla_adapter.multitask_contract_v6 import NATIVE_METHOD
        method = NATIVE_METHOD
    clusters = {item['id']:item for item in population['clusters']}
    panel = []
    for row, arm in planned:
        cluster = clusters[row['cluster']]
        family = row['family']
        if cluster['family'] != family or cluster['reset'] != row['reset']:
            raise ValueError('SEALED_M_CLUSTER_AND_CONDITION_IDENTITY_REQUIRED')
        cid = row['id'] if role == 'EXPERT' else f"{row['id']}-{arm}"
        condition = dict(templates[family]['mobile_condition'],mode='M_FREE',
            method=method,family=family,layout=row['cluster'],
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
        scope=spec['scope'],sim_only_role=item['role'],registered_panel=panel_pin,panel_entry_id=item['id'],
        task_instruction_id=task.instruction,mobile_condition=item['condition'],multitask_geometry=item['geometry'],
        language=language_for(task.instruction),wall_limit_s=1800,max_ticks=cap,policy_queries=(cap-5703)//250,
        new_physical_requests_budget=1,model_service_timeout_s=30,expert_training_eligible=expert,
        model_path=item['scene']['path'],unattended_control=str(root/'control'))
    case['collection_contract'] = ('SLA_V6_MULTITASK_TEACHER_CURRENT_FULL_H40_AND_CONSUMED_MASK' if expert
                                   else 'SLA_V6_MULTITASK_G7_STUDENT_EVALUATION_ONLY')
    case['stage_identity_sha256'] = hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
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
        if spec['scope'] in (d4.SCOPE, followup.SCOPE):
            case['d4_endpoint'] = spec['endpoints'][arm]
        if spec['scope'] == followup.SCOPE:
            case['core_RGB_restore'] = spec['core_RGB_restores'].get(arm)
            case['source_RGB_restore'] = spec.get('source_RGB_restores', {}).get(arm)
        if spec['scope'] == scale.SCOPE:
            case['scale_offline_endpoint'] = spec['endpoints'][arm]
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


def bind_gpu(spec, child, folder):
    """Bind a new GPU child to the existing site's owner and finite allocation."""
    template = read_pin(spec['gpu_template'])
    base = read_pin(template['site_binding'])
    if (base['allocation_deadline_epoch']!=spec['allocation_deadline_epoch'] or base['uid']!=spec['uid']
            or base['host']!=spec['host'] or base['allocation']['neural_owner_lock']!=template['neural_owner_lock']
            or template['neural_owner_lock']!=spec['neural_owner_lock']):
        raise PermissionError('ACTUAL_EXISTING_SITE_AND_GPU_OWNER_BINDING_REQUIRED')
    for key in ('python','parent_manifest','parent_params','tokenizer','initial_function_rule','upstream_sha','GPU_uuid',
                'JAX_memory_fraction','allocator_limit_bytes','minimum_host_available_bytes','RPC_timeout_s',
                'model_service_timeout_s','save_and_exit_on_limit','frame_cache_bytes'):
        child[key] = template[key]
    child.update(parent_host_cast_v7=True,max_gpu_hours=12,max_walltime_s=43200,compilation_cache_dir=str(folder/'cache'))
    env = {key:value for key,value in template['native_environment'].items() if key!='PYTHONPATH'}
    child['native_environment'].update(env,TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),
                                       JAX_COMPILATION_CACHE_DIR=str(folder/'cache'))
    if spec['scope'] in (d4.SCOPE, followup.SCOPE, scale.SCOPE):
        child['native_environment']['PYTHONPATH'] = ':'.join((
            spec['source_root'], spec['source_root']+'/src', template['native_environment']['PYTHONPATH']))
    base['allocation']['campaign_root'] = spec['registered_root']
    base['paths'] = {key:str(Path(child[key]).resolve(strict=True)) for key in ('parent_params','dataset','compilation_cache_dir')}
    write(folder/'SITE_BINDING.private.json',base)
    child['site_binding'] = pin(folder/'SITE_BINDING.private.json')


def student_worker(spec, panel):
    """Prepare a nonintegrating G7 worker for exactly this new student panel."""
    from go2arx5_vla.sla_adapter.multitask_contract_v6 import NATIVE_METHOD

    root = Path(spec['registered_root'])
    folder = root/'worker'
    dataset = read_pin(spec['multitask_dataset'])
    is_scale = spec['scope'] == scale.SCOPE
    staged = spec['scope'] in (d4.SCOPE, followup.SCOPE, scale.SCOPE)
    if dataset['state'] != 'committed' or (
            dataset['split'] != 'TRAIN' if staged
            else dataset['compiled_population_sha256'] != spec['compiled_population']['sha256']):
        raise ValueError('M_WORKER_REQUIRES_COMMITTED_MATCHED_TARGET_POPULATION')
    manifests = {arm:read_pin(dict(path=str(Path(package['path'])/'MANIFEST.private.json'),sha256=package['sha256']))
                 for arm, package in spec['packages'].items()}
    stats = dataset['statistics']['sha256']
    for arm, manifest in manifests.items():
        if is_scale:
            matches = (manifest['target_data_manifest_sha256'] == spec['multitask_dataset']['sha256']
                       and (manifest['arm'],manifest['step']) == scale.ENDPOINTS[arm])
        else:
            matches = (manifest['bindings']['dataset_manifest_sha256'] == spec['multitask_dataset']['sha256']
                       and manifest['bindings']['arm'] == arm)
        if not matches or manifest['statistics_sha256'] != stats:
            raise ValueError('M_WORKER_REQUIRES_SHARED_SEALED_TARGET_DATA_AND_STATISTICS')
    requests = sum(120 if Task.parse(row['instruction']).operation=='HOLD' else 180 for row in panel)
    fixed = spec['new_model_generations_budget'] if staged else 34560
    if not 0 < requests <= fixed or (staged and requests != fixed):
        raise ValueError('M_WORKER_REQUIRES_FINITE_REGISTERED_STUDENT_QUERY_CAP')
    dirs(folder)
    ipc = folder/'ipc'
    ipc.mkdir()
    child = dict(spec,registered_root=str(folder),campaign_root=str(root),role='SLA_V6_M_STUDENT_G7_WORKER',
        physics=False,new_physical_requests_budget=0,new_optimizer_updates_budget=0,new_training_visits_budget=0,
        ipc_root=str(ipc),live_root=str(folder),selected_combo=NATIVE_METHOD,archive_all_branches=False,
        dataset=str(Path(spec['multitask_dataset']['path']).parent),
        new_complete_generations_budget=requests,new_model_generations_budget=requests,unified_requests_budget=requests)
    child['native_environment'] = dict(spec['native_environment'])
    child['request_bindings_by_arm'] = {arm:dict(arm=arm,policy_package_sha256=spec['packages'][arm]['sha256'],
        statistics_sha256=manifest['statistics_sha256'],codec=NATIVE_METHOD,source_sha=spec['source_sha'],
        plan_commit=spec['plan_commit'],compiled_population_sha256=spec['compiled_population']['sha256'])
        for arm, manifest in manifests.items()}
    bind_gpu(spec,child,folder)
    if staged:
        parent = child['parent_manifest']
        if parent['sha256'] != d4.PARENT_SHA or digest(parent['path']) != d4.PARENT_SHA:
            raise ValueError('D4_CORE_PARENT_BYTES_MUST_MATCH_THE_SEALED_VANDA_PARENT')
    elif any(manifest['parent_manifest']!=child['parent_manifest'] for manifest in manifests.values()):
        raise ValueError('M_WORKER_REQUIRES_THE_ACTUAL_SHARED_OFFICIAL_PARENT')
    child['gpu_environment'] = dict(child['native_environment'])
    # The native case preflight reads this immutable file. The existing
    # registration transaction will verify the same bytes and create its lease.
    write(folder/'EXECUTION_SPEC.json',child)
    return folder, child


def prepare_cards(spec, *, teacher_stop=None):
    """Prepare the parent, native case specs and any G7 worker for registration."""
    from .multitask_teacher_campaign_v6 import interface_records

    root = Path(spec['registered_root'])
    now = time.time()
    identity = (spec['scope'], spec['plan_commit'])
    allowed = {(SCOPE, PLAN)} if teacher_stop is not None else {
        (SCOPE, PLAN), (d4.SCOPE, d4.PLAN), (followup.SCOPE, followup.PLAN), (scale.SCOPE, scale.PLAN)}
    if (not now+1980 < spec['deadline_epoch'] <= min(now+43200,spec['campaign_deadline_epoch'],spec['allocation_deadline_epoch']-60)
            or identity not in allowed
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
    if spec['scope'] in (followup.SCOPE, scale.SCOPE):
        interface_records(interface,spec['mobile_world_lock'],identity=(d4.E0_SCOPE,d4.E0_PLAN))
    else:
        interface_records(interface,spec['mobile_world_lock'])
    templates = {case['mobile_condition']['family']:case for case in map(read,interface['cases'])}
    population = read_pin(spec['compiled_population'])
    if teacher_stop is not None:
        prefix = verified_teacher_prefix(spec['previous_teacher_results'],spec['compiled_population'],
                                          world_lock=spec['mobile_world_lock'])
        selected = [(row,None) for row in teacher_slice(population,prefix,teacher_stop)]
        role = 'EXPERT'
    else:
        from .multitask_student_campaign_v6 import endpoint_records, expected_conditions, student_arms

        arms = student_arms(spec)
        endpoint_records(spec)
        selected = expected_conditions(population,spec['evaluation_stage'],arms)
        if spec['scope'] in (d4.SCOPE,followup.SCOPE,scale.SCOPE):
            from .multitask_panel_suffix import budgets, verified_prefix

            prefix = verified_prefix(spec, selected)
            selected = selected[len(prefix):]
            requests, ticks, generations = budgets(selected)
            spec = dict(spec,new_physical_requests_budget=requests,
                        new_model_generations_budget=generations,new_native_tick_budget=ticks)
        role = 'STUDENT'
    panel = panel_cards(population,selected,templates,role=role)
    require_unstarted_cases(spec['registry'],{row['id'] for row in panel})
    if not root.exists():
        dirs(root)
    workers = []
    if role == 'STUDENT':
        folder, worker = student_worker(spec,panel)
        spec = dict(spec,resident_server_spec=pin(folder/'EXECUTION_SPEC.json'))
        workers.append((folder,worker))
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
    return parent, workers+entries


def register_student_panel(spec):
    """Register the finite student stage without starting a process."""
    from go2arx5_vla.world_model.mfree_recovery_v2 import register

    root = Path(spec['registered_root'])
    if root.exists():
        raise PermissionError('NEW_OWNED_CARDS_REQUIRE_UNOPENED_STUDENT_ROOT')
    spec = dict(spec,role='SLA_V6_M_STUDENT_'+spec['evaluation_stage'],registry_scope=spec['scope'])
    parent, entries = prepare_cards(spec)
    register(parent,[(root,parent),*entries])
    return pin(root/'EXECUTION_SPEC.json')
