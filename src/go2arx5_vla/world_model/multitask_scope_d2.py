"""Separate PR 411 scope for cold isolated MuJoCo worlds."""

import time
from pathlib import Path
import os

from .multitask_score_v6 import Task
from .multitask_scope_v6 import VERSION, read_pin, validate_frozen_assets
from .storage import digest

PLAN = '2352547871684c7617c2c89754fbf9c70a534abb'
SCOPE = 'SLA_D2_MFREE_NATIVE_MUJOCO'
D3_PLAN = '323551a76e4462436d4ed13f1cbb205156d7ce99'
D3_SCOPE = 'SLA_D3_MFREE_NATIVE_MUJOCO'
D3_TEACHER = 'CONTACT_D3_V1'
PROFILES = ('LEGACY_N1', 'LEGACY_N2')
STAGED_TEACHER = 'STAGED_CALIBRATED_V2'
START = 5703
PROBE_TICKS = 10703


def protocol(spec):
    if spec.get('scope') == D3_SCOPE:
        return D3_SCOPE, D3_PLAN, 'CURRENT_USER_EXPLICIT_PR413_FIXED_D3_PLAN'
    return SCOPE, PLAN, 'CURRENT_USER_EXPLICIT_PR411_FIXED_D2_PLAN'


def bound_slot(spec):
    _, plan, _ = protocol(spec)
    binding = read_pin(spec['d2_world_binding'])
    slots = binding['slots']
    profile = spec['d2_profile']
    if (profile not in PROFILES or len(slots) != (1 if profile=='LEGACY_N1' else 2)
            or binding['plan_commit'] != plan or binding['source_sha'] != spec['source_sha']
            or binding['campaign_root'] != spec['campaign_root']
            or binding['deadline_epoch'] != spec['deadline_epoch']):
        raise PermissionError('D2_PINNED_CAMPAIGN_WORLD_BINDING')
    occupied = set(binding['reserved_cpus'])
    roots, domains = set(), set()
    for world in slots:
        root = Path(world['root'])
        domain = world['ROS_DOMAIN_ID']
        cpus = set(world['cpus'])
        lock = root/'world.lock'
        if (root.resolve(strict=True) != root or root.parent != Path(spec['campaign_root'])/'worlds'
                or root in roots or domain in domains or domain not in range(192,200)
                or not cpus or len(cpus) != len(world['cpus']) or cpus & occupied
                or world['mobile_world_lock'] != str(lock)):
            raise PermissionError('D2_DISTINCT_WORLD_ROOTS_DOMAINS_CPUS_LOCKS')
        roots.add(root)
        domains.add(domain)
        occupied.update(cpus)
    matches = [s for s in slots if s['id']==spec['d2_world_id']]
    if len(matches) != 1:
        raise PermissionError('D2_EXACT_WORLD_SLOT')
    world, = matches
    for key in ('mobile_world_lock','ROS_DOMAIN_ID','cpus'):
        if spec[key] != world[key]:
            raise PermissionError('D2_WORLD_SLOT_SPEC_CHANGED')
    return world


def validate(spec, *, before_spawn=False):
    scope, plan, authority_source = protocol(spec)
    env = spec['native_environment']
    role = spec['sim_only_role']
    task = Task.parse(spec['task_instruction_id'])
    probe = role == 'INTERFACE'
    condition = spec['mobile_condition']
    cap = PROBE_TICKS if probe else 35703 if task.operation=='HOLD' else 50703
    stage = spec['d2_stage']
    world = bound_slot(spec)
    case = Path(spec['registered_root'])
    scientific_stages = ('R1', 'R2', 'Q') if scope == D3_SCOPE else ('D', 'Q')
    if (spec['scope'] != scope or spec['plan_commit'] != plan
            or spec['mobile_experiment_version'] != VERSION or spec['native_backend'] != 'MUJOCO_ONLY'
            or spec['execution_clock_profile'] != 'SIM_PAUSED_NATIVE'
            or role not in ('INTERFACE','EXPERT') or spec['physics'] is not True
            or (probe and stage!='E') or (not probe and stage not in scientific_stages)
            or spec['hardware_connections_allowed'] is not False
            or spec['WBC_changes'] is not False or spec['safety_changes'] is not False
            or spec['physics_dt_s'] != .002 or spec['max_ticks'] != cap
            or spec['policy_queries'] != (cap-START)//250
            or spec['new_physical_requests_budget'] != 1 or spec['new_model_generations_budget']
            or spec['new_optimizer_updates_budget'] or spec['new_training_visits_budget']
            or condition['mode'] != 'M_FREE' or condition['family'] not in ('F1','F2')
            or condition['method'] != ('SIM_INTERFACE_HOLD' if probe else 'EXPERT')
            or env.get('CUDA_VISIBLE_DEVICES') != '' or env.get('ROS_LOCALHOST_ONLY') != '1'
            or env.get('ROS_AUTOMATIC_DISCOVERY_RANGE') != 'LOCALHOST'
            or env.get('ROS_DOMAIN_ID') != str(world['ROS_DOMAIN_ID'])
            or any(Path(spec[k]) != case/k for k in ('tmp','logs','cache','data'))
            or any(Path(env[k]) != case/child for k,child in
                   (('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('ROS_LOG_DIR','logs/ros')))
            or case.parent != Path(spec['campaign_root'])/'cases'
            or spec['mobile_group'] != spec['campaign_root']
            or (not before_spawn and set(spec['cpus']) != os.sched_getaffinity(0))):
        raise PermissionError('D2_FROZEN_MOBILE_CONTRACT_OR_WORLD_IDENTITY')
    authority = read_pin(spec['sim_only_project_authorization'])
    now = time.time()
    if (authority.get('source') != authority_source
            or authority['plan_commit'] != plan or authority['source_sha'] != spec['source_sha']
            or authority['hardware_allowed'] is not False
            or not now < spec['deadline_epoch'] <= min(authority['deadline_epoch'],
                   spec['allocation_deadline_epoch'], spec['campaign_deadline_epoch'])):
        raise PermissionError('D2_ACTUAL_FINITE_SIMULATION_AUTHORITY')
    validate_frozen_assets(spec)
    item = next(r for r in read_pin(spec['registered_panel']) if r['id']==spec['panel_entry_id'])
    if (item['role'] != role or item['instruction'] != task.instruction
            or item['condition'] != condition or item['geometry'] != spec['multitask_geometry']
            or item['scene'] != dict(path=spec['model_path'], sha256=spec['frozen_files'][spec['model_path']])
            or item['world'] != world['id'] or item['stage'] != stage
            or (probe and (item['motion'] != spec['d2_probe_motion']
                           or item['motion'] not in ('STABLE','SMALL_EEF')))):
        raise PermissionError('D2_EXACT_PREREGISTERED_CASE')
    local_stop=spec.get('d2_stop_at_task_tick')
    stop_pair=spec.get('d2_stop_pair')
    if local_stop != item.get('stop_at_task_tick') or (local_stop is not None and (
            not probe or spec['d2_profile']!='LEGACY_N2' or local_stop!=1500)):
        raise PermissionError('D2_EXACT_PREREGISTERED_ENGINEERING_STOP')
    if (stop_pair != item.get('stop_pair') or (stop_pair is not None and (
            not probe or spec['d2_profile']!='LEGACY_N2' or stop_pair not in (1,2)))):
        raise PermissionError('D2_EXACT_PREREGISTERED_STOP_PAIR_BATCH')
    candidate = spec.get('d2_teacher_candidate')
    candidates = (None, D3_TEACHER) if scope == D3_SCOPE else (None, STAGED_TEACHER)
    if (candidate != item.get('d2_teacher_candidate')
            or candidate not in candidates
            or (probe and candidate is not None)):
        raise PermissionError('D2_EXACT_PREREGISTERED_TEACHER_CANDIDATE')
    if scope == D3_SCOPE:
        from .multitask_contact_teacher_d3 import MECHANISMS

        if (not probe and candidate != D3_TEACHER
                or (stage == 'R1' and spec['d2_profile'] != 'LEGACY_N1')
                or spec.get('d3_ablation') != item.get('d3_ablation')
                or spec.get('d3_ablation') not in (None, *MECHANISMS)
                or (stage != 'R1' and spec.get('d3_ablation') is not None)
                or spec.get('d3_stop_barrier') != item.get('d3_stop_barrier')
                or bool(spec.get('d3_stop_barrier')) != bool(stop_pair)):
            raise PermissionError('D3_EXACT_PREREGISTERED_ARM_AND_ENGINEERING_BARRIER')
        barrier = spec.get('d3_stop_barrier')
        if barrier:
            expected_stop = 'W1' if stop_pair == 1 else 'W2'
            if (barrier != dict(stop_world=expected_stop, maximum_wait_s=90)
                    or (local_stop == 1500) != (world['id'] == expected_stop)):
                raise PermissionError('D3_STOP_PAIR_WORLD_AND_BOUNDED_WAIT')
    if not probe:
        runtime = spec['d2_g7_runtime']
        if (runtime['environment']['CUDA_VISIBLE_DEVICES'] != ''
                or not Path(runtime['python']).is_absolute()
                or digest(runtime['python']) != runtime['python_sha256']):
            raise PermissionError('D2_EXACT_INDEPENDENT_SLA_CPU_RUNTIME_REQUIRED')
    return task


def episode(adapter):
    if adapter.spec['sim_only_role'] == 'INTERFACE':
        from .multitask_probe_d2 import episode as probe
        return probe(adapter)
    from .multitask_teacher_d2 import PlacementTeacher
    from .multitask_teacher_episode_v6 import episode as teacher
    if adapter.spec.get('d2_teacher_candidate') == D3_TEACHER:
        from functools import partial
        from .multitask_contact_teacher_d3 import ContactTeacher

        return teacher(adapter, placement_teacher=partial(ContactTeacher, ablated=adapter.spec.get('d3_ablation')))
    if adapter.spec.get('d2_teacher_candidate') == STAGED_TEACHER:
        from .multitask_staged_placement_d2 import StagedPlacementTeacher
        return teacher(adapter, placement_teacher=StagedPlacementTeacher)
    return teacher(adapter, placement_teacher=PlacementTeacher)
