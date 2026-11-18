"""Separate PR 411 scope for cold isolated MuJoCo worlds."""

import time
from pathlib import Path
import os

from .multitask_score_v6 import Task
from .multitask_scope_v6 import VERSION, read_pin, validate_frozen_assets

PLAN = '2352547871684c7617c2c89754fbf9c70a534abb'
SCOPE = 'SLA_D2_MFREE_NATIVE_MUJOCO'
PROFILES = ('LEGACY_N1', 'LEGACY_N2')
START = 5703
PROBE_TICKS = 10703


def bound_slot(spec):
    binding = read_pin(spec['d2_world_binding'])
    slots = binding['slots']
    profile = spec['d2_profile']
    if (profile not in PROFILES or len(slots) != (1 if profile=='LEGACY_N1' else 2)
            or binding['plan_commit'] != PLAN or binding['source_sha'] != spec['source_sha']
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
    env = spec['native_environment']
    role = spec['sim_only_role']
    task = Task.parse(spec['task_instruction_id'])
    probe = role == 'INTERFACE'
    condition = spec['mobile_condition']
    cap = PROBE_TICKS if probe else 35703 if task.operation=='HOLD' else 50703
    stage = spec['d2_stage']
    world = bound_slot(spec)
    case = Path(spec['registered_root'])
    if (spec['scope'] != SCOPE or spec['plan_commit'] != PLAN
            or spec['mobile_experiment_version'] != VERSION or spec['native_backend'] != 'MUJOCO_ONLY'
            or spec['execution_clock_profile'] != 'SIM_PAUSED_NATIVE'
            or role not in ('INTERFACE','EXPERT') or spec['physics'] is not True
            or (probe and stage!='E') or (not probe and stage not in ('D','Q'))
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
    if (authority.get('source') != 'CURRENT_USER_EXPLICIT_PR411_FIXED_D2_PLAN'
            or authority['plan_commit'] != PLAN or authority['source_sha'] != spec['source_sha']
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
    return task


def episode(adapter):
    if adapter.spec['sim_only_role'] == 'INTERFACE':
        from .multitask_probe_d2 import episode as probe
        return probe(adapter)
    from .multitask_teacher_d2 import PlacementTeacher
    from .multitask_teacher_episode_v6 import episode as teacher
    return teacher(adapter, placement_teacher=PlacementTeacher)
