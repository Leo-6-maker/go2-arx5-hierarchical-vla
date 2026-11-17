"""Explicit V6 two-object MuJoCo interface and Teacher scope."""
import hashlib
import json
import time
from pathlib import Path

from .multitask_score_v6 import Task
from .sim_only_scope_v51 import BINARIES, INTERFACE_TICKS
from .storage import digest

PLAN = 'ae3d3c0dceade437b0104a150cd0402fef650052'
SCOPE = 'SLA_V6_MULTITASK_NATIVE'
VERSION = 'MFREE_MULTITASK_NATIVE_V6'


def read_pin(pin):
    path = Path(pin['path'])
    if digest(path) != pin['sha256']:
        raise PermissionError('PINNED_MULTITASK_INPUT_CHANGED:'+path.name)
    return json.loads(path.read_bytes())


def validate(spec):
    condition=spec['mobile_condition'];role=spec['sim_only_role']
    task=Task.parse(spec['task_instruction_id'])
    interface=role=='INTERFACE'
    cap=INTERFACE_TICKS if interface else 35703 if task.operation=='HOLD' else 50703
    expected_method='SIM_INTERFACE_HOLD' if interface else 'EXPERT'
    env=spec['native_environment']
    if (spec.get('scope')!=SCOPE or spec['plan_commit']!=PLAN
            or spec['mobile_experiment_version']!=VERSION or role not in ('INTERFACE','EXPERT')
            or spec['native_backend']!='MUJOCO_ONLY' or spec['hardware_connections_allowed'] is not False
            or spec['execution_clock_profile']!='SIM_PAUSED_NATIVE' or spec['physics'] is not True
            or spec['WBC_changes'] is not False or spec['safety_changes'] is not False
            or spec['physics_dt_s']!=.002 or spec['max_ticks']!=cap or
            spec['policy_queries']!=(0 if interface else (cap-5703)//250)
            or spec['new_physical_requests_budget']!=1 or spec['new_model_generations_budget']
            or spec['new_optimizer_updates_budget'] or spec['new_training_visits_budget']
            or condition['family'] not in ('F1','F2') or condition['mode']!='M_FREE'
            or condition['method']!=expected_method or env.get('CUDA_VISIBLE_DEVICES')!=''
            or env.get('ROS_LOCALHOST_ONLY')!='1' or env.get('ROS_AUTOMATIC_DISCOVERY_RANGE')!='LOCALHOST'
            or spec['ROS_DOMAIN_ID'] not in range(192,200)):
        raise PermissionError('FIXED_V6_MFREE_MULTITASK_MUJOCO_LIMITS')
    authority=read_pin(spec['sim_only_project_authorization'])
    if (authority.get('source')!='CURRENT_USER_EXPLICIT_FIXED_SLA_V6_PLAN'
            or authority.get('plan_commit')!=PLAN or authority.get('source_sha')!=spec['source_sha']
            or authority.get('hardware_allowed') is not False or time.time()>=spec['deadline_epoch']
            or spec['deadline_epoch']>min(authority['deadline_epoch'],spec['allocation_deadline_epoch'])):
        raise PermissionError('ACTUAL_FINITE_V6_MULTITASK_AUTHORITY_REQUIRED')

    for key,sha in BINARIES.items():
        if spec['frozen_files'].get(spec[key])!=sha:
            raise PermissionError('FROZEN_SOLVER_CONTROLLER_AND_SIMULATOR_REQUIRED')
    observer=read_pin(spec['compiled_observer_result'])
    outer=read_pin(spec['compiled_outer_result'])
    if observer['exit_code'] or outer['exit_code']:
        raise ValueError('SEALED_BINARY_COLD_COMPILATION_REQUIRED')
    for key,d in (('observer_library',observer['observer_candidate']),
                  ('outer_library',outer['outer_candidate'])):
        if spec['frozen_files'].get(spec[key])!=d['library_sha256']:
            raise PermissionError('ONLY_EXACT_COLD_COMPILED_MULTITASK_BINARIES')
    manifest=read_pin(spec['registered_panel'])
    item=next(row for row in manifest if row['id']==spec['panel_entry_id'])
    if (item['instruction']!=task.instruction or item['role']!=role
            or item['condition']!=condition or item['geometry']!=spec['multitask_geometry']
            or item['scene']!=dict(path=spec['model_path'],sha256=spec['frozen_files'][spec['model_path']])):
        raise ValueError('EXACT_MULTITASK_CONDITION_SCENE_AND_INSTRUCTION_REQUIRED')
    return task


def episode(adapter):
    if adapter.spec['sim_only_role']=='EXPERT':
        from .multitask_teacher_episode_v6 import episode as teacher
        return teacher(adapter)
    from .sim_only_mobile_v51 import interface_episode
    from .multitask_camera_v6 import TwoObjectCameraSession
    from .fusion_paused_observation import _pause, _after_pause

    result=interface_episode(adapter)
    before=_pause(adapter)
    session=TwoObjectCameraSession(adapter.spec['model_path'])
    try:
        images,body,_,calibration=session.render_memory(before['packet'])
        _after_pause(adapter,before)
    finally:
        session.close()
    result.update(schema='SLA_V6_MULTITASK_INTERFACE_ACTUAL',
        instruction=adapter.task.instruction,state_ABI=[41,38,20],RGB_step=before['packet']['step'],
        body_rgb_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
        wrist_rgb_sha256=hashlib.sha256(images['wrist_rgb'].tobytes()).hexdigest(),
        RGB_calibration_clock_s=calibration['wrist_rgb']['timestamp_s'])
    return result
