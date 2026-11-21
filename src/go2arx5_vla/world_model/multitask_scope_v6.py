"""Explicit V6 two-object MuJoCo interface and Teacher scope."""
import hashlib
import json
import time
from pathlib import Path

from go2arx5_vla.sla_adapter.multitask_contract_v6 import ARMS, NATIVE_METHOD, NATIVE_VERSION, contract
from .multitask_score_v6 import Task
from . import multisource_d4_native as d4
from . import sla_followup_native_v1 as followup
from . import scale20k_native as scale
from . import target_adaptation_teacher as adaptation
from .sim_only_scope_v51 import BINARIES, INTERFACE_TICKS
from .storage import digest

PLAN = 'ae3d3c0dceade437b0104a150cd0402fef650052'
SCOPE = 'SLA_V6_MULTITASK_NATIVE'
VERSION = NATIVE_VERSION


def scope_identity(spec):
    if spec.get('scope') == 'SLA_CORE_HOPPER_V1_STUDENT':
        from scripts.sla import core_hopper_native_scope as student
        return student.SCOPE, student.PLAN, student.ENDPOINTS, student.AUTHORITY
    if spec.get('scope') == 'SLA_CORE_HOPPER_V1_EXPERT':
        from scripts.sla.core_hopper_expert import SCOPE as ch_scope, PLAN as ch_plan, AUTHORITY
        return ch_scope, ch_plan, (), AUTHORITY
    if spec.get('scope') == d4.E0_SCOPE:
        return d4.E0_SCOPE, d4.E0_PLAN, (), d4.AUTHORITY_SOURCE
    if spec.get('scope') == d4.SCOPE:
        return d4.SCOPE, d4.PLAN, d4.ARMS, d4.AUTHORITY_SOURCE
    if spec.get('scope') == followup.SCOPE:
        return followup.SCOPE, followup.PLAN, followup.ARMS, followup.AUTHORITY_SOURCE
    if spec.get('scope') == scale.SCOPE:
        return scale.SCOPE, scale.PLAN, scale.ENDPOINTS, scale.AUTHORITY_SOURCE
    if spec.get('scope') in adaptation.SCOPES:
        plan, authority, _ = adaptation.SCOPES[spec['scope']]
        return spec['scope'], plan, (), authority
    return SCOPE, PLAN, ARMS, 'CURRENT_USER_EXPLICIT_FIXED_SLA_V6_PLAN'


def read_pin(pin):
    path = Path(pin['path'])
    if digest(path) != pin['sha256']:
        raise PermissionError('PINNED_MULTITASK_INPUT_CHANGED:'+path.name)
    return json.loads(path.read_bytes())


def query_identity(spec):
    condition = spec['mobile_condition']
    task = Task.parse(spec['task_instruction_id'])
    cap = 35703 if task.operation == 'HOLD' else 50703
    arm = spec['arm']
    scope, plan, arms, _ = scope_identity(spec)
    if (spec.get('scope') != scope or spec.get('plan_commit') != plan
            or spec.get('mobile_experiment_version') != VERSION or spec['sim_only_role'] != 'STUDENT'
            or spec['execution_clock_profile'] != 'SIM_PAUSED_NATIVE' or spec['physics_dt_s'] != .002
            or condition['method'] != NATIVE_METHOD or condition['mode'] != 'M_FREE'
            or condition['family'] not in ('F1','F2') or spec['max_ticks'] != cap
            or spec['policy_queries'] != (cap-5703)//250 or arm not in arms
            or spec['G7_package'] != spec['packages'][arm]):
        raise ValueError('EXACT_V6_STUDENT_ARM_AND_NATIVE_SCOPE_REQUIRED')


def student_package(spec):
    arm, package = spec['arm'], spec['G7_package']
    scope, plan, _, _ = scope_identity(spec)
    if scope == 'SLA_CORE_HOPPER_V1_STUDENT':
        from scripts.sla.core_hopper_native_scope import validate_case
        validate_case(spec)
    elif scope == d4.SCOPE:
        d4.endpoint(arm, package, spec['d4_endpoint'], spec['ordinary_RGB_restore'])
    elif scope == followup.SCOPE:
        followup.endpoint(arm, package, spec['d4_endpoint'], spec['ordinary_RGB_restore'],
                          core_replay=spec.get('core_RGB_restore'), source_restore=spec.get('source_RGB_restore'))
    elif scope == scale.SCOPE:
        scale.endpoint(arm, package, spec['scale_offline_endpoint'], spec['ordinary_RGB_restore'])
    else:
        manifest = read_pin(dict(path=str(Path(package['path'])/'MANIFEST.private.json'),sha256=package['sha256']))
        contract(manifest)
        restored = read_pin(spec['ordinary_RGB_restore'])
        if (manifest['bindings']['arm'] != arm or restored['package'] != package or restored['exit_code']
                or not restored['passed'] or restored['actual_generations'] != 16 or restored['new_optimizer_updates']
                or restored['new_native_requests'] or len(restored['comparisons']) != 16
                or not all(row['normalized_exact'] and row['EEF10_exact'] for row in restored['comparisons'])):
            raise ValueError('INDEPENDENT_ACTUAL_G7_RGB16_RESTORE_REQUIRED')
    server = read_pin(spec['resident_server_spec'])
    if (server['scope'] != scope or server['plan_commit'] != plan or server['physics']
            or server['source_sha'] != spec['source_sha'] or server['selected_combo'] != NATIVE_METHOD
            or server['packages'] != spec['packages'] or server['campaign_root'] != spec['mobile_group']
            or server['request_bindings_by_arm'][arm]['policy_package_sha256'] != package['sha256']):
        raise ValueError('EXACT_M_G7_STUDENT_SERVER_AND_ARM_BINDING_REQUIRED')


def validate(spec):
    condition=spec['mobile_condition'];role=spec['sim_only_role']
    task=Task.parse(spec['task_instruction_id'])
    interface=role=='INTERFACE'
    student=role=='STUDENT'
    cap=INTERFACE_TICKS if interface else 35703 if task.operation=='HOLD' else 50703
    expected_method='SIM_INTERFACE_HOLD' if interface else NATIVE_METHOD if student else 'EXPERT'
    env=spec['native_environment']
    scope, plan, _, authority_source = scope_identity(spec)
    if (spec.get('scope')!=scope or spec['plan_commit']!=plan
            or spec['mobile_experiment_version']!=VERSION or role not in ('INTERFACE','EXPERT','STUDENT')
            or (scope in (d4.SCOPE, followup.SCOPE, scale.SCOPE, 'SLA_CORE_HOPPER_V1_STUDENT') and not student) or (scope == d4.E0_SCOPE and not interface)
            or (scope in adaptation.SCOPES and (role != 'EXPERT'
                or (condition['family'],task.instruction) not in adaptation.batch_cells(scope)))
            or spec['native_backend']!='MUJOCO_ONLY' or spec['hardware_connections_allowed'] is not False
            or spec['execution_clock_profile']!='SIM_PAUSED_NATIVE' or spec['physics'] is not True
            or spec['WBC_changes'] is not False or spec['safety_changes'] is not False
            or spec['physics_dt_s']!=.002 or spec['max_ticks']!=cap or
            spec['policy_queries']!=(0 if interface else (cap-5703)//250)
            or spec['new_physical_requests_budget']!=1
            or spec['new_model_generations_budget']!=(spec['policy_queries'] if student else 0)
            or spec['new_optimizer_updates_budget'] or spec['new_training_visits_budget']
            or condition['family'] not in ('F1','F2') or condition['mode']!='M_FREE'
            or condition['method']!=expected_method or env.get('CUDA_VISIBLE_DEVICES')!=''
            or env.get('ROS_LOCALHOST_ONLY')!='1' or env.get('ROS_AUTOMATIC_DISCOVERY_RANGE')!='LOCALHOST'
            or spec['ROS_DOMAIN_ID'] not in range(192,200)):
        raise PermissionError('FIXED_V6_MFREE_MULTITASK_MUJOCO_LIMITS')
    authority=read_pin(spec['sim_only_project_authorization'])
    if (authority.get('source')!=authority_source
            or authority.get('plan_commit')!=plan or authority.get('source_sha')!=spec['source_sha']
            or authority.get('hardware_allowed') is not False or time.time()>=spec['deadline_epoch']
            or spec['deadline_epoch']>min(authority['deadline_epoch'],spec['allocation_deadline_epoch'])):
        raise PermissionError('ACTUAL_FINITE_V6_MULTITASK_AUTHORITY_REQUIRED')
    if scope == 'SLA_CORE_HOPPER_V1_EXPERT':
        from scripts.sla.core_hopper_expert import validate_member
        validate_member(spec, authority)
    if scope in adaptation.SCOPES and (authority['native_request_cap'] != len(adaptation.batch_cells(scope))
            or authority['native_tick_cap'] != adaptation.tick_cap(scope)):
        raise PermissionError('FIXED_FIRST6_TRAIN_ADAPTATION_BUDGET_REQUIRED')

    validate_frozen_assets(spec)
    manifest=read_pin(spec['registered_panel'])
    item=next(row for row in manifest if row['id']==spec['panel_entry_id'])
    if (item['instruction']!=task.instruction or item['role']!=role
            or item['condition']!=condition or item['geometry']!=spec['multitask_geometry']
            or item['scene']!=dict(path=spec['model_path'],sha256=spec['frozen_files'][spec['model_path']])):
        raise ValueError('EXACT_MULTITASK_CONDITION_SCENE_AND_INSTRUCTION_REQUIRED')
    if student:
        query_identity(spec)
        if item['arm'] != spec['arm'] or spec['model_service_timeout_s'] != 30:
            raise ValueError('EXACT_PREREGISTERED_G7_ARM_AND_FINITE_SIM_PAUSE_REQUIRED')
        student_package(spec)
    return task


def validate_frozen_assets(spec):
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

def episode(adapter):
    if adapter.spec['sim_only_role']=='EXPERT':
        from .multitask_teacher_episode_v6 import episode as teacher
        if adapter.spec.get('scope')=='SLA_CORE_HOPPER_V1_EXPERT' and adapter.spec.get('carry_repair'):
            repair=adapter.spec['carry_repair']
            if repair.get('kind'):
                from scripts.sla.core_hopper_red_approach import KIND, RedHoldPitch
                if repair['kind']!=KIND:
                    raise ValueError('Unregistered expert repair')
                return teacher(adapter,placement_teacher=RedHoldPitch)
            from scripts.sla.core_hopper_blue_carry import BlueRightCarry
            return teacher(adapter,placement_teacher=BlueRightCarry)
        return teacher(adapter)
    if adapter.spec['sim_only_role']=='STUDENT':
        from .multitask_student_episode_v6 import episode as student
        return student(adapter)
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
