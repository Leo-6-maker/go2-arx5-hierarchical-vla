"""New V6 target identity over the unchanged isolated native consumer."""
from pathlib import Path
from .storage import digest
from .sim_only_scope_v51 import BINARIES,TOTAL_TICKS,INTERFACE_TICKS
from go2arx5_vla.sla_adapter.target_contract_v6 import PLAN,B1,contract
VERSION='MFREE_B1_FULL_H40_SIM_ONLY_V6'
SCOPE='SLA_V6_TARGET_NATIVE'


def read(pin):
    if digest(pin['path'])!=pin['sha256']:raise ValueError('V6_TARGET_PIN_CHANGED')
    return __import__('json').loads(Path(pin['path']).read_bytes())


def validate(spec):
    authority=read(spec['sim_only_project_authorization']);role=spec['sim_only_role'];condition=spec['mobile_condition']
    if(authority.get('schema')!='USER_PROJECT_AUTHORIZED_ISOLATED_MUJOCO_V6_T'
       or authority.get('plan_commit')!=PLAN or authority.get('source_sha')!=spec['source_sha']
       or authority.get('source')!='CURRENT_USER_EXPLICIT_FIXED_SLA_V6_PLAN'
       or authority.get('hardware_allowed')is not False or spec.get('scope')!=SCOPE or spec.get('plan_commit')!=PLAN
       or spec.get('mobile_experiment_version')!=VERSION or spec.get('native_backend')!='MUJOCO_ONLY' or not spec['physics']
       or spec.get('hardware_connections_allowed')is not False or spec.get('execution_clock_profile')!='SIM_PAUSED_NATIVE'
       or spec['physics_dt_s']!=.002 or condition.get('mode')!='M_FREE' or condition.get('family')not in('F1','F2')
       or spec.get('optimizer_updates')!=0 or spec.get('WBC_changes')is not False or spec.get('safety_changes')is not False
       or spec['model_service_timeout_s']!=30 or role not in('INTERFACE','STUDENT')
       or spec['deadline_epoch']>min(authority['deadline_epoch'],1795027883.855488) or spec['new_physical_requests_budget']!=1):
        raise PermissionError('BOUNDED_V6_TARGET_MUJOCO_ONLY_FROZEN_CONTRACT_REQUIRED')
    if role=='STUDENT':
        arm=spec['arm'];packages=spec['packages']
        if(arm not in('T-B1','T-FM','T-H40')or packages['T-B1']['sha256']!=B1
           or spec['SHIFT7_package']!=packages[arm]or spec['max_ticks']!=TOTAL_TICKS or condition['method']!='SHIFT7_SINGLE'):
            raise ValueError('PREREGISTERED_V6_TARGET_ARM_AND_ORIGINAL_H40_INTERFACE_REQUIRED')
        panel=read(spec['registered_panel']);entry=next(r for r in panel if r['id']==spec['panel_entry_id'])
        if any(entry[k]!=condition[k]for k in('family','layout','reset'))or entry['noise_seed']!=condition['seed']or entry['arm']!=arm:
            raise ValueError('SEALED_V6_TARGET_PANEL_CONDITION_REQUIRED')
        if entry['scene']['sha256']!=spec['frozen_files'][spec['model_path']]:raise ValueError('SEALED_TRAIN_SUPPORT_SCENE_REQUIRED')
        package=packages[arm];manifest=read(dict(path=str(Path(package['path'])/'MANIFEST.private.json'),sha256=package['sha256']))
        if arm=='T-B1':
            from go2arx5_vla.sla_adapter.mobile_contract_v52 import contract as original_contract
            original_contract(manifest)
        else:contract(manifest)
        restored=read(spec['ordinary_RGB_restore'])
        if restored['arm']!=arm or restored['package']!=package or not restored['all_bit_exact']or restored['generations']!=16:
            raise ValueError('ACTUAL_TARGET_RGB16_RESTORATION_REQUIRED')
    elif spec['max_ticks']!=INTERFACE_TICKS or condition['method']!='SIM_INTERFACE_HOLD':raise ValueError('FINITE_UNCHANGED_NONCONTACT_INTERFACE_ONLY')
    for key,expected in BINARIES.items():
        if spec['frozen_files'].get(spec[key])!=expected or digest(spec[key])!=expected:raise PermissionError('FROZEN_NATIVE_WBC_CONTROLLER_REQUIRED')
    env=spec['native_environment']
    if(env.get('ROS_LOCALHOST_ONLY')!='1'or env.get('ROS_AUTOMATIC_DISCOVERY_RANGE')!='LOCALHOST'
       or env.get('CUDA_VISIBLE_DEVICES')!=''or spec['ROS_DOMAIN_ID']not in range(192,200)):
        raise PermissionError('ISOLATED_CPU_RENDERING_LOCALHOST_SIMULATION_REQUIRED')
    return dict(scope=SCOPE,plan=PLAN,hardware_allowed=False,original_full_H40_P10_TTL_STOP=True)


def episode(adapter):
    from .sim_only_mobile_v51 import interface_episode,student_episode
    validate(adapter.spec);result=(interface_episode if adapter.spec['sim_only_role']=='INTERFACE'else student_episode)(adapter)
    result.update(schema='SLA_V6_TARGET_ACTUAL_NATIVE_EPISODE',arm=adapter.spec.get('arm'),plan_commit=PLAN)
    return result
