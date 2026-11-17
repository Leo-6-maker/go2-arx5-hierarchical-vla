"""Fresh bounded V5.2 identity, using the existing isolated native consumer."""
import json
from pathlib import Path
from .storage import digest
from .sim_only_scope_v51 import BINARIES,STARTUP_TICKS,TOTAL_TICKS,INTERFACE_TICKS
from go2arx5_vla.sla_adapter.mobile_contract_v52 import PLAN,PARENT,contract

VERSION='MFREE_SHIFT7_SIM_ONLY_V52'


def read(pin):
    if digest(pin['path'])!=pin['sha256']:raise ValueError('V52_PIN_CHANGED')
    return json.loads(Path(pin['path']).read_bytes())


def query_identity(spec):
    condition=spec['mobile_condition'];arm=spec['arm']
    if (spec.get('mobile_experiment_version')!=VERSION or spec.get('scope')!='SLA_MOBILE_F2_V52'
            or spec.get('plan_commit')!=PLAN or spec.get('execution_clock_profile')!='SIM_PAUSED_NATIVE'
            or condition.get('method')!='SHIFT7_SINGLE' or condition.get('mode')!='M_FREE'
            or condition.get('family')not in('F1','F2') or arm not in('B0','B1','B2')
            or spec['max_ticks']!=TOTAL_TICKS or spec['physics_dt_s']!=.002
            or spec['SHIFT7_package']!=spec['packages'][arm]
            or spec['packages']['B0']['sha256']!=PARENT):
        raise ValueError('EXPLICIT_PREREGISTERED_V52_SINGLE_MODEL_IDENTITY_REQUIRED')


def validate(spec):
    authority=read(spec['sim_only_project_authorization']);role=spec['sim_only_role']
    if (authority.get('schema')!='USER_PROJECT_AUTHORIZED_ISOLATED_MUJOCO_V52'
            or authority.get('plan_commit')!=PLAN or authority.get('source_sha')!=spec['source_sha']
            or authority.get('source')!='CURRENT_USER_EXPLICIT_PR404_V52_INSTRUCTION'
            or authority.get('hardware_allowed')is not False
            or spec.get('scope')!='SLA_MOBILE_F2_V52' or spec.get('plan_commit')!=PLAN
            or spec.get('native_backend')!='MUJOCO_ONLY' or not spec['physics']
            or spec.get('hardware_connections_allowed')is not False
            or spec.get('execution_clock_profile')!='SIM_PAUSED_NATIVE' or spec['physics_dt_s']!=.002
            or spec['mobile_condition'].get('mode')!='M_FREE' or spec['mobile_condition'].get('family')not in('F1','F2')
            or spec.get('optimizer_updates')!=0 or spec.get('WBC_changes')is not False
            or spec.get('safety_changes')is not False or spec['model_service_timeout_s']!=30
            or role not in('INTERFACE','STUDENT')
            or spec['deadline_epoch']>min(authority['deadline_epoch'],1795027883.855488)
            or spec['new_physical_requests_budget']!=1):
        raise PermissionError('BOUNDED_PROJECT_AUTHORIZED_V52_MUJOCO_ONLY')
    if role=='STUDENT':
        query_identity(spec)
        panel=read(spec['registered_panel']);entry=next(r for r in panel if r['id']==spec['panel_entry_id'])
        condition=spec['mobile_condition']
        if any(entry[k]!=condition[k]for k in('family','layout','reset'))or entry['noise_seed']!=condition['seed']:
            raise ValueError('EXACT_FROZEN_PANEL_CONDITION_REQUIRED')
        if entry.get('arm',spec['arm'])!=spec['arm']or entry['scene']['sha256']!=spec['frozen_files'][spec['model_path']]:
            raise ValueError('FROZEN_PANEL_ARM_AND_SCENE_REQUIRED')
        package=spec['SHIFT7_package'];manifest=read(dict(path=str(Path(package['path'])/'MANIFEST.private.json'),sha256=package['sha256']))
        if spec['arm']=='B0':
            if package['sha256']!=PARENT:raise ValueError('EXACT_FROZEN_B0_PACKAGE_BYTES_REQUIRED')
        else:
            contract(manifest)
        restoration=read(spec['ordinary_RGB_restore'])
        if restoration['arm']!=spec['arm']or restoration['package']!=package or not restoration['all_bit_exact']or restoration['generations']!=16:
            raise ValueError('ACTUAL_ARM_RGB16_RESTORATION_REQUIRED')
    elif (spec.get('mobile_experiment_version')!=VERSION or spec['max_ticks']!=INTERFACE_TICKS
            or spec['mobile_condition']['method']!='SIM_INTERFACE_HOLD'):
        raise ValueError('FINITE_UNCHANGED_NONCONTACT_INTERFACE_ONLY')
    for key,expected in BINARIES.items():
        if spec['frozen_files'].get(spec[key])!=expected or digest(spec[key])!=expected:
            raise PermissionError('FROZEN_NATIVE_SOLVER_CONTROLLER_REQUIRED')
    env=spec['native_environment']
    if (env.get('ROS_LOCALHOST_ONLY')!='1' or env.get('ROS_AUTOMATIC_DISCOVERY_RANGE')!='LOCALHOST'
            or env.get('CUDA_VISIBLE_DEVICES')!='' or spec['ROS_DOMAIN_ID']not in range(192,200)):
        raise PermissionError('CPU_RENDERING_LOCALHOST_SIMULATION_ONLY')
    return dict(scope='SIM_ONLY_PROJECT_AUTHORIZED',plan=PLAN,independent_signature_required=False,hardware_allowed=False)


def episode(adapter):
    from .sim_only_mobile_v51 import interface_episode,student_episode
    validate(adapter.spec)
    result=(interface_episode if adapter.spec['sim_only_role']=='INTERFACE'else student_episode)(adapter)
    result.update(schema='SLA_MOBILE_V52_ACTUAL_NATIVE_EPISODE',arm=adapter.spec.get('arm'),plan_commit=PLAN)
    return result
