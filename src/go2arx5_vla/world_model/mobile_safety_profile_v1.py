"""Bind the mobile operating guard to deployed files, never tune the WBC."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np

SCHEMA='MOBILE_ACTUAL_STATE_SAFETY_PROFILE_V1'
TASK_INFO_SHA='0b69bd58db364a01e87331b0e4e777a9684fda18a4f67fa60aee55174f72dcca'
BASE_YAML_SHA='b42c80faa2bb9e910cba0de65e0de9952d9c27643a9184a2475972bc1b642381'
POLICY_YAML_SHA='d39fbb039751f24f7fbae04e4402f803595fc009456e18a10bb7bce992b19855'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block(text,name):
    text=re.sub(r'(?m);.*$|//.*$','',text)
    match=re.search(r'\b'+re.escape(name)+r'\s*\{',text)
    if not match:raise ValueError('MISSING_NATIVE_INFO_BLOCK:'+name)
    start=match.end();depth=1
    for i in range(start,len(text)):
        depth+=(text[i]=='{')-(text[i]=='}')
        if depth==0:return text[start:i]
    raise ValueError('INCOMPLETE_NATIVE_INFO_BLOCK')


def binding(native_root):
    import yaml
    root=Path(native_root)
    paths=dict(task_info=root/'src/ctrl/go2_x5_ocs2/config/task_floating.info',
        base_yaml=root/'src/ctrl/rl_sar/policy/go2_x5/base.yaml',
        policy_yaml=root/'src/ctrl/rl_sar/policy/go2_x5/N1_s29_175632/config.yaml')
    for key,pin in dict(task_info=TASK_INFO_SHA,base_yaml=BASE_YAML_SHA,policy_yaml=POLICY_YAML_SHA).items():
        if sha(paths[key])!=pin:raise ValueError('DEPLOYED_SAFETY_SOURCE_CHANGED:'+key)
    base=yaml.safe_load(paths['base_yaml'].read_text())['go2_x5']
    policy=yaml.safe_load(paths['policy_yaml'].read_text())['go2_x5/N1_s29_175632']
    if base['config_name']!='N1_s29_175632':raise ValueError('SAME_FROZEN_LEG_POLICY_REQUIRED')
    native=block(block(paths['task_info'].read_text(),'basePositionLimits'),'limits')
    bounds={}
    for axis in('z','pitch','roll'):
        item=block(native,axis)
        bounds[axis]=[float(re.search(r'\b'+key+r'\s+([-+.\deE]+)',item)[1])for key in('lower','upper')]
    limits=np.asarray(base['torque_limits'],float)
    names=base['joint_names']
    if len(names)!=len(limits)or len(names)!=18 or np.any(limits<=0):raise ValueError('EXACT_NATIVE18_ACTUATOR_BOUNDS')
    # Native SDK stores these command limits as float; preserve that round trip.
    torque={name:float(np.float32(value))for name,value in zip(names,limits,strict=True)}
    return dict(schema=SCHEMA,sources={k:dict(path=str(v),sha256=sha(v))for k,v in paths.items()},
        mount_position_z=bounds['z'],pitch=bounds['pitch'],roll=bounds['roll'],
        workspace_radius_m=1.,torque_by_joint=torque,
        maximum_unsupported_duration_s=1./float(policy['gait_frequency']),
        gait=dict(frequency_hz=policy['gait_frequency'],duty=policy['gait_duration'],phases=policy['gait_phases']),
        stance_slip='RECORD_CONTACT_TANGENTIAL_SPEED; NATIVE_CONTACT_FRICTION_AND_WARNINGS_RETAINED',
        joint_position_safety='UNCHANGED_OBSERVER_MODEL_LIMITS_WITH_ORIGINAL_TOLERANCES',
        physical_acceleration_protection='UNCHANGED_NATIVE_BRIDGE_OUTPUT_SLEW',
        operating_bounds_are_not_hardware_certification=True,control_parameters_changed=False)


def write_native_binding(path,value):
    lines=['MOBILE_SAFETY_V1',*map(str,[*value['mount_position_z'],*value['pitch'],*value['roll'],
        value['workspace_radius_m'],value['maximum_unsupported_duration_s']]),str(len(value['torque_by_joint']))]
    lines.extend(name+' '+format(limit,'.17g')for name,limit in value['torque_by_joint'].items())
    Path(path).write_text('\n'.join(lines)+'\n')


def require_armed(adapter):
    path=Path(adapter.output)/'MOBILE_SAFETY_ARMED'
    if (not adapter.spec.get('mobile_wbc_boundary_v1')or not path.is_file()
            or not getattr(adapter,'mobile_motion_armed',False)
            or sha(adapter.spec['mobile_safety_library'])!=adapter.spec['mobile_safety_library_sha256']):
        raise PermissionError('ACTUAL_MOBILE_GUARD_MUST_BE_INSTALLED_AND_ARMED')
    event=Path(adapter.output)/'MOBILE_SAFETY_STOP.private.json'
    if event.exists():raise RuntimeError('MOBILE_ACTUAL_STATE_PROTECTION_STOP:'+event.read_text())
