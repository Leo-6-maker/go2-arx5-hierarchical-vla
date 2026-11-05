"""NIGHT1's immutable finite episode contract under the existing AUTO1 owner."""
from pathlib import Path
import hashlib,json,math,time
from .storage import digest

EPISODE_SCHEMA='WM02_NIGHT1_FINITE_EPISODE_V1'
DECISION_SCHEMA='WM02_NIGHT1_LATE_DECISION_EPISODE_V3'
DECISION_RULE='HOLD8_THEN_DECIDE_RAMP8_FIXED_GOAL_V3'
PLAN_SHA='6e99a72e56eaa1066e567fb66f95ad7121695742'
DURATIONS=(24,48,96,160)


def episode_config(spec):
    path=Path(spec['episode_manifest']);value=json.loads(path.read_text())
    decision=value.get('schema')==DECISION_SCHEMA
    if (digest(path)!=spec['episode_manifest_sha256']or value.get('schema')not in(EPISODE_SCHEMA,DECISION_SCHEMA)
            or value['run']!=spec['registered_root']or value['phase']!='collect-target-night1'
            or value['split']not in('TRAIN','DEV')or type(value['task_atoms'])is not int
            or value['task_atoms']not in DURATIONS
            or value['purpose']not in(('DECISION_SINGLE','DECISION_PRODUCTION')if decision else('INITIAL_SINGLE','DURATION_SINGLE','DURATION_DUAL','SCENE_SINGLE','PRODUCTION'))
            or value['reference_rule']!=(DECISION_RULE if decision else'PREDECLARED_RAMP8_THEN_FIXED_GOAL_V1')
            or(decision and value['task_atoms']!=24)
            or len(value['offset_xyz_m'])!=3
            or not all(math.isfinite(x)for x in value['offset_xyz_m'])
            or sum(abs(x)>1e-10 for x in value['offset_xyz_m'])>1
            or any(abs(x)>.0010000001 for x in value['offset_xyz_m'])
            or abs(value['offset_xyz_m'][2])>1e-10):raise PermissionError('PINNED_FINITE_NIGHT1_EPISODE_REQUIRED')
    assets=json.loads(Path(spec['scene_allowlist']).read_text())
    if digest(spec['scene_allowlist'])!=spec['scene_allowlist_sha256']:raise PermissionError('SCENE_ALLOWLIST_CHANGED')
    matches=[x for x in assets['scenes']if x['scene_id']==value['scene_id']]
    if len(matches)!=1:raise PermissionError('EXACT_ALLOWLISTED_SCENE_REQUIRED')
    scene=matches[0]
    if (scene['model_path']!=spec['model_path']or scene['model_sha256']!=digest(spec['model_path'])
            or scene['parent']!=value['parent']or scene['split']!=value['split']
            or scene['robot_fingerprint']!=assets['robot_fingerprint']
            or scene['static_check']!='PASSED_NO_INTEGRATION'
            or assets['native_binary_sha256']!=digest(spec['native_binary'])):raise PermissionError('STATIC_SCENE_AND_FROZEN_ROBOT_IDENTITY_REQUIRED')
    return value


def episode_limit(spec,previous):
    e=episode_config(spec);atoms=e['task_atoms']
    if (not previous or previous['status']!='NATIVE_STAGE_PASSED'or not previous['shutdown_clean']
            or not previous.get('owned_process_exits')or any(p['exit_code']!=0 for p in previous['owned_process_exits'])):raise PermissionError('ACTUAL_NATIVE_PREDECESSOR_PASS_REQUIRED')
    prior_atoms=(previous['actual_target_integrations']-2503)//25
    if e['purpose'].startswith('DECISION_'):
        if prior_atoms<24:raise PermissionError('ACTUAL_COMPLETE_QUALIFIED_TASK_PREFIX_REQUIRED')
        if e['purpose']=='DECISION_PRODUCTION'and(previous['scene_sha256']!=digest(spec['model_path'])or previous.get('reference_qualification',{}).get('reference_rule')!=DECISION_RULE):raise PermissionError('ACTUAL_SAME_SCENE_LATE_DECISION_QUALIFICATION_REQUIRED')
    elif e['purpose']=='INITIAL_SINGLE':
        if atoms!=24 or previous.get('phase')!='g2-standing-eef-y-v2'or previous['actual_target_integrations']!=2703:raise PermissionError('NEW_CANDIDATE_NATIVE_G2_REQUIRED')
    elif e['purpose']=='DURATION_SINGLE':
        expected={48:24,96:48,160:96}.get(atoms)
        if expected is None or previous['actual_target_integrations']!=2503+expected*25:raise PermissionError('SINGLE_VARIABLE_DURATION_LADDER_REQUIRED')
    elif prior_atoms!=atoms:raise PermissionError('ACTUAL_SAME_LENGTH_QUALIFICATION_REQUIRED')
    if e['purpose']=='PRODUCTION'and previous['scene_sha256']!=digest(spec['model_path']):raise PermissionError('ACTUAL_SCENE_DYNAMIC_QUALIFICATION_REQUIRED')
    return 2503+25*atoms


def validate_night1(spec,registry,now=None):
    now=time.time()if now is None else now;path=Path(spec['night1_binding']);b=json.loads(path.read_text())
    if (digest(path)!=spec['night1_binding_sha256']or b['plan_sha']!=PLAN_SHA
            or b['owner']!='EXISTING_LOCAL_PC'or b['campaign_root']!=spec['campaign_root']
            or b['max_native_slots']!=2 or b.get('engineering_budget_caps')is not None
            or now>=b['dispatch_cutoff_epoch']or spec['deadline_epoch']>b['deadline_epoch']
            or b['deadline_epoch']>min(b['activation_epoch']+8*3600,1793840400,b['original_campaign_deadline_epoch'])):raise PermissionError('REGISTERED_LIVE_NIGHT1_WINDOW_REQUIRED')
    if not any(e['path']==spec['registered_root']and e['source_sha']==spec['source_sha']and e.get('night1_binding')==str(path)for e in registry['entries']):
        raise PermissionError('EXISTING_REGISTRY_NIGHT1_RUN_IDENTITY_REQUIRED')


def model_fingerprint(model,version='QPOS0_XY_V1'):
    """Keep the robot/model fixed; mask only the declared object reset cells."""
    import numpy as np,mujoco
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'task_cube')
    if body<0:raise ValueError('ORIGINAL_TASK_CUBE_REQUIRED')
    joint=int(model.body_jntadr[body]);address=int(model.jnt_qposadr[joint])
    if model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('ORIGINAL_FREE_OBJECT_REQUIRED')
    home=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home')
    if version not in('QPOS0_XY_V1','HOME_KEYFRAME_XY_V2','HOME_ROOT_OBJECT_XY_V3')or(version!='QPOS0_XY_V1'and home<0):raise ValueError('EXACT_ORIGINAL_HOME_LAYOUT_VERSION_REQUIRED')
    if version=='HOME_ROOT_OBJECT_XY_V3'and not any(model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and model.jnt_qposadr[j]==0 for j in range(model.njnt)):raise ValueError('ORIGINAL_FREE_ROOT_ADDRESS_REQUIRED')
    h=hashlib.sha256();fields={}
    for name in sorted(dir(model)):
        if name.startswith('_'):continue
        value=getattr(model,name)
        if isinstance(value,np.ndarray):
            a=value.copy()
            if name=='qpos0'and version=='QPOS0_XY_V1':a[address:address+2]=0.
            if name=='key_qpos'and version in('HOME_KEYFRAME_XY_V2','HOME_ROOT_OBJECT_XY_V3'):a[home,address:address+2]=0.
            if name=='key_qpos'and version=='HOME_ROOT_OBJECT_XY_V3':a[home,:2]=0.
            identity=hashlib.sha256(a.tobytes()).hexdigest();fields[name]=dict(dtype=str(a.dtype),shape=list(a.shape),sha256=identity)
        elif isinstance(value,(str,bytes,int,float)):
            fields[name]=value.decode(errors='strict')if isinstance(value,bytes)else value
    options={}
    for name in sorted(dir(model.opt)):
        if name.startswith('_'):continue
        value=getattr(model.opt,name)
        if isinstance(value,np.ndarray):options[name]=value.tolist()
        elif isinstance(value,(int,float)):options[name]=value
    document=dict(model_fields=fields,options=options,mutable_initial_qpos_indices=[address,address+1],version=mujoco.__version__)
    if version!='QPOS0_XY_V1':document.update(layout_fingerprint_version=version,mutable_home_keyframe_index=home)
    if version=='HOME_ROOT_OBJECT_XY_V3':document['mutable_home_root_qpos_indices']=[0,1]
    h.update(json.dumps(document,sort_keys=True,allow_nan=False).encode());return h.hexdigest(),document
