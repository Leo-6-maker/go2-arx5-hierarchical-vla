"""AUTO1 finite single-device admission; existing registry/owner remain in use."""
import json,os,time
from pathlib import Path


def validate_device(spec,actual):
    allowed={'GPU-845348f1-1511-dc19-0877-cc97658e3edd','GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'}
    night1=bool(spec.get('night1_binding'))
    endpoint_ok=(type(spec.get('endpoint'))is int and spec['endpoint']>=2)if night1 else spec.get('endpoint')in(200,100)
    if (spec.get('GPU_uuid')not in allowed or actual['uuid']!=spec['GPU_uuid']
            or spec.get('profile')!='wm02-auto1-target-neural' or spec.get('physics')is not False
            or spec.get('data_domain')!=('WM02_TARGET_NATIVE_REPLAY_V2'if night1 else'WM02_TARGET_NATIVE_STATE_V1')
            or not endpoint_ok or spec.get('Actor_updates')!=0
            or not 0<spec.get('allocator_limit_bytes',0)<=13*1024**3
            or actual['free_bytes']<spec['allocator_limit_bytes']+1024**3):
        raise PermissionError('FINITE_TARGET_NATIVE_MODEL_AND_ACTUAL_GPU_HEADROOM_REQUIRED')


def validate_campaign(spec):
    from .storage import digest
    campaign=Path(spec['campaign_root']);lease=json.loads((campaign/'control/LEASE.json').read_text())
    if (spec['host']!='corelab-4080s'or spec['uid']!=2009 or os.getuid()!=2009
            or lease['status']!='ACTIVE'or lease['scope']!='WM02_AUTO1_SIM_ONLY'
            or lease['run_uuid']!=campaign.name or lease['owner']!='EXISTING_LOCAL_PC'
            or lease['binding_sha256']!=digest(spec['campaign_binding'])
            or time.time()>=lease['deadline_epoch']or(campaign/'STOP.json').exists()):
        raise PermissionError('ACTUAL_ACTIVE_AUTO1_TARGET_MODEL_SCOPE_REQUIRED')
    registry=json.loads(Path(spec['registry']).read_text())
    if not any(e.get('scope')=='WM02_AUTO1_SIM_ONLY'and e.get('path')==str(campaign)and e.get('owner')=='EXISTING_LOCAL_PC'for e in registry['entries']):raise PermissionError('ACTUAL_AUTO1_REGISTRY_REQUIRED')
    # Match the replay to its actual checked numerical build, before torch.
    build=json.loads(Path(spec['replay_result']).read_text())
    night1=bool(spec.get('night1_binding'))
    if (digest(spec['replay_result'])!=spec['replay_result_sha256']or build.get('schema')!=('WM02_TARGET_NATIVE_REPLAY_V2'if night1 else'WM02_TARGET_NATIVE_STATE_V1')
            or build.get('status')!='ACTUAL_TARGET_SEQUENTIAL_REPLAY_BUILT'
            or digest(spec['replay'])!=build.get('replay_sha256')):raise PermissionError('REAL_NATIVE_REPLAY_BUILD_IDENTITY_REQUIRED')
    if night1:
        from .target_manifest import validate_night1
        validate_night1(spec,registry)
        if build.get('windows',0)<1 or build['task_atoms']<24:raise PermissionError('REAL_NIGHT1_WINDOWS_REQUIRED')
        if spec.get('resume')and digest(spec['resume'])!=spec.get('resume_sha256'):raise PermissionError('FROZEN_NIGHT1_RESUME_IDENTITY_REQUIRED')
    elif (spec['endpoint']==200 and(build['episodes']!=8 or spec.get('resume'))
            or spec['endpoint']==100 and(build['episodes']!=12 or not spec.get('resume')or digest(spec['resume'])!=spec.get('resume_sha256'))):raise PermissionError('EXACT_FIRST8_OR_SECOND4_TARGET_PIPELINE_REQUIRED')
    for key in('baseline_fit_replay','baseline_evaluation_replay'):
        if spec.get(key)and digest(spec[key])!=spec[key+'_sha256']:raise PermissionError('FROZEN_BASELINE_REPLAY_CHANGED')
