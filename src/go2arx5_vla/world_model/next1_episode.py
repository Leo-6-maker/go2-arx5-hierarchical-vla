"""NEXT1's causal steady-prefix contract, separate from closed NIGHT1."""
from pathlib import Path
import json
import numpy as np
from .storage import digest

RULE='FIXED_HOLD40_BURN8_DECIDE_RAMP8_V1'


def episode_config(spec):
    path=Path(spec['episode_manifest']);e=json.loads(path.read_text())
    allowed=('STEADY_SINGLE','STEADY_DUAL','ACTION_PILOT','ACTION_DEV','HOLD_REPEAT')
    if (digest(path)!=spec['episode_manifest_sha256'] or e.get('schema')!='WM02_NEXT1_EPISODE_V1'
            or e.get('run')!=spec['registered_root']or e.get('phase')!='collect-target-next1'
            or e.get('purpose')not in allowed or e.get('reference_rule')!=RULE
            or e.get('task_atoms')not in(48,64,96,160)or e.get('prehold_atoms')!=40
            or e.get('screen_atom')!=48 or e.get('split')not in('TRAIN','DEV')
            or e.get('screen')!=dict(window_s=.4,relative_TCP_diameter_m=.002,max_relative_TCP_speed_m_s=.025)
            or np.asarray(e['offset_xyz_m']).shape!=(3,)or not np.isfinite(e['offset_xyz_m']).all()
            or tuple(e['offset_xyz_m'])not in((0,0,0),(.001,0,0),(-.001,0,0),(0,.001,0))
            or(e['purpose'].startswith('STEADY_')and any(e['offset_xyz_m']))
            or(not e['purpose'].startswith('STEADY_')and e['task_atoms']!=64)):
        raise PermissionError('EXACT_PREDECLARED_NEXT1_STEADY_CAUSAL_EPISODE_REQUIRED')
    assets=json.loads(Path(spec['scene_allowlist']).read_text())
    if digest(spec['scene_allowlist'])!=spec['scene_allowlist_sha256']:raise PermissionError('NEXT1_SCENE_ALLOWLIST_CHANGED')
    matches=[x for x in assets['scenes']if x['scene_id']==e['scene_id']]
    if len(matches)!=1:raise PermissionError('NEXT1_KNOWN_PHYSICAL_PARENT_REQUIRED')
    a=matches[0]
    if (a['parent']!=e['parent']or a['split']!=e['split']or a['model_path']!=spec['model_path']
            or a['model_sha256']!=digest(spec['model_path'])or assets['native_binary_sha256']!=digest(spec['native_binary'])):raise PermissionError('NEXT1_FROZEN_MODEL_PARENT_SPLIT_REQUIRED')
    return e


def episode_limit(spec,previous):
    e=episode_config(spec)
    if not previous or previous.get('status')!='NATIVE_STAGE_PASSED'or not previous.get('shutdown_clean')or previous.get('native_binary_sha256')!=digest(spec['native_binary']):raise PermissionError('EXISTING_C08_NATIVE_QUALIFICATION_REQUIRED')
    return 2503+25*e['task_atoms']


def assess(adapter,episode):
    from scripts.pi0292.durable import write
    from .target_stability import motion,screen
    samples=adapter.next1_samples[-9:]
    b=np.asarray([s['qpos'][:3]for s in samples]);q=np.asarray([s['qpos'][3:7]for s in samples]);tcp=np.asarray([s['tcp'][:3,3]for s in samples]);t=np.asarray([s['time_s']for s in samples])
    relative=motion(b,q,tcp)['relative_tcp'];value=screen(t,relative)
    value.update(schema='WM02_NEXT1_SCIENTIFIC_PREFIX_V1',reference_rule=RULE,screen_atom=episode['screen_atom'],actual_native_step=adapter.observe()['step'],independent_initialization=True,source_sha=adapter.spec['source_sha'],native_binary_sha256=digest(adapter.spec['native_binary']),world_TCP_displacement_m=(tcp[-1]-tcp[0]).tolist(),BASE_displacement_m=(b[-1]-b[0]).tolist(),physics_paused_at_assessment=True,thresholds_not_native_safety=True)
    write(adapter.output/'STEADY_PREFIX_RESULT.private.json',value);adapter.next1_screen=value
    return value
