"""NEXT2 local response examination; fresh scope and prediction-first evidence."""
from pathlib import Path
import json
import numpy as np
from .storage import digest
from .next1_episode import RULE,assess


def episode_config(spec):
    p=Path(spec['episode_manifest']);e=json.loads(p.read_text())
    if(digest(p)!=spec['episode_manifest_sha256']or e.get('schema')!='WM02_NEXT2_EPISODE_V1'or e.get('run')!=spec['registered_root']or e.get('phase')!='collect-target-next2'
       or e.get('purpose')not in('SCENE_QUALIFICATION','PROSPECTIVE','HOLD_REPEAT')or e.get('reference_rule')!=RULE or e.get('task_atoms')!=64 or e.get('prehold_atoms')!=40 or e.get('screen_atom')!=48
       or e.get('screen')!=dict(window_s=.4,relative_TCP_diameter_m=.002,max_relative_TCP_speed_m_s=.025)
       or tuple(e['offset_xyz_m'])not in((0,0,0),(.001,0,0),(-.001,0,0),(0,.001,0))or not spec.get('next2_binding')or spec.get('next1_binding')):raise PermissionError('EXACT_NEXT2_PREDICTION_FIRST_EPISODE_REQUIRED')
    assets=json.loads(Path(spec['scene_allowlist']).read_text());matches=[a for a in assets['scenes']if a['scene_id']==e['scene_id']]
    if digest(spec['scene_allowlist'])!=spec['scene_allowlist_sha256']or len(matches)!=1:raise PermissionError('FROZEN_NEXT2_SCENE_REQUIRED')
    a=matches[0]
    if a['scene_id']not in['P%02d'%i for i in range(9,17)]or a['parent']!=e['parent']or a['split']!=e['split']or a['model_path']!=spec['model_path']or a['model_sha256']!=digest(spec['model_path'])or assets['native_binary_sha256']!=digest(spec['native_binary']):raise PermissionError('P09_TO_P16_ORIGINAL_PHYSICAL_PARENT_AND_SPLIT_REQUIRED')
    for key in('frozen_predictors','frozen_thresholds'):
        if digest(spec[key])!=spec[key+'_sha256']:raise PermissionError('PRE_SAMPLING_FROZEN_MODEL_OR_THRESHOLD_CHANGED')
    if e['purpose']!='SCENE_QUALIFICATION':
        p=Path(spec['scene_qualification']);q=json.loads(p.read_text());es=json.loads((p.parent.parent/'EPISODE_MANIFEST.private.json').read_text());prefix=json.loads((p.parent/'STEADY_PREFIX_RESULT.private.json').read_text())
        if digest(p)!=spec['scene_qualification_sha256']or q.get('status')!='NATIVE_STAGE_PASSED'or not q.get('shutdown_clean')or q.get('scene_sha256')!=a['model_sha256']or es.get('purpose')!='SCENE_QUALIFICATION'or es.get('scene_id')!=e['scene_id']or not prefix.get('qualified'):raise PermissionError('CURRENT_PHYSICAL_PARENT_SAFE_QUALIFICATION_REQUIRED')
    proofs=spec['steady_prefix_qualifications']
    if len(proofs)!=2 or len({p['path']for p in proofs})!=2:raise PermissionError('TWO_PRIOR_INDEPENDENT_STEADY_PREFIXES_REQUIRED')
    for proof in proofs:
        path=Path(proof['path']);v=json.loads(path.read_text());r=json.loads((path.parent/'RESULT.json').read_text())
        if digest(path)!=proof['sha256']or digest(path.parent/'RESULT.json')!=proof['native_result_sha256']or not v.get('qualified')or v.get('screen_atom')!=48 or r.get('status')!='NATIVE_STAGE_PASSED'or r.get('physics_stops')or not r.get('shutdown_clean')or r.get('native_binary_sha256')!=assets['native_binary_sha256']:raise PermissionError('ACTUAL_C08_STABLE_PREFIX_PROOFS_REQUIRED')
    return e


def episode_limit(spec,previous):
    e=episode_config(spec)
    if not previous or previous.get('status')!='NATIVE_STAGE_PASSED'or not previous.get('shutdown_clean')or previous.get('native_binary_sha256')!=digest(spec['native_binary']):raise PermissionError('PRIOR_C08_NATIVE_SAFE_ENDPOINT_REQUIRED')
    return 2503+25*e['task_atoms']


def state_vector(s):
    tp,op=s['tcp'],s['object_pose']
    return np.r_[s['qpos'],s['qvel'],s['ctrl'],tp[:3,3],tp[:3,0],tp[:3,1],op[:3,3],op[:3,0],op[:3,1],s['finger_q']]


def predict_before_install(adapter,episode):
    from .target_arx import CausalARX
    from .next2_response import HoldResponse,future_choice_mask
    from .target_reference import require_install_clock
    from scripts.pi0292.durable import write
    spec=adapter.spec;f=json.loads(Path(spec['frozen_predictors']).read_text());meta=f['metadata'];models={}
    for label in('ARX','NO_FUTURE_CHOICE_ARX','HOLD_RESPONSE'):
        m=f[label]
        if digest(m['path'])!=m['sha256']:raise PermissionError('FROZEN_PREDICTOR_BYTES_CHANGED')
        with np.load(m['path'],allow_pickle=False)as arrays:models[label]=HoldResponse.from_arrays(meta,arrays,baseline=m['baseline'],ridge=m['ridge'])if label=='HOLD_RESPONSE'else CausalARX.from_arrays(meta,arrays)
    adapter.wait_source();packet=adapter.observe();source=float(adapter.latest['ee'].time);require_install_clock(source,packet['time_s']);history=np.asarray([state_vector(s)for s in adapter.next1_samples[-9:]])
    if history.shape!=(9,106)or packet['step']!=3703:raise ValueError('EXACT_LIVE_PRE_ACTION_HISTORY_REQUIRED')
    initial=np.r_[adapter.active_reference['action_EEF10'][-1],adapter.base_pose_hold];base=dict(state=history[None],history_goal=np.repeat(initial[None,None],8,axis=1),source_age=np.full((1,24),packet['time_s']-source),window_start_s=np.asarray([0.]),window_start_atom=np.asarray([0]),future_prefix_remaining=np.asarray([7.]))
    rows=[]
    for name,offset in [('HOLD',(0,0,0)),('PLUS_X_1MM',(.001,0,0)),('MINUS_X_1MM',(-.001,0,0)),('PLUS_Y_1MM',(0,.001,0))]:
        goal=initial.copy();goal[:3]+=offset;data=dict(base,goal=goal[None]);pred={k:m.predict(future_choice_mask(data)if k=='NO_FUTURE_CHOICE_ARX'else data,np.asarray([0]))[0].tolist()for k,m in models.items()};rows.append(dict(action=name,offset_xyz_m=offset,predictions=pred))
    value=dict(schema='WM02_NEXT2_PREDICTION_FIRST_V1',source_sha=spec['source_sha'],native_step=packet['step'],sim_time_s=packet['time_s'],source_mpc_s=source,future_response_observed=False,chosen_offset_xyz_m=episode['offset_xyz_m'],models_sha256={k:f[k]['sha256']for k in models},frozen_predictors_sha256=spec['frozen_predictors_sha256'],thresholds_sha256=spec['frozen_thresholds_sha256'],observed_history=history.tolist(),model_input_clock=dict(window_start_s=0.,window_start_atom=0,future_prefix_remaining=7.,actual_current_source_age_s=packet['time_s']-source,clock_origin='REPLAY_WINDOW_AFTER_EXCLUDED_HOLD40; matches frozen fit, not full native task age'),past_reference=initial.tolist(),candidates=rows,counterfactual='MODEL_ONLY_FOUR_CANDIDATES; physical comparisons use independent restarts, no full-state forks')
    write(adapter.output/'PREDICTIONS_BEFORE_ACTION.private.json',value)
    if adapter.observe()['step']!=packet['step']:raise RuntimeError('PHYSICS_ADVANCED_DURING_PREDICTION')
    adapter.command_event(dict(phase='predictions-sealed-before-action',native_step=packet['step'],sha256=digest(adapter.output/'PREDICTIONS_BEFORE_ACTION.private.json'),future_response_observed=False));return value
