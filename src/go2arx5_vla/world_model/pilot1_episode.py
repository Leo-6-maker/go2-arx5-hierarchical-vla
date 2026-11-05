"""PILOT1 one-decision native task; original action timing and safety remain intact."""
from pathlib import Path
import json
import numpy as np
from .storage import digest
from .next1_episode import RULE

OFFSETS = np.asarray([[0.,0.,0.],[.001,0.,0.],[-.001,0.,0.],[0.,.001,0.]])
METHODS = ('GEOMETRIC_DIRECTION_RULE', 'TRAIN_MEAN_RESPONSE_RULE', 'FROZEN_ARX_SELECTOR', 'TRAINED_ACTOR')


def episode_config(spec):
    path = Path(spec['episode_manifest'])
    episode = json.loads(path.read_text())
    pilot2=episode.get('schema')=='WM02_PILOT2_COVERAGE_EPISODE_V1'
    if pilot2:
        from .pilot2_coverage import coverage_contract
        coverage_contract(spec,episode)
    if (digest(path) != spec['episode_manifest_sha256'] or episode.get('schema') != ('WM02_PILOT2_COVERAGE_EPISODE_V1'if pilot2 else'WM02_PILOT1_EPISODE_V1')
            or episode.get('run') != spec['registered_root'] or episode.get('phase') != 'collect-target-pilot1'
            or episode.get('purpose') not in (('SCENE_QUALIFICATION','ACTION_COVERAGE')if pilot2 else('SCENE_QUALIFICATION', 'EVALUATION', 'TRAIN_FEEDBACK'))
            or episode.get('reference_rule') != RULE or episode.get('task_atoms') != 64
            or episode.get('prehold_atoms') != 40 or episode.get('screen_atom') != 48
            or episode.get('screen') != dict(window_s=.4, relative_TCP_diameter_m=.002, max_relative_TCP_speed_m_s=.025)
            or episode.get('split') not in (('TRAIN','DEV')if pilot2 else('TRAIN','DEV','TEST')) or episode.get('method') not in (('PREDECLARED_ACTION',)if pilot2 else METHODS)
            or tuple(episode['offset_xyz_m']) != (0,0,0) or not spec.get('pilot1_binding')):
        raise PermissionError('EXACT_PILOT1_TASK_AND_PREDECLARED_NATIVE_METHOD_REQUIRED')
    assets = json.loads(Path(spec['scene_allowlist']).read_text())
    matches = [s for s in assets['scenes'] if s['scene_id'] == episode['scene_id']]
    if digest(spec['scene_allowlist']) != spec['scene_allowlist_sha256'] or len(matches) != 1:
        raise PermissionError('PILOT1_FROZEN_SCENE_IDENTITY_REQUIRED')
    scene = matches[0]
    if (scene['parent'] != episode['parent'] or scene['split'] != episode['split']
            or scene['model_path'] != spec['model_path'] or scene['model_sha256'] != digest(spec['model_path'])
            or scene['static_check'] != 'PASSED_NO_INTEGRATION'
            or assets['native_binary_sha256'] != digest(spec['native_binary'])):
        raise PermissionError('NEW_PHYSICAL_PARENT_SPLIT_AND_UNCHANGED_EXECUTOR_REQUIRED')
    for key in ('actor_numpy', 'task', 'ARX'):
        if digest(spec[key]) != spec[key+'_sha256']:
            raise PermissionError('FROZEN_PILOT1_ACTOR_TEACHER_OR_TASK_CHANGED')
    from .pilot1_numpy import goal_offsets
    task=json.loads(Path(spec['task']).read_text())
    if spec.get('train_mean_response_model'):
        from .pilot2_coverage import PLAN
        from .pilot2_mean_rule import responses
        if spec.get('pilot2_plan_sha')!=PLAN:raise PermissionError('PILOT2_TRAIN_MEAN_RULE_SCOPE_REQUIRED')
        responses(spec,task)
    goals=goal_offsets(task)
    if (type(episode.get('goal_index'))is not int or not 0<=episode['goal_index']<len(goals)
            or not np.array_equal(np.asarray(episode.get('task_goal_offset_xy_m')),goals[episode['goal_index']])
            or(episode['purpose']=='TRAIN_FEEDBACK'and episode['split']!='TRAIN')):
        raise PermissionError('FIXED_TRAIN_ONLY_GOAL_GENERATOR_AND_FEEDBACK_PARTITION_REQUIRED')
    if episode['purpose'] != 'SCENE_QUALIFICATION':
        proof = Path(spec['scene_qualification'])
        result = json.loads(proof.read_text())
        old = json.loads((proof.parent.parent/'EPISODE_MANIFEST.private.json').read_text())
        prefix = json.loads((proof.parent/'STEADY_PREFIX_RESULT.private.json').read_text())
        if (digest(proof) != spec['scene_qualification_sha256'] or result.get('status') != 'NATIVE_STAGE_PASSED'
                or not result.get('shutdown_clean') or not prefix.get('qualified')
                or old.get('purpose') != 'SCENE_QUALIFICATION' or old['scene_id'] != episode['scene_id']
                or result['scene_sha256'] != scene['model_sha256']):
            raise PermissionError('NEW_SCENE_NATIVE_PREFIX_QUALIFICATION_REQUIRED')
    return episode


def episode_limit(spec, previous):
    episode_config(spec)
    if (not previous or previous.get('status') != 'NATIVE_STAGE_PASSED' or not previous.get('shutdown_clean')
            or previous.get('native_binary_sha256') != digest(spec['native_binary'])):
        raise PermissionError('UNCHANGED_C08_NATIVE_PREDECESSOR_REQUIRED')
    return 4103


def choose(adapter, episode):
    if episode.get('schema')=='WM02_PILOT2_COVERAGE_EPISODE_V1':
        from .pilot2_coverage import choose_predeclared
        return choose_predeclared(adapter,episode)
    from .target_arx import CausalARX
    from .next2_episode import state_vector
    from .target_reference import require_install_clock
    from scripts.pi0292.durable import write
    # Import NumPy-only inference: the native Python/ROS environment does not load torch.
    spec = adapter.spec
    task = json.loads(Path(spec['task']).read_text())
    meta = json.loads(Path(spec['teacher_metadata']).read_text())
    if digest(spec['teacher_metadata']) != spec['teacher_metadata_sha256']:
        raise PermissionError('TEACHER_OBSERVATION_LAYOUT_CHANGED')
    with np.load(spec['ARX'], allow_pickle=False) as arrays:
        model = CausalARX.from_arrays(meta, arrays)
    with np.load(spec['actor_numpy'], allow_pickle=False) as arrays:
        actor = {k: arrays[k].copy() for k in arrays.files}
    adapter.wait_source()
    packet = adapter.observe()
    if packet['step'] != 3703:
        raise ValueError('EXACT_REAL_PILOT1_DECISION_BOUNDARY_REQUIRED')
    source = float(adapter.latest['ee'].time)
    require_install_clock(source, packet['time_s'])
    history = np.asarray([state_vector(s) for s in adapter.next1_samples[-9:]])
    initial = np.r_[adapter.active_reference['action_EEF10'][-1], adapter.base_pose_hold]
    data = dict(state=history[None], goal=initial[None], history_goal=np.repeat(initial[None,None],8,axis=1),
                source_age=np.full((1,24),packet['time_s']-source), window_start_s=np.asarray([0.]),
                window_start_atom=np.asarray([0]), future_prefix_remaining=np.asarray([7.]))
    target_delta = np.asarray(episode['task_goal_offset_xy_m'])
    if target_delta.shape != (2,) or not np.isfinite(target_delta).all():
        raise ValueError('PREDECLARED_TASK_GOAL_REQUIRED')
    tcp = adapter.measured_task_state()['tcp'][:3,3].copy()
    task_goal = tcp[:2] + target_delta
    predictions = []
    for offset in OFFSETS:
        sample = dict(data); sample['goal'] = initial[None].copy(); sample['goal'][:,:3] += offset
        predictions.append(model.predict(sample,np.asarray([0]))[0])
    predictions = np.asarray(predictions)
    xy = predictions[:,-1,model.layout['tcp_position_m']][:,:2]
    cost = task['nonhold_action_cost_m'] * np.asarray([0,1,1,1])
    from .pilot2_mean_rule import responses
    mean_response,mean_sha=responses(spec,task)
    errors = dict(GEOMETRIC_DIRECTION_RULE=np.linalg.norm(initial[None,:2]+OFFSETS[:,:2]-task_goal,axis=1)+cost,
                  TRAIN_MEAN_RESPONSE_RULE=np.linalg.norm(mean_response-target_delta,axis=1)+cost,
                  FROZEN_ARX_SELECTOR=np.linalg.norm(xy-task_goal,axis=1)+cost)
    from .pilot1_numpy import actor_features, logits
    inputs = actor_features(data,meta,np.asarray([0]),target_delta[None])
    scores = logits(actor,inputs)[0]
    choices = {k:int(v.argmin()) for k,v in errors.items()}; choices['TRAINED_ACTOR']=int(scores.argmax())
    selected = 0 if episode['purpose']=='SCENE_QUALIFICATION' else choices[episode['method']]
    value = dict(schema='WM02_PILOT1_DECISION_BEFORE_ACTION_V1', source_sha=spec['source_sha'], native_step=packet['step'],
                 task_name=task['schema'], method=episode['method'], split=episode['split'], scene_id=episode['scene_id'],
                 goal_offset_xy_m=target_delta.tolist(), actual_decision_TCP_m=tcp.tolist(), task_goal_xy_m=task_goal.tolist(),
                 actual_history=history.tolist(), past_reference=initial.tolist(), source_age_s=packet['time_s']-source,
                 predictions=predictions.tolist(), predicted_objective_m={k:v.tolist() for k,v in errors.items()},
                 actor_logits=scores.tolist(), proposed_actions=choices, selected_action=selected,
                 selected_offset_xyz_m=OFFSETS[selected].tolist(), future_response_observed=False,
                 Actor_sha256=spec['actor_numpy_sha256'], teacher_sha256=spec['ARX_sha256'], task_sha256=spec['task_sha256'])
    value['train_mean_response_model_sha256']=mean_sha
    write(adapter.output/'PILOT1_DECISION.private.json',value)
    if adapter.observe()['step'] != packet['step']:
        raise RuntimeError('PHYSICS_ADVANCED_DURING_ACTION_CHOICE')
    adapter.command_event(dict(phase='pilot1-choice-before-install',native_step=packet['step'],decision_sha256=digest(adapter.output/'PILOT1_DECISION.private.json'),selected_action=selected))
    return OFFSETS[selected].copy()


def score(adapter, episode):
    from scripts.pi0292.durable import write
    decision = json.loads((adapter.output/'PILOT1_DECISION.private.json').read_text())
    task = json.loads(Path(adapter.spec['task']).read_text())
    actual = adapter.measured_task_state()['tcp'][:3,3]
    error = float(np.linalg.norm(actual[:2]-decision['task_goal_xy_m']))
    value = dict(schema='LOCAL_EEF_REACH_PILOT_SCORE_V1', method=episode['method'], purpose=episode['purpose'],
                 split=episode['split'], parent=episode['parent'], actual_endpoint_TCP_m=actual.tolist(),
                 task_goal_xy_m=decision['task_goal_xy_m'], actual_XY_error_m=error,
                 action=decision['selected_action'], objective_m=error+task['nonhold_action_cost_m']*(decision['selected_action']!=0),
                 success_tolerance_m=task['success_tolerance_m'], local_XY_success=error<=task['success_tolerance_m'],
                 Z_displacement_m=float(actual[2]-decision['actual_decision_TCP_m'][2]), full_manipulation_success=False)
    write(adapter.output/'PILOT1_SCORE.private.json',value)
    return value
