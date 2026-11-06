"""Known-intent four-arm calibration on the unchanged one-decision native path."""
from pathlib import Path
import json
import numpy as np
from .fusion_contract import PLAN, OFFSETS
from .fusion_corrector import ARMS, choose as select
from .storage import digest

SCHEMA = 'WM02_FUSION_LOCAL_CALIBRATION_EPISODE_V1'


def episode_config(spec):
    from .fusion_scope import validate
    from .next1_episode import RULE
    validate(spec)
    p = Path(spec['episode_manifest']); e = json.loads(p.read_text())
    if (digest(p) != spec['episode_manifest_sha256'] or e.get('schema') != SCHEMA
            or e.get('run') != spec['registered_root'] or e.get('phase') != 'collect-target-pilot1'
            or spec.get('fusion_plan_sha') != PLAN
            or e.get('purpose') not in ('SCENE_QUALIFICATION', 'EVALUATION', 'TRAIN_FEEDBACK')
            or e.get('split') not in ('TRAIN', 'DEV', 'TEST')
            or e.get('purpose') == 'TRAIN_FEEDBACK' and e.get('split') != 'TRAIN'
            or e.get('reference_rule') != RULE or e.get('task_atoms') != 64
            or e.get('prehold_atoms') != 40 or e.get('screen_atom') != 48
            or e.get('screen') != dict(window_s=.4, relative_TCP_diameter_m=.002,
                                       max_relative_TCP_speed_m_s=.025)
            or e.get('method') not in ARMS or tuple(e.get('offset_xyz_m', [])) != (0, 0, 0)
            or type(e.get('nominal_action')) is not int or e['nominal_action'] not in range(4)
            or e.get('desired_outcome_provenance') != 'CALIBRATION_INTENT'
            or e.get('inference_mode') != 'CALIBRATION_NO_VLA'
            or tuple(e.get('desired_delta_xy_m', [])) not in tuple(map(tuple, OFFSETS[:, :2]))):
        raise PermissionError('EXACT_KNOWN_INTENT_FOUR_ARM_FUSION_CONTRACT_REQUIRED')
    catalog = json.loads(Path(spec['scene_allowlist']).read_text())
    matches = [s for s in catalog['scenes'] if s['scene_id'] == e['scene_id']]
    if digest(spec['scene_allowlist']) != spec['scene_allowlist_sha256'] or len(matches) != 1:
        raise PermissionError('FUSION_PREDECLARED_PHYSICAL_PARENT_REQUIRED')
    scene = matches[0]
    if (scene['parent'] != e['parent'] or scene['split'] != e['split']
            or scene['model_path'] != spec['model_path'] or digest(spec['model_path']) != scene['model_sha256']
            or scene['static_check'] != 'PASSED_NO_INTEGRATION'
            or catalog['native_binary_sha256'] != digest(spec['native_binary'])):
        raise PermissionError('ACTUAL_FUSION_SCENE_AND_UNCHANGED_EXECUTOR_REQUIRED')
    for name in ['task', 'ARX', 'teacher_metadata', 'train_mean_response_model']:
        if digest(spec[name]) != spec[name+'_sha256']:
            raise PermissionError('FROZEN_FUSION_TASK_OR_TRAIN_MODEL_CHANGED')
    if e['purpose'] != 'SCENE_QUALIFICATION':
        proof = Path(spec['scene_qualification']); result = json.loads(proof.read_text())
        old = json.loads((proof.parent.parent/'EPISODE_MANIFEST.private.json').read_text())
        prefix = json.loads((proof.parent/'STEADY_PREFIX_RESULT.private.json').read_text())
        if (digest(proof) != spec['scene_qualification_sha256']
                or result.get('status') != 'NATIVE_STAGE_PASSED' or not result.get('shutdown_clean')
                or result.get('physics_stops') or result.get('scene_sha256') != scene['model_sha256']
                or result.get('native_binary_sha256') != catalog['native_binary_sha256']
                or old.get('schema') != SCHEMA or old.get('scene_id') != e['scene_id']
                or old.get('purpose') != 'SCENE_QUALIFICATION' or not prefix.get('qualified')):
            raise PermissionError('FRESH_FUSION_SCENE_NATIVE_PREFIX_REQUIRED')
    return e


def choose(adapter, episode):
    from .target_arx import CausalARX
    from .next2_episode import state_vector
    from .target_reference import require_install_clock, fixed_goal_poses
    from .pilot2_mean_rule import responses
    from ..task_data.codec import pose10
    from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
    from scripts.pi0292.durable import write
    spec = adapter.spec; meta = json.loads(Path(spec['teacher_metadata']).read_text())
    with np.load(spec['ARX'], allow_pickle=False) as z: model = CausalARX.from_arrays(meta, z)
    task = json.loads(Path(spec['task']).read_text()); mean, mean_sha = responses(spec, task)
    adapter.wait_source(); packet = adapter.observe(); source = float(adapter.latest['ee'].time)
    require_install_clock(source, packet['time_s'])
    history = np.asarray([state_vector(s) for s in adapter.next1_samples[-9:]])
    if packet['step'] != 3703 or history.shape != (9, model.dimension):
        raise ValueError('ACTUAL_SAME_BOUNDARY_H16_FUSION_HISTORY_REQUIRED')
    reference = np.r_[adapter.active_reference['action_EEF10'][-1], adapter.base_pose_hold]
    data = dict(state=history[None], goal=reference[None],
        history_goal=np.repeat(reference[None, None], 8, axis=1),
        source_age=np.full((1, 24), packet['time_s']-source), window_start_s=np.array([0.]),
        window_start_atom=np.array([0]), future_prefix_remaining=np.array([7.]))
    predictions = []
    for offset in OFFSETS:
        candidate = dict(data, goal=reference[None].copy()); candidate['goal'][:, :3] += offset
        predictions.append(model.predict(candidate, np.array([0]))[0])
    predictions = np.asarray(predictions); current = adapter.measured_task_state()
    tcp = current['tcp'][:3, 3].copy(); desired = tcp[:2]+np.asarray(episode['desired_delta_xy_m'])
    value = select(arm=episode['method'], nominal_action=episode['nominal_action'],
        desired_xy_m=desired, provenance=episode['desired_outcome_provenance'], actual_tcp_xy_m=tcp[:2],
        static_response_xy_m=mean, WM_endpoint_xy_m=predictions[:, -1, model.layout['tcp_position_m']][:, :2],
        nonhold_cost_m=task['nonhold_action_cost_m'], observation_time_s=packet['time_s'], source_time_s=source)
    if value['status'] != 'LOCAL_CANDIDATE_SELECTED_NOT_INSTALLED':
        raise RuntimeError('STALE_FUSION_PROPOSAL_NO_INSTALL_REQUIRE_FRESH_OBSERVATION')
    selected = 0 if episode['purpose'] == 'SCENE_QUALIFICATION' else value['selected_action']
    # Initial TCP is the actual task start, exactly as in the qualified compiler.
    initial = adapter.fusion_initial_tcp
    def preview(index):
        goal = initial.copy(); goal[:3, 3] += OFFSETS[index]
        return np.stack([pose10(p, float(teacher_physical_fraction(1.))) for p in
                         fixed_goal_poses(initial, goal, renewed=False)])
    value.update(schema='WM02_FUSION_DECISION_BEFORE_ACTION_V1', source_sha=spec['source_sha'],
        native_step=packet['step'], sim_time_s=packet['time_s'], actual_history=history.tolist(),
        proposed_corrector_action=value['selected_action'], selected_action=selected,
        selected_offset_xyz_m=OFFSETS[selected].tolist(),
        corrected_offset_xyz_m=OFFSETS[selected].tolist(),
        correction_xyz_m=(OFFSETS[selected]-OFFSETS[episode['nominal_action']]).tolist(),
        qualified_hold_only=episode['purpose'] == 'SCENE_QUALIFICATION',
        actual_decision_TCP_m=tcp.tolist(), actual_decision_qpos=current['qpos'].tolist(),
        desired_delta_xy_m=episode['desired_delta_xy_m'], nominal_reference_EEF10=preview(episode['nominal_action']).tolist(),
        proposed_reference_EEF10=preview(selected).tolist(), predictions=predictions.tolist(),
        past_reference=reference.tolist(), method=episode['method'], parent=episode['parent'], split=episode['split'],
        task_sha256=spec['task_sha256'], WM_sha256=spec['ARX_sha256'], static_model_sha256=mean_sha,
        inference_mode='CALIBRATION_NO_VLA', VLA_inferences=0, vla_raw_action=None,
        installed_reference='NOT_YET_INSTALLED', consumed_reference='NOT_YET_OBSERVED',
        controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',
        model_clock=dict(window_start_s=0., future_prefix_remaining=7., history_atoms=8, forecast_atoms=16),
        executor_model_transfer='FROZEN_C08_ARX_ON_C10_EXECUTOR_DIAGNOSTIC_NOT_REFIT',
        observation_information='MATCHED_SIMULATOR_STATE_INCLUDING_OBJECT_POSE_NOT_RGB_DEPLOYMENT',
        decision_wall_mode='PAUSE_ASSISTED_DIAGNOSTIC')
    write(adapter.output/'FUSION_DECISION.private.json', value)
    if adapter.observe()['step'] != packet['step']: raise RuntimeError('PHYSICS_ADVANCED_DURING_FUSION_CHOICE')
    adapter.fusion_selected_action = selected; adapter.fusion_install_action_count = 0
    adapter.command_event(dict(phase='fusion-decision-before-install', native_step=packet['step'],
        decision_sha256=digest(adapter.output/'FUSION_DECISION.private.json'), selected_action=selected))
    return OFFSETS[selected].copy()


def score(adapter, episode):
    from scripts.pi0292.durable import write
    from .target_stability import motion
    decision = json.loads((adapter.output/'FUSION_DECISION.private.json').read_text())
    task = json.loads(Path(adapter.spec['task']).read_text()); state = adapter.measured_task_state()
    endpoint = state['tcp'][:3, 3]; target = np.asarray(decision['desired_xy_m'])
    error = float(np.linalg.norm(endpoint[:2]-target)); q0 = np.array(decision['actual_decision_qpos'])
    relative = motion(np.array([q0[:3], state['qpos'][:3]]),
        np.array([q0[3:7], state['qpos'][3:7]]),
        np.array([decision['actual_decision_TCP_m'], endpoint]))['relative_tcp']
    result = dict(schema='WM02_FUSION_ACTUAL_LOCAL_XY_SCORE_V1', inference_mode='CALIBRATION_NO_VLA',
        method=episode['method'], purpose=episode['purpose'], parent=episode['parent'], split=episode['split'],
        desired_outcome_provenance='CALIBRATION_INTENT', desired_xy_m=target.tolist(),
        actual_endpoint_TCP_m=endpoint.tolist(), actual_XY_error_m=error,
        local_XY_success=error <= task['success_tolerance_m'], success_tolerance_m=task['success_tolerance_m'],
        selected_action=decision['selected_action'], nominal_action=decision['nominal_action'],
        objective_m=error+task['nonhold_action_cost_m']*(decision['selected_action'] != 0),
        actual_world_displacement_m=(endpoint-decision['actual_decision_TCP_m']).tolist(),
        actual_BASE_relative_displacement_m=(relative[-1]-relative[0]).tolist(),
        actual_BASE_displacement_m=(state['qpos'][:3]-q0[:3]).tolist(),
        controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',
        exact_counterfactual_fork=False, full_task_success=False, VLA_inferences=0,
        CALIBRATION_CORRECTION_GAIN='NOT_DECIDED_BY_SINGLE_EPISODE', TASK_TRANSFER_GAIN=False)
    write(adapter.output/'FUSION_SCORE.private.json', result); return result
