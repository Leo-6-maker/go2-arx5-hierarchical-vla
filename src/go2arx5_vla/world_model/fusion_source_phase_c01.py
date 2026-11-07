"""Distinct source-phase scene: qualify initialization and save one query only.

No generated reference is installed, no new contact/action domain is qualified,
and the old T06 reference recipe is explicitly a calibration recipe only.
"""
import copy
import json
from pathlib import Path
import numpy as np
from .storage import digest,write_json

CONDITION='SOURCE_PHASE_C01_QUERY_ONLY_AT34303_V1'
B=34303
SCENE_ID='FUSION_SOURCE_PHASE_C01_V1'
PARENT='FUSION_SOURCE_PHASE_C01_PRIVILEGED_DIAGNOSTIC'
SCENE='24fc70f0b9911312f17a083cd2f254252e0003583aac3520a4e3ce642253b45f'
STATIC='a32a163bd0a21e4a4f330e212a93316e9495d8e42f098ffe5a585b52813fa6e5'
ROBOT='1cdaf83146d8a9961d5d8de200098338b41010eb9492a6b69fe37c02662723ed'
CALIBRATION_SCENE='40408bbcbef2469396187c741750905a6ba9ed7161dc031d14bfab5fed62ddd2'
WRAPPER='WM02_C01_ACTUAL_SCENE_AND_NOMINAL_CALIBRATION_PREFIX_V1'
CAPTURE_DISTANCE_M=(.07,.09)
NOMINAL_REFERENCE_SHA='f658b8b6d9eec66230d633b227b1505ad0554191b1758447d88495f3ef941e85'


def nominal_reference(reference):
    from scripts.pi0292.relay_protocol import unseal
    from . import fusion_task_phase_initialization as original
    need(set(reference)=={'path','sha256'}and reference['sha256']==NOMINAL_REFERENCE_SHA,'DECLARED_NOMINAL_CALIBRATION_EVIDENCE')
    path=Path(reference['path']);need(path.is_absolute()and path.resolve()==path and not path.is_symlink()and digest(path)==NOMINAL_REFERENCE_SHA,'EXACT_PRIOR_CALIBRATION_BINDING')
    saved=json.loads(path.read_bytes());body=unseal(saved['frozen_input'])
    inputs=dict(frozen=saved['frozen_input'],raw=np.repeat(original.pose10(np.asarray(body['full_goal_TCP_target']),original.OPEN)[None],40,axis=0),provenance=original._provenance(body))
    original._inputs(inputs)
    need(body['scene_sha256']==CALIBRATION_SCENE and saved['actual_VLA_commands']==0,'CALIBRATION_INTENT_NOT_ACHIEVED_EXPERT_ACTION')
    return np.asarray(body['frozen_cube_pose_target'],float)


def need(ok,why):
    if not ok:raise ValueError('SOURCE_PHASE_C01_'+why)


def scene(spec):
    from .fusion_live_program_replay_v1 import assignment
    need(spec.get('task_condition')==CONDITION and spec.get('phase')=='collect-target-fusion-task-phase-live-skill-v9'
         and spec.get('kind')=='FUSION_TASK_PHASE_LIVE_SKILL_V9'
         and spec.get('live_skill_start_step')==spec.get('max_ticks')==B
         and assignment(spec)['split']=='DIAGNOSTIC'
         and all(spec.get(k)is None for k in ('task_reference_diagnostic','program_response_forecast','program_response_counterfactual','program_reference_intervention','program_reference_selection')),
         'DISTINCT_DIAGNOSTIC_QUERY_ONLY_SCOPE')
    binding=spec.get('source_phase_scene_binding')or{}
    need(set(binding)=={'scene_id','parent','model_sha256','static_result','static_result_sha256','nominal_calibration_goal_binding','nominal_calibration_goal_binding_sha256'}
         and binding['scene_id']==SCENE_ID and binding['parent']==PARENT
         and binding['model_sha256']==SCENE and binding['static_result_sha256']==STATIC,
         'EXACT_COMPILED_STATIC_SCENE_BINDING_NOT_PERMISSION')
    nominal_reference(dict(path=binding['nominal_calibration_goal_binding'],sha256=binding['nominal_calibration_goal_binding_sha256']))
    proof=Path(binding['static_result']);model=Path(spec['model_path'])
    need(all(p.is_absolute()and p.resolve()==p and not p.is_symlink()for p in (proof,model))
         and digest(proof)==STATIC and digest(model)==SCENE,'EXACT_PRIVATE_STATIC_AND_MODEL_BYTES')
    value=json.loads(proof.read_bytes())
    need(value['status']=='STATIC_ONLY_PASSED_NEW_NATIVE_ADMISSION_REQUIRED'
         and value['scene_sha256']==SCENE and value['robot_contract']['sha256']==ROBOT
         and value['robot_contract']['controller_or_safety_changes']is False
         and value['robot_contract']['original_dt_s']==.002
         and value['dense_prefix_geometry']['saved_pose_FK_calls']==B
         and value['dense_prefix_geometry']['contact_free_at_saved_poses']is True
         and value['new_native_admission']is False,'STATIC_QUALIFICATION_IS_NOT_DYNAMIC_ADMISSION')
    from .fusion_live_program_replay_v1 import assignment as declared_assignment
    a=declared_assignment(spec);panel=Path(spec['response_learning_panel'])
    need(not panel.is_symlink()and digest(panel)==a['panel_sha256']==spec['response_learning_panel_sha256'],'FROZEN_QUERY_PANEL')
    rows=[r for r in json.loads(panel.read_bytes())['assignments']if r['native_run']==Path(spec['registered_root']).name]
    need(len(rows)==1 and rows[0]['parent']==a['parent']and rows[0]['split']=='DIAGNOSTIC'
         and rows[0]['scene_sha256']==SCENE and rows[0]['task_condition']==CONDITION,'EXACT_NEW_SCENE_PANEL_ROW')
    return binding


def episode(run):
    from .fusion_task_phase_pose_slow_feedback_v9 import make_episode
    return dict(schema=WRAPPER,run=str(run),actual_scene_id=SCENE_ID,actual_scene_sha256=SCENE,
        nominal_calibration_goal_binding_sha256=NOMINAL_REFERENCE_SHA,
        role='NOMINAL_T06_REFERENCE_RECIPE_WITH_SEPARATELY_DECLARED_C01_PHYSICAL_SCENE',
        calibration_prefix_recipe=make_episode(run=str(run),geometry_scale=.5))


def prefix_episode(document,spec):
    from .fusion_task_phase_pose_slow_feedback_v9 import episode_contract
    need(document==episode(spec['registered_root']),'EXACT_ACTUAL_SCENE_WRAPPER_AND_CALIBRATION_RECIPE')
    return episode_contract(document['calibration_prefix_recipe'])


def proposal_geometry(program,cube):
    anchor=program.decision_TCP_target[:3,3];cube=np.asarray(cube)[:3,3]
    ray=cube-anchor;distance=float(np.linalg.norm(ray));direction=ray/distance
    delta=program.desired_path[:,:3]-anchor;signed=delta@direction
    horizons={}
    for name,indices in [('RAW_KNOT',[(k,int(program.knot_indices[k-1]))for k in (1,4,8,16,40)]),
                         ('ADAPTED_ATOM',[(k,min(k,program.end_atom))for k in (1,4,8,16)])]:
        horizons[name]={str(k):dict(proposed_atom=i,proposed_elapsed_s=i*.05,
            signed_progress_m=float(signed[i]),lateral_m=float(np.linalg.norm(delta[i]-signed[i]*direction)),
            cube_distance_reduction_m=float(distance-np.linalg.norm(program.desired_path[i,:3]-cube)),
            finger_opening_m=float(program.desired_path[i,9]*.044))for k,i in indices}
    return dict(schema='WM02_C01_FULL_PROPOSAL_GEOMETRY_NOT_TASK_SCORE_V1',initial_cube_distance_m=distance,
        final_signed_progress_m=float(signed[-1]),final_cube_distance_reduction_m=float(distance-np.linalg.norm(program.desired_path[-1,:3]-cube)),
        maximum_lateral_drift_m=float(np.linalg.norm(delta-signed[:,None]*direction,axis=1).max()),
        maximum_TCP_excursion_m=float(np.linalg.norm(delta,axis=1).max()),
        minimum_desired_or_reference_cube_distance_m=float(min(np.linalg.norm(program.desired_path[:,:3]-cube,axis=1).min(),np.linalg.norm(program.reference_path[:,:3]-cube,axis=1).min())),
        minimum_finger_opening_m=float(program.desired_path[:,9].min()*.044),
        fixed_desired_horizons_not_executed=horizons,full_H40_adaptation=program.evidence(),
        native_execution_authorized=False,actual_task_success_established=False)


def query_only(adapter,report,samples,windows,*,task,camera):
    from . import fusion_paused_query_bridge as bridge
    from . import fusion_paused_inference as inference
    from .fusion_trajectory_adapter import TrajectoryAdaptation
    from .fusion_graded_short_skill_v1 import screen_proposal
    from .fusion_bounded_reference_tracking_v1 import screen
    from .fusion_live_skill_segment import jsonable
    s=adapter.spec;scene(s)
    ready=report['task_phase_initialization']['readiness']
    need(adapter.observe()['step']==B and adapter.trace_sealed is False
         and ready['query_ready']is False and ready['predicates']['full_grade']is False
         and all(v is True for k,v in ready['predicates'].items()if k not in ('full_grade','actual_cube_distance'))
         and len(windows)==(B-2503)//25 and report['all_40_knots_consumed']is True,'ACTUAL_UNCHANGED_STANDING_HOLD_AND_NONCONTACT_PREDICATES_REQUIRED')
    distance=float(np.linalg.norm(np.asarray(samples[-1]['tcp'])[:3,3]-np.asarray(samples[-1]['object_pose'])[:3,3]))
    need(CAPTURE_DISTANCE_M[0]<=distance<=CAPTURE_DISTANCE_M[1],'DISTINCT_80MM_CAPTURE_PHASE_NOT_OLD_INITIALIZATION_OR_ACTION_DOMAIN')
    from .fusion_task_phase_live_skill_native_v8 import _noncontact
    _noncontact(samples[-1],np.asarray(samples[-1]['object_pose']))
    from .fusion_observation import TargetShadowBinding
    from scripts.pi0292.relay_protocol import digest as document_hash
    frozen_spec=document_hash(s)
    binding=TargetShadowBinding(target_world_frame='MUJOCO_GO2_ARX5_WORLD',scene_sha256=digest(s['model_path']),
        target_runtime_sha256=digest(Path(s['registered_root'])/'EXECUTION_SPEC.json'),
        T_source_from_target=s['paused_source_from_target'],expected_transform_sha256=s['paused_frame_map_sha256'])
    query=bridge.capture_and_wait_query(adapter,binding=binding,task=task,task_sha256=document_hash(task),
        gpu_spec_path=Path(s['paused_query_gpu_spec']),capture_index=0,camera_recipe=camera)
    need(document_hash(s)==frozen_spec and adapter.observe()['step']==B and not adapter.trace_sealed,'SAME_UNSEALED_SCOPE_AND_CLOCK_AFTER_QUERY')
    gpu=json.loads(Path(s['paused_query_gpu_spec']).read_bytes());payload,identity,_=inference.read_payload(gpu)
    action=Path(gpu['registered_root'])/'fit-01/REAL_VLA_ACTION.npz'
    result=json.loads((action.parent/'RESULT.private.json').read_bytes())
    need(digest(action)==result['action_sha256']==query['query_result']['action_sha256']and identity==result['observation_binding'],'SAME_RETURNED_QUERY_AND_FULL_ACTION')
    with np.load(action,allow_pickle=False)as data:raw=np.array(data['physical_H40'],copy=True)
    need(raw.shape==(40,10)and np.isfinite(raw).all(),'FULL_ORIGINAL_PHYSICAL_H40_REQUIRED')
    current=np.asarray(samples[-1]['tcp']);cube=np.asarray(samples[-1]['object_pose'])
    # Use the same observed source-frame TCP semantics as the ordinary consumer.
    from .fusion_task_phase_live_skill_native_v8 import matrix10
    source_tcp=matrix10(np.asarray(payload['state_eef10']))
    program=TrajectoryAdaptation(raw,source_tcp,np.asarray(s['paused_source_from_target']),current,
        adapter.reference_history[-1]['pose'],adapter.base_pose_hold,1.,time_stretch_factor=2)
    try:
        strict=screen_proposal(program,cube);mechanical=screen(program,cube)
    except ValueError as error:
        if str(error)!='GRADED_SHORT_START_OUTSIDE_PRECONTACT_REQUIRED':raise
        strict=dict(status='OUTSIDE_ORIGINAL_SHORT_TASK_START_DOMAIN',qualified=False,original_rejection=str(error),score_not_redefined=True)
        mechanical=dict(status='OUTSIDE_ORIGINAL_140MM_DOMAIN',eligible_for_original_native_certificate=False,original_rejection=str(error),domain_not_redefined=True)
    proposal=dict(schema='WM02_C01_FROZEN_GR00T_FULL_PROPOSAL_QUERY_ONLY_V1',native_step=B,
        original_decoded_physical_H40=raw,action_npz_sha256=digest(action),observation_binding=identity,
        adaptation=program.evidence(),strict_short_desired_intent=strict,
        old_140mm_mechanical_domain_screen=mechanical,source_phase_proposal_geometry=proposal_geometry(program,cube),
        native_generated_action_installations=0,generated_action_integrations=0,
        full_H40_consumption=False,new_action_or_contact_qualification=False,
        actual_scene_id=SCENE_ID,actual_scene_sha256=SCENE)
    proposal=jsonable(proposal);write_json(adapter.output/'SOURCE_PHASE_QUERY_ONLY_PROPOSAL.private.json',proposal)
    need(adapter.observe()['step']==B,'QUERY_ONLY_CANNOT_ADVANCE_PHYSICS')
    out=copy.deepcopy(report)
    out.update(schema='WM02_SOURCE_PHASE_C01_NATIVE_INITIALIZATION_AND_QUERY_ONLY_V1',
        actual_scene_id=SCENE_ID,actual_scene_sha256=SCENE,
        source_phase_capture_distance_m=distance,source_phase_capture_distance_bounds_m=CAPTURE_DISTANCE_M,
        original_initialization_readiness_unchanged=True,
        initialization_episode_is_nominal_calibration_recipe=True,
        initialization_all40_knots_consumed=report['all_40_knots_consumed'],
        query_only_no_generated_actions=True,all_40_knots_consumed=False,raw_knot_consumed=[False]*40,
        GR00T_raw_H40_consumed=False,actual_VLA_commands=0,VLA_commands=0,VLA_action_installations=0,
        VLA_queries=1,VLA_inferences=1,actual_VLA_inferences=1,unique_generated_action_chunks=1,
        native_process_Processor_decodes=0,additional_action_decodes=0,paused_query=query,
        query_only_proposal_sha256=digest(adapter.output/'SOURCE_PHASE_QUERY_ONLY_PROPOSAL.private.json'),
        strict_short_desired_intent=proposal['strict_short_desired_intent'],
        source_phase_proposal_geometry=proposal['source_phase_proposal_geometry'],
        old_140mm_mechanical_domain_screen=proposal['old_140mm_mechanical_domain_screen'],
        native_endpoint_step=B,WM_updates=0,GR00T_updates=0,Actor_updates=0,
        full_task_success=False,target_task_success=False,source_task_phase_equivalence=False)
    return jsonable(out)
