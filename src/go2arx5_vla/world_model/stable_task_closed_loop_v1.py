"""Three real-observation learned prefixes in one original native world.

This entry requires real TRAIN timing and an already sealed candidate DEV
experiment. It neither supplies a Teacher fallback nor resets control history.
"""
import copy
import time
from pathlib import Path
import numpy as np
from . import stable_task_prefix_v1 as prefix
from . import stable_task_scene_v1 as scene
from . import stable_task_evaluation_v1 as evaluation
from . import stable_task_policy_v1 as policy
from . import stable_start_short_task_v1 as common
from . import stable_task_dataset_v1 as dataset
from .stable_task_data_v1 import capture_input,task_score_contract
from .storage import write_json,jsonable,digest

PHASE,KIND,B,E=prefix.PHASE,prefix.KIND,prefix.B,prefix.E
need=prefix.need
SOURCE_MEMBERS=tuple(dict.fromkeys(evaluation.SOURCE_MEMBERS+(
    'src/go2arx5_vla/world_model/stable_task_prefix_v1.py',
    'src/go2arx5_vla/world_model/stable_task_prefix_qualification_v1.py',
    'src/go2arx5_vla/world_model/stable_task_closed_loop_v1.py',
    'scripts/wm02/qualify_stable_task_prefix_v1.py',
    'scripts/wm02/stable_task_closed_loop_check_v1.py')))


def validate(spec):
    from .stable_task_prefix_qualification_v1 import SCHEMA,verify_inputs
    prefix.declared(spec);scene.scene(spec)
    candidate=spec.get('stable_task_candidate')or{}
    need(set(candidate)=={'training_result','export'}and spec.get('data_role')=='DEV'
        and spec.get('snapshot_binding_mode')=='PRE_REGISTERED_STABLE_TASK_CLOSED_LOOP_V1'
        and not spec.get('paused_query_pair')and not spec.get('paused_query_gpu_spec')
        and spec.get('live_skill_geometry_scale')==1. and spec.get('live_skill_time_stretch_factor')==2,
        'EXPLICIT_CLOSED_LOOP_SAME_CANDIDATE_AND_NO_QUERY_PAIR')
    q=scene.pinned(spec['stable_task_prefix_qualification'])
    need(q.get('schema')==SCHEMA
        and q.get('status')=='REAL_TRAIN_PREFIX_AND_SEALED_CANDIDATE_DEV_CHECKED_NOT_DYNAMIC_PERMISSION'
        and q.get('candidate')==candidate and q.get('proposed_protocol')==prefix.protocol()
        and q.get('qualified_TRAIN_prefix_native_ticks')==3000 and q.get('actual_numpy_inference_calls')==200
        and q.get('native_integrations')==q.get('optimizer_updates')==q.get('teacher_fallbacks')==0
        and q.get('native_execution_authorized')is False
        and q.get('qualification_module_sha256')==digest(Path(__file__).with_name('stable_task_prefix_qualification_v1.py')),
        'REAL_PINNED_TRAIN_TIMING_AND_CANDIDATE_DEV_NOT_A_DYNAMIC_GRANT')
    panel=scene.pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    need(panel.get('candidate')==candidate and panel.get('closed_loop_protocol')==prefix.protocol()
        and panel.get('prefix_qualification')==spec['stable_task_prefix_qualification'],
        'PANEL_FREEZES_ONE_CANDIDATE_AND_QUALIFICATION_BEFORE_DEV')
    verify_inputs(candidate,q['TRAIN_native_result'],q['DEV_native_result'])
    return candidate


def episode_config(spec):
    validate(spec)
    document=scene.pinned(dict(path=spec['episode_manifest'],sha256=spec['episode_manifest_sha256']))
    episode=scene.prefix_episode(document,spec)
    episode,source=common.operational.recipe_config(spec,episode,actual_phase=PHASE,
        actual_kind=KIND,source_members=SOURCE_MEMBERS)
    return episode,dict(source,stable_task_scene_reference=spec['stable_task_scene_reference'])


def episode_limit(spec,previous):
    episode_config(spec);common.audit_prefix_predecessor(spec,previous)
    return E


def validate_adapter_install(adapter,action):
    from .fusion_trajectory_adapter import TrajectoryAdaptation
    need(adapter.spec.get('phase')==PHASE and adapter.spec.get('kind')==KIND,'ACTUAL_CLOSED_LOOP_PHASE')
    step=adapter.observe()['step'];active=getattr(adapter,'stable_task_active_cycle',None)
    if active is None:
        need(step<B,'CALIBRATION_BEFORE_POLICY')
        return common.prefix.validate_install_preview(action,initial_tcp=adapter.trajectory_initial_tcp,
            program=adapter.trajectory_program,step=step,base=adapter.base_pose_hold,
            frozen_base=adapter.trajectory_initial_base,end_step=B)
    c=prefix.cycle(adapter.spec,active['index']);program=active['program']
    need(all(active[k]==v for k,v in c.items())and type(program)is TrajectoryAdaptation
        and c['start_step']<=step<c['end_step']and(step-c['start_step'])%25==0
        and program.geometry_scale==1. and program.time_stretch_factor==2
        and np.array_equal(action,program.preview((step-c['start_step'])//25))
        and np.array_equal(program.base_pose_hold,adapter.base_pose_hold)
        and np.array_equal(adapter.base_pose_hold,adapter.trajectory_initial_base),
        'EXACT_CURRENT_CYCLE_FULL_CONTROLLER_PREVIEW_AND_FIXED_BASE')
    return dict(status='ORIGINAL_NATIVE_CERTIFICATE_AND_ACK_STILL_REQUIRED',native_step=step,
        cycle_index=c['index'],physics_advance_authorized=False)


def _link_replacement(adapter,previous,current):
    first=current['installs'][0]
    need(first['stage']=='ORIGINAL_INSTALL_ACKED_BEFORE_EFFECT'
        and previous['cycle']['end_step']==current['cycle']['start_step']==first['native_step']
        and first['previous_reference_sha256']==previous['final_reference_sha256']
        and first['wire_sha256']!=previous['final_reference_sha256'],
        'ACTUAL_NEXT_INSTALL_CANCELS_PREVIOUS_SUFFIX_AT_SAME_CLOCK')
    proof=dict(kind='SUPERSEDED_BY_ORIGINAL_NEW_REFERENCE_ACK',native_step=first['native_step'],
        old_reference_sha256=previous['final_reference_sha256'],new_reference_sha256=first['wire_sha256'],
        original_install_ACK=current['original_install_ACKs'][0],
        old_controller_preview_retained=True,future_may_have_influenced_previous_MPC=True)
    previous.update(cancellation_status='SUPERSEDED_BY_ORIGINAL_NEW_REFERENCE_ACK',cancellation=proof)
    path=adapter.output/'STABLE_TASK_CYCLES'/f"{previous['cycle']['index']:02d}"/'REPLACEMENT.private.json'
    need(not path.exists(),'NEVER_OVERWRITE_REPLACEMENT_PROOF');write_json(path,proof)


def continue_native(adapter,report,samples,windows):
    from .fusion_task_phase_live_skill_native_v8 import _noncontact
    from .fusion_task_reference_diagnostic_v1 import require_standing
    from .fusion_paused_observation import verify_still_paused
    from .fusion_short_skill import _score_measured_samples
    candidate=validate(adapter.spec);ready=report['task_phase_initialization']['readiness']
    need(adapter.observe()['step']==B and not adapter.trace_sealed
        and all(v is True for k,v in ready['predicates'].items()if k!='full_grade')
        and report['all_40_knots_consumed']is True and len(windows)==(B-2503)//25,
        'REAL_STABLE_PREFIX_BEFORE_FIRST_POLICY_OBSERVATION')
    samples,windows=list(samples),list(windows);initial=dict(qpos=np.asarray(samples[-1]['qpos']).copy())
    cube=np.asarray(samples[-1]['object_pose']).copy();arrays=policy.load_export(candidate['export'])
    cycles=[];observations=[]
    def observe(sample):
        _noncontact(sample,cube);require_standing(sample,initial)
    for index,step in enumerate(prefix.observation_steps(adapter.spec)):
        need(adapter.observe()['step']==step,'FRESH_REAL_OBSERVATION_AT_REGISTERED_CYCLE')
        observation=capture_input(adapter);observations.append(observation)
        inference_started=time.monotonic()
        delta,inference=policy.predict_from_observation(adapter.spec,observation,arrays,expected_step=step)
        inference['feature_read_and_numpy_inference_wall_s']=time.monotonic()-inference_started
        packet,_=dataset.load(observation['path'],observation['sha256'],maximum=16*1024**2)
        inference['capture_wall_s']=packet.get('capture_wall_s')
        inference['pause_after_inference']=verify_still_paused(adapter,packet)
        program,proposal=policy.build_for_delta(samples[-1]['tcp'],adapter.reference_history[-1]['pose'],
            adapter.base_pose_hold,delta)
        proposal.update(inference=inference,observation=observation,cycle_index=index,
            candidate=candidate,teacher_fallback=False,assistance='NATIVE_QUOTA_PAUSED_REAL_OBSERVATION')
        write_json(adapter.output/f'STABLE_TASK_POLICY_{index:02d}.private.json',proposal)
        current,samples,windows=prefix.run_prefix(adapter,program,index=index,
            past_samples=samples,past_windows=windows,sample_observer=observe)
        if cycles:_link_replacement(adapter,cycles[-1],current)
        cycles.append(current)
    # A deterministic termination HOLD is explicit and is not a fourth query.
    program,proposal=policy.build_for_delta(samples[-1]['tcp'],adapter.reference_history[-1]['pose'],
        adapter.base_pose_hold,np.zeros(3))
    proposal.update(kind='DETERMINISTIC_TERMINATION_HOLD',policy_queries=0,teacher_fallback=False)
    write_json(adapter.output/'STABLE_TASK_TERMINATION_HOLD.private.json',proposal)
    current,samples,windows=prefix.run_prefix(adapter,program,index=3,
        past_samples=samples,past_windows=windows,sample_observer=observe)
    _link_replacement(adapter,cycles[-1],current);cycles.append(current)
    score=_score_measured_samples([s for s in samples if s['native_step']>=B],
        common.score_contract(),expected_contract=common.score_contract())
    stable=all(v for k,v in score['predicates'].items()if k not in('signed_progress','distance_reduction'))
    success=(score['approach_step_success']if adapter.spec['task_command']!='HOLD_OPEN'
        else stable and score['metrics']['maximum_TCP_excursion_m']<=.002)
    total=(E-2503)//25
    return jsonable(dict(copy.deepcopy(report),schema='WM02_STABLE_TASK_CLOSED_LOOP_NATIVE_ACTUAL_V1',
        actual_spec_phase=PHASE,native_endpoint_step=E,complete_task_atoms=total,
        data_role=adapter.spec['data_role'],data_parent=adapter.spec['data_parent'],
        actual_scene_sha256=digest(adapter.spec['model_path']),task_native_ticks=E-2503,
        solver_target_confirmed_atoms=total,gripper_confirmed_atoms=total,windows=windows,
        initialization_all40_knots_consumed=report['all_40_knots_consumed'],
        all_40_knots_consumed=False,raw_knot_consumed=[False]*40,GR00T_raw_H40_consumed=False,
        generated_cycles=cycles,policy_observations=observations,policy_queries=3,
        TASK_ADAPTER_inferences=3,TASK_ADAPTER_updates=0,GR00T_updates=0,WM_updates=0,Actor_updates=0,
        candidate=candidate,stable_task_closed_loop=prefix.protocol(),native_clock_rebased=False,
        online_learned_policy=True,policy_native_ticks=E-B,
        calibration_prefix=dict(kind='UNCHANGED_HALF_CALIBRATION_BEFORE_POLICY',
            privileged_calibration_geometry=True,native_ticks=B-2503,policy_queries=0),
        teacher_fallbacks=0,real_time_deployment=False,assistance='NATIVE_QUOTA_PAUSED_REAL_OBSERVATION',
        full_MPC_hidden_state_restored=False,source_task_phase_equivalence=False,
        task_condition_score_contract=task_score_contract(adapter.spec['task_command']),
        capability_score=score,task_condition_success=bool(success),expert_training_eligible=False,
        full_task_success=False,target_task_success=False,raw_reference_is_GR00T=False,
        final_HOLD_cancel_status='REQUIRES_OUTER_ORIGINAL_CANCEL_AND_FINAL_TRACE'))


def complete_consumption(spec,complete):
    prefix.declared(spec);cycles=complete.get('generated_cycles')or[]
    if not(len(cycles)==4 and complete.get('stable_task_closed_loop')==prefix.protocol()
        and complete.get('policy_queries')==complete.get('TASK_ADAPTER_inferences')==3
        and complete.get('teacher_fallbacks')==0 and complete.get('GR00T_raw_H40_consumed')is False
        and complete.get('all_40_knots_consumed')is False):return False
    observations=complete.get('policy_observations')or[]
    if [o.get('native_step')for o in observations]!=list(prefix.observation_steps(spec)):return False
    for i,row in enumerate(cycles):
        c=prefix.cycle(spec,i)
        if not(row.get('cycle')==c and row.get('complete_suffix_atoms')==len(row.get('windows',[]))==c['atoms']
            and row.get('solver_target_confirmed_atoms')==row.get('gripper_confirmed_atoms')==c['atoms']):return False
        rows=row['windows'];offset=(c['start_step']-2503)//25
        if rows!=complete['windows'][offset:offset+c['atoms']]:return False
        for j,w in enumerate(rows):
            grip=w.get('gripper')or{}
            if not(w.get('native_start_step')==c['start_step']+25*j
                and w.get('native_end_step')==c['start_step']+25*(j+1)and w.get('actual_ticks')==25
                and w.get('solver_policies_with_exact_installed_target')
                and grip.get('command_consumed')is True
                and (grip.get('submitted')or{}).get('sequence')==offset+j):return False
        indices=row['adaptation']['clock']['knot_indices']
        if row.get('raw_knot_consumed')!=[int(k)<c['atoms']for k in indices]:return False
        if i<3:
            if row.get('cancellation_status')!='SUPERSEDED_BY_ORIGINAL_NEW_REFERENCE_ACK':return False
            replacement=row.get('cancellation')or{};following=cycles[i+1]['installs'][0]
            if not(replacement.get('native_step')==c['end_step']==following['native_step']
                and replacement.get('old_reference_sha256')==row['final_reference_sha256']==following['previous_reference_sha256']
                and replacement.get('new_reference_sha256')==following['wire_sha256']):return False
    return True


def run(adapter):
    from .fusion_task_phase_xyz_feedback_native_v4 import run_feedback_program
    episode,source=episode_config(adapter.spec)
    return run_feedback_program(adapter,episode,source,recipe_module=common.prefix,actual_phase=PHASE,
        report_schema='WM02_CLOSED_LOOP_CALIBRATION_PREFIX_V1',
        goal_schema='WM02_CLOSED_LOOP_CALIBRATION_GOAL_V1',
        decision_schema='WM02_CLOSED_LOOP_CALIBRATION_DECISION_V1',
        feedback_report_schema='WM02_CLOSED_LOOP_CALIBRATION_FEEDBACK_V1',continuation=continue_native)
