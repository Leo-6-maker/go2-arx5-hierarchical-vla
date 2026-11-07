"""One actual frozen query then its entire qualified H40 in the same world.

This is paused-assisted graded diagnosis, not source-phase equivalence,
non-paused deployment, a WM correction, expert supervision or full-task PASS.
"""
import copy
import time
from pathlib import Path
import numpy as np

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import digest as document_hash
from .fusion_observation import TargetShadowBinding
from . import fusion_task_phase_pose_continuation_v7 as prefix
from .fusion_task_phase_live_skill_contract_v8 import PHASE,KIND,B,E,ROLE,need,episode_config
from .fusion_trajectory_adapter import TrajectoryAdaptation
from .storage import digest,write_json

def validate_adapter_install(adapter,action):
    need(adapter.spec.get('phase')==PHASE and adapter.spec.get('kind')==KIND,
         'ACTUAL_TYPED_LIVE_NATIVE_SCOPE')
    step=adapter.observe()['step'];program=getattr(adapter,'live_skill_program',None)
    if program is None:
        need(step<B,'NO_SUFFIX_INSTALL_BEFORE_SAVED_QUERY_AND_FULL_PROPOSAL')
        return prefix.validate_install_preview(action,initial_tcp=adapter.trajectory_initial_tcp,
            program=adapter.trajectory_program,step=step,base=adapter.base_pose_hold,
            frozen_base=adapter.trajectory_initial_base,end_step=B)
    need(type(program)is TrajectoryAdaptation and B<=step<E and (step-B)%25==0
         and np.array_equal(np.asarray(action),program.preview((step-B)//25))
         and np.array_equal(program.base_pose_hold,np.asarray(adapter.base_pose_hold))
         and np.array_equal(np.asarray(adapter.base_pose_hold),adapter.trajectory_initial_base),
         'WHOLE_LIVE_SUFFIX_EXACT_LOCAL_CLOCK_AND_UNCHANGED_BASE')
    return dict(status='EXACT_GENERATED_FULL_H40_SUFFIX_NOT_NATIVE_QUALIFIED',native_step=step,
        points_checked=40,original_certify_chunk_required=True,original_install_ACK_required=True,
        physics_advance_authorized=False)

def _noncontact(sample,cube):
    need(sample.get('safety_valid')is True and sample.get('native_abort')is False
         and sample.get('robot_object_contact')is False
         and sample.get('left_object_contact')is False and sample.get('right_object_contact')is False
         and sample.get('native_grasp')is False and sample.get('gripper_state_valid')is True
         and sample.get('object_present')is True,
         'ACTUAL_GRADED_DIAGNOSTIC_NONCONTACT_MEASUREMENT')
    need(np.allclose(np.asarray(sample['object_pose']),cube,rtol=0,atol=1e-9),
         'ACTUAL_OBJECT_UNMOVED_DIAGNOSTIC_NOT_CONTACT_SCOPE')

def run(adapter):
    from .fusion_task_phase_xyz_feedback_native_v4 import run_feedback_program
    episode,source,task,camera=episode_config(adapter.spec)
    def continuation(current,report,samples,windows):
        return _continue(current,report,samples,windows,task=task,camera=camera)
    return run_feedback_program(adapter,episode,source,recipe_module=prefix,actual_phase=PHASE,
        report_schema='WM02_NATIVE_GRADED_INITIALIZATION_PREFIX_LIVE_V8',
        goal_schema='WM02_NATIVE_GRADED_INITIALIZATION_GOAL_BINDING_LIVE_V8',
        decision_schema='WM02_NATIVE_FINITE_POSE_FEEDBACK_DECISION_LIVE_V8',
        feedback_report_schema='WM02_NATIVE_FINITE_POSE_FEEDBACK_REPORT_LIVE_V8',
        continuation=continuation)

def _continue(adapter,report,samples,windows,*,task,camera):
    from .fusion_paused_query_bridge import capture_and_wait_query
    from .fusion_paused_inference import read_payload
    from .fusion_live_skill_segment import run_consumed_segment
    spec=adapter.spec;start,cpu=time.monotonic(),time.process_time()
    identity=(spec.get('phase'),spec.get('kind'))
    need(identity in ((PHASE,KIND),
        ('collect-target-fusion-task-phase-live-skill-v9','FUSION_TASK_PHASE_LIVE_SKILL_V9')),
        'EXACT_VERSIONED_GENERATED_SKILL_CONSUMER')
    version='V9' if identity[0].endswith('-v9') else 'V8'
    factor=spec.get('live_skill_time_stretch_factor') if version=='V9' else 2
    need(type(factor)is int and factor in(1,2),
         'EXPLICIT_PREREGISTERED_FULL_H40_TIME_MAPPING')
    is_short=False;is_tracking=False;is_mechanical=False
    if version=='V9':
        from .fusion_task_phase_live_skill_contract_v9 import TASK_CONDITIONS,SHORT_LANGUAGE_TASK_CONDITIONS,TRACKING_TASK_CONDITION,MECHANICAL_TASK_CONDITION
        need(spec.get('task_condition')in TASK_CONDITIONS,'EXPLICIT_V9_FULL_OR_DISTINCT_SHORT_TASK')
        is_short=spec['task_condition']in SHORT_LANGUAGE_TASK_CONDITIONS
        is_tracking=spec['task_condition']==TRACKING_TASK_CONDITION
        is_mechanical=spec['task_condition']==MECHANICAL_TASK_CONDITION
    frozen_spec=document_hash(spec)
    ready=report['task_phase_initialization']['readiness']
    need(adapter.observe()['step']==B and not adapter.trace_sealed
         and ready['query_ready']is False and ready['predicates']['full_grade']is False
         and all(v is True for k,v in ready['predicates'].items()if k!='full_grade')
         and len(windows)==(B-2503)//25 and report['all_40_knots_consumed']is True,
         'ACTUAL_MEASURED_NINE_HALF_PREDICATES_NOT_RELABELLED_FULL_QUERY_READY')
    cube=np.asarray(samples[-1]['object_pose'],float).copy()
    _noncontact(samples[-1],cube)
    binding=TargetShadowBinding(target_world_frame='MUJOCO_GO2_ARX5_WORLD',
        scene_sha256=digest(spec['model_path']),
        target_runtime_sha256=digest(Path(spec['registered_root'])/'EXECUTION_SPEC.json'),
        T_source_from_target=spec['paused_source_from_target'],
        expected_transform_sha256=spec['paused_frame_map_sha256'])
    query=None;proposal=None
    teacher_reference=None;unused_GR00T_program=None;unused_GR00T_intent=None
    try:
        query=capture_and_wait_query(adapter,binding=binding,task=task,task_sha256=document_hash(task),
            gpu_spec_path=Path(spec['paused_query_gpu_spec']),capture_index=0,camera_recipe=camera)
        need(document_hash(spec)==frozen_spec and adapter.observe()['step']==B and not adapter.trace_sealed,
             'SAME_UNSEALED_WORLD_SPEC_AND_CLOCK_AFTER_GPU_QUERY')
        gpu=common_json(spec['paused_query_gpu_spec'])
        payload,identity,_=read_payload(gpu)
        output=Path(gpu['registered_root'])/'fit-01'
        result=common_json(output/'RESULT.private.json')
        action=output/'REAL_VLA_ACTION.npz'
        need(digest(action)==result['action_sha256']==query['query_result']['action_sha256']
             and identity==result['observation_binding'], 'SAME_RETURNED_PHYSICAL_H40_AND_QUERY_PAYLOAD')
        with np.load(action,allow_pickle=False)as saved:
            raw=saved['physical_H40'].copy()
        actual=np.asarray(samples[-1]['tcp'],float)
        anchor=np.asarray(adapter.reference_history[-1]['pose'],float)
        if spec.get('program_reference_intervention') is not None:
            from .fusion_program_reference_intervention_v1 import validate_scope
            validate_scope(spec)
        program=TrajectoryAdaptation(raw,matrix10(payload['state_eef10']),
            spec['paused_source_from_target'],actual,anchor,adapter.base_pose_hold,
            1.,time_stretch_factor=factor,
            reference_intervention=spec.get('program_reference_intervention'))
        proposal=dict(schema='WM02_ACTUAL_GENERATED_LIVE_PROPOSAL_'+version,
            query=query,action_npz_sha256=digest(action),observation_identity=identity,
            original_decoded_physical_H40=raw.tolist(),adaptation=program.evidence(),
            raw_knots=40,geometry_scale=1.,time_stretch_factor=factor,clips=0,
            original_Processor_extra_decodes=0,native_step=B,pre_effect=True,
            actual_contact_domain='NONCONTACT_ONLY_NOT_GRASP_OR_RELEASE',
            cube_ground_truth_use='DIAGNOSTIC_SAFETY_ENVELOPE_NOT_MODEL_OR_GR00T_INPUT',
            policy_input_distribution_equivalence=False)
        write_json(adapter.output/'LIVE_SKILL_GENERATED_PROPOSAL.private.json',proposal)
        if spec.get('task_reference_diagnostic') is not None:
            from .fusion_task_reference_diagnostic_v1 import build,validate_registration
            from .fusion_graded_short_skill_v1 import screen_proposal
            validate_registration(spec)
            unused_GR00T_program=program
            unused_GR00T_intent=screen_proposal(program,cube)
            program,teacher_reference=build(actual_TCP=actual,initial_reference_TCP=anchor,
                base_pose_hold=adapter.base_pose_hold,actual_object_pose=cube)
            write_json(adapter.output/'GEOMETRIC_TASK_REFERENCE_PROPOSAL.private.json',teacher_reference)
        need((E-B)//25>=program.end_atom+8,
             'WHOLE_H40_AND_TAIL_MUST_FIT_PREREGISTERED_E_NO_PREFIX_SELECTION')
        clearance=np.linalg.norm(program.desired_path[:,:3]-cube[:3,3],axis=1)
        reference_clearance=np.linalg.norm(program.reference_path[:,:3]-cube[:3,3],axis=1)
        need(float(min(clearance.min(),reference_clearance.min()))>=.14,
             'FULL_PROPOSAL_OUTSIDE_PREDECLARED_NONCONTACT_ENVELOPE_KEEP_ALL40_REJECT_NO_CLIP')
        short_intent=None;tracking_intent=None;mechanical_screen=None
        if is_short:
            from .fusion_graded_short_skill_v1 import screen_proposal
            short_intent=screen_proposal(program,cube)
            write_json(adapter.output/'LIVE_SKILL_SHORT_INTENT_SCREEN.private.json',short_intent)
            if is_mechanical:
                from .fusion_bounded_reference_tracking_v1 import screen
                mechanical_screen=screen(program,cube)
                write_json(adapter.output/'LIVE_SKILL_MECHANICAL_SCREEN.private.json',mechanical_screen)
                need(mechanical_screen['eligible_for_original_native_certificate'],
                     'WHOLE_GENERATED_MECHANICAL_DOMAIN_REJECT_KEEP_ALL40_NO_INSTALL')
            elif is_tracking:
                from .fusion_graded_short_skill_v1 import tracking_diagnostic_screen
                tracking_intent=tracking_diagnostic_screen(program,cube)
                write_json(adapter.output/'LIVE_SKILL_TRACKING_DIAGNOSTIC_SCREEN.private.json',tracking_intent)
                need(tracking_intent['eligible_for_original_native_certificate'],
                     'WHOLE_GENERATED_TRACKING_DIAGNOSTIC_REJECT_KEEP_ALL40_NO_INSTALL')
            else:
                need(short_intent['qualified'],'WHOLE_GENERATED_SHORT_INTENT_REJECT_KEEP_ALL40_NO_INSTALL')
        adapter.live_skill_program=program
        suffix,all_samples,all_windows=run_consumed_segment(adapter,program,start_step=B,end_step=E,
            past_samples=samples,past_windows=windows,sample_observer=lambda s:_noncontact(s,cube))
        if spec.get('program_reference_selection') is not None:
            selected=adapter.live_skill_program
            need(type(selected)is type(program)
                 and selected.evidence()==suffix['adaptation']
                 and np.array_equal(selected.decoded_H40,program.decoded_H40)
                 and np.array_equal(selected.desired_path,program.desired_path)
                 and selected.end_atom==program.end_atom,
                 'ACTUAL_SELECTED_REFERENCE_REPORT_MUST_PRESERVE_FULL_INTENT_AND_CLOCK')
            program=selected
        total=(E-2503)//25
        rows=all_windows[len(windows):]
        tracking=[float(np.linalg.norm(np.asarray(r['actual_TCP'])[:3,3]
                   -matrix10(np.asarray(r['desired_interval_end_EEF10']))[:3,3]))for r in rows]
        final=np.asarray(all_samples[-1]['tcp'])
        desired_final=matrix10(program.desired_at(program.end_atom))
        reference_final=matrix10(program.preview(program.end_atom)[0])
        complete=dict(copy.deepcopy(report),schema='WM02_NATIVE_GENERATED_GRADED_LIVE_SKILL_'+version,
            actual_spec_phase=spec['phase'],experiment_role=ROLE,native_endpoint_step=E,
            desired_outcome_provenance='ACTUAL_FROZEN_VLA_PROPOSAL_NOT_EXPERT_LABEL',
            episode_role=('SLOW_V9_INITIALIZATION_PREFIX_ONLY' if version=='V9'
                          else 'UNCHANGED_V7_INITIALIZATION_PREFIX_ONLY'),
            initialization_adaptation=report.get('adaptation'),adaptation=program.evidence(),
            initialization_raw_knot_consumed=report.get('raw_knot_consumed'),
            finite_feedback_stage='INITIALIZATION_PREFIX_ONLY',
            original_install_ACKs=list(report.get('original_install_ACKs',[]))+suffix['original_install_ACKs'],
            actual_endpoint_TCP=final.tolist(),
            actual_minus_desired_xyz_m=(final[:3,3]-desired_final[:3,3]).tolist(),
            actual_minus_reference_xyz_m=(final[:3,3]-reference_final[:3,3]).tolist(),
            task_native_ticks=E-2503,post_decision_atoms=(E-3703)//25,
            tail_hold_atoms=8+suffix['extra_final_hold_atoms'],minimum_tail_hold_atoms=8,
            complete_task_atoms=total,solver_target_confirmed_atoms=total,gripper_confirmed_atoms=total,
            windows=all_windows,initialization_all40_knots_consumed=report['all_40_knots_consumed'],
            all_40_knots_consumed=suffix['all_40_knots_consumed'],raw_knot_consumed=suffix['raw_knot_consumed'],
            paused_query=query,generated_proposal_path=str(adapter.output/'LIVE_SKILL_GENERATED_PROPOSAL.private.json'),
            generated_proposal_sha256=digest(adapter.output/'LIVE_SKILL_GENERATED_PROPOSAL.private.json'),
            generated_segment=suffix,actual_VLA_inferences=1,VLA_inferences=1,VLA_queries=1,
            actual_VLA_commands=1,VLA_commands=1,unique_generated_action_chunks=1,
            VLA_action_installations=len(suffix['install_attempts']),
            native_process_Processor_decodes=0,additional_action_decodes=0,
            GPU_action_decode_contract='ONE_ORIGINAL_PROCESSOR_DECODE_INSIDE_RESIDENT_GENERATE',
            native_integrations_by_query=0,WM_updates=0,Actor_updates=0,GR00T_updates=0,
            max_measured_generated_tracking_position_error_m=max(tracking),
            full_task_success=False,target_task_success=False,source_task_phase_equivalence=False,
            paused_assisted=True,online_VLA=False,online_closed_loop=False,
            feedback_state106_contains_privileged_simulator_fields_not_deployable_policy_input=True,
            query_and_suffix_wall_s=time.monotonic()-start,query_and_suffix_CPU_s=time.process_time()-cpu)
        if program.reference_intervention is not None:
            hold=program.reference_intervention.candidate=='HOLD'
            complete.update(reference_intervention=program.reference_intervention.document(),
                generated_raw_H40_intent_retained=True,post_query_HOLD_control=hold,
                actual_VLA_commands=0 if hold else 1,VLA_commands=0 if hold else 1,
                VLA_action_installations=0 if hold else len(suffix['install_attempts']),
                actual_reference_installations=len(suffix['install_attempts']),
                full_H40_intent_is_an_executed_skill=not hold)
        if teacher_reference is not None:
            complete.update(desired_outcome_provenance=teacher_reference['producer'],
                task_reference_diagnostic=spec['task_reference_diagnostic'],
                task_reference_producer=teacher_reference['producer'],
                executed_reference_proposal_sha256=digest(adapter.output/'GEOMETRIC_TASK_REFERENCE_PROPOSAL.private.json'),
                unused_GR00T_proposal_retained=True,unused_GR00T_adaptation=unused_GR00T_program.evidence(),
                unused_GR00T_short_intent_screen=unused_GR00T_intent,
                all_40_knots_consumed=False,raw_knot_consumed=[False]*40,
                GR00T_raw_H40_consumed=False,Teacher_reference_all40_consumed=suffix['all_40_knots_consumed'],
                Teacher_reference_commands=1,actual_VLA_commands=0,VLA_commands=0,VLA_action_installations=0,
                actual_reference_installations=len(suffix['install_attempts']),
                full_H40_intent_is_an_executed_skill=False,
                privileged_diagnostic_not_deployable_VLA=True,GR00T_or_WM_task_transfer=False)
        for k in ('original_Processor_decodes','Processor_decodes'):complete.pop(k,None)
        if is_short:
            from . import fusion_graded_short_skill_v1 as short
            complete.update(short_task_condition=spec['task_condition'],
                short_intent_screen=short_intent,
                graded_short_skill_score=short.score(
                    [sample for sample in all_samples if sample['native_step']>=B],short.score_contract()))
            if is_tracking:
                complete.update(tracking_diagnostic_screen=tracking_intent,
                    exploratory_tracking_diagnostic=True,strict_short_proposal_failure_retained=not short_intent['qualified'],
                    strict_short_score_is_diagnostic_not_formal_task_evaluation=True)
            if is_mechanical:
                complete.update(mechanical_screen=mechanical_screen,
                    engineering_full_H40_consumption_only=not complete.get('post_query_HOLD_control',False),
                    strict_short_proposal_failure_retained=not short_intent['qualified'],
                    strict_short_score_is_diagnostic_not_formal_task_evaluation=True)
        return complete
    except BaseException as error:
        write_json(adapter.output/'LIVE_SKILL_FAILED.private.json',dict(
            schema='WM02_GRADED_LIVE_SKILL_FAILURE_'+version,phase=spec['phase'],native_step=adapter.observe()['step'],
            failure=type(error).__name__+': '+str(error),query=query,
            proposal_saved=proposal is not None,automatic_retry=False,full_task_success=False,
            wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu))
        raise

def common_json(path):
    import json
    path=Path(path)
    need(not path.is_symlink(),'CANONICAL_ACTUAL_JSON_FILE')
    return json.loads(path.read_bytes())
