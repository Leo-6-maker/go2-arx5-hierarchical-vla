"""New contact-task construction on the original single-world native loop."""
import copy
from pathlib import Path
import numpy as np
from . import task_learning_grasp_v1 as task
from . import stable_task_scene_v1 as scenes
from . import stable_task_prefix_v1 as consumer
from . import stable_start_short_task_v1 as common
from . import target_teacher_prefix_diagnostic_v1 as previous
from .storage import digest,write_json,jsonable

SOURCE_MEMBERS=tuple(dict.fromkeys(previous.SOURCE_MEMBERS+(
    'src/go2arx5_vla/world_model/task_learning_grasp_v1.py',
    'src/go2arx5_vla/world_model/task_learning_grasp_native_v1.py')))


def scene_admission(spec):
    task.declared(spec)
    proof,_=scenes.reference_geometry(spec['stable_task_scene_reference'],digest(spec['model_path']))
    task.need(proof['split']==spec.get('data_role')=='TRAIN' and proof['parent']==spec['data_parent']
        and spec['scene_id']=='STABLE_TASK_'+proof['layout'],'ORIGINAL_TRAIN_GEOMETRY_CONSTRUCTION_PILOT')
    panel=scenes.pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    task.need(panel.get('schema')=='TASK_LEARNING_GRASP_CONSTRUCTION_PANEL_V1'
        and panel.get('protocol')==task.protocol() and len(panel['assignments'])==1,'FROZEN_SINGLE_CONSTRUCTION_PILOT')
    row=panel['assignments'][0]
    task.need(row['native_run']==Path(spec['registered_root']).name and row['parent']==proof['parent']
        and row['model_sha256']==digest(spec['model_path']) and row['command']==task.COMMAND
        and row['purpose']=='TASK_CONSTRUCTION_NOT_TEACHER_QUALIFICATION_OR_BC','NEW_TASK_SCENE_RUN_JOIN')
    return proof


def validate(spec):
    scene_admission(spec)
    task.need(spec.get('experiment_role')=='TASK_LEARNING_TASK_CONSTRUCTION_NO_BC'
        and spec.get('snapshot_binding_mode')=='PRE_REGISTERED_TASK_LEARNING_GRASP_V1'
        and spec.get('live_skill_geometry_scale')==1. and spec.get('live_skill_time_stretch_factor')==1
        and not any(spec.get(k)for k in ('target_gr00t_collection','target_gr00t_policy',
            'target_gr00t_gpu_root','paused_query_pair','paused_query_gpu_spec',
            'stable_task_candidate','stable_task_evaluation','program_reference_selection')),
        'NEW_CONTACT_TASK_WITHOUT_NEURAL_OR_PRIVILEGED_STUDENT_INPUTS')


def episode_config(spec):
    validate(spec)
    document=scenes.pinned(dict(path=spec['episode_manifest'],sha256=spec['episode_manifest_sha256']))
    episode=scenes.prefix_episode(document,spec)
    episode,source=common.operational.recipe_config(spec,episode,actual_phase=task.PHASE,
        actual_kind=task.KIND,source_members=SOURCE_MEMBERS)
    return episode,dict(source,stable_task_scene_reference=spec['stable_task_scene_reference'])


def episode_limit(spec,prior):
    episode_config(spec);common.audit_prefix_predecessor(spec,prior)
    return task.E


def validate_adapter_install(adapter,action):
    validate(adapter.spec);step=adapter.observe()['step']
    active=getattr(adapter,'stable_task_active_cycle',None)
    if active is None:
        task.need(step<task.B,'ORIGINAL_CALIBRATION_PRECEDES_NEW_TASK')
        return common.prefix.validate_install_preview(action,initial_tcp=adapter.trajectory_initial_tcp,
            program=adapter.trajectory_program,step=step,base=adapter.base_pose_hold,
            frozen_base=adapter.trajectory_initial_base,end_step=task.B)
    c=consumer.cycle(adapter.spec,active['index']);program=active['program']
    task.need(type(program)is task.FiftyMsProgram and all(active[k]==v for k,v in c.items())
        and c['start_step']<=step<c['end_step'] and (step-c['start_step'])%25==0
        and np.array_equal(action,program.preview((step-c['start_step'])//25))
        and np.array_equal(program.base_pose_hold,adapter.base_pose_hold)
        and np.array_equal(adapter.base_pose_hold,adapter.trajectory_initial_base),
        'EXACT_FULL40_NATIVE_50MS_REFERENCE_AND_BASE_HOLD')
    program.validate_envelope()
    return dict(status='ORIGINAL_NATIVE_CERTIFICATE_AND_ACK_REQUIRED',native_step=step,
        current_Teacher_stage=getattr(adapter,'task_learning_stage','PREGRASP'),physics_advance_authorized=False)


def continue_native(adapter,report,samples,windows):
    from .fusion_task_reference_diagnostic_v1 import require_standing
    from .stable_task_closed_loop_v1 import _link_replacement
    from .stable_task_data_v1 import capture_input
    from .target_gr00t_collection_v1 import require_start_readiness
    validate(adapter.spec);ready=report['task_phase_initialization']['readiness']
    task.need(adapter.observe()['step']==task.B and not adapter.trace_sealed
        and all(v is True for k,v in ready['predicates'].items()if k!='full_grade')
        and report['all_40_knots_consumed']is True,'ORIGINAL_STABLE_PREFIX_BEFORE_CONTACT_TASK')
    start_ready=require_start_readiness(adapter,samples[-1])
    samples,windows=list(samples),list(windows)
    initial=dict(qpos=np.asarray(samples[-1]['qpos']).copy())
    teacher=task.Teacher(samples[-1]);cycles=[];decisions=[];captures=[];confirmed=[]
    task_samples=[samples[-1]]
    def observe(s):
        require_standing(s,initial)
        task.need(all(s.get(k)is True for k in ('safety_valid','contact_valid','gripper_state_valid','object_present'))
            and not any(s.get(k)for k in ('native_abort','illegal_contact','joint_limit_penetration','reset','takeover')),
            'UNCHANGED_NATIVE_CONTACT_AND_STANDING_SAFETY')
        if s['native_step']>task_samples[-1]['native_step']:task_samples.append(copy.deepcopy(s))
    for index,step in enumerate(consumer.observation_steps(adapter.spec)):
        task.need(adapter.observe()['step']==samples[-1]['native_step']==step,'REAL_CONTINUOUS_TEACHER_CLOCK')
        observe(samples[-1])
        capture=capture_input(adapter,language=task.LANGUAGE,task_score=task.score_contract())
        captures.append(capture)
        write_json(adapter.output/'TASK_LEARNING_CAPTURES.private.json',dict(records=captures,complete=False))
        program,proposal=teacher.decide(samples[-1],adapter.reference_history[-1]['pose'],adapter.base_pose_hold)
        adapter.task_learning_stage=teacher.stage
        decision_path=adapter.output/f'TASK_LEARNING_DECISION_{index:03d}.private.json'
        write_json(decision_path,dict(proposal,native_step=step,observation=capture,
            original_actual_TCP=np.asarray(samples[-1]['tcp']).tolist(),adaptation=program.evidence(),
            privileged_Teacher_metadata_not_student_input=True,expert_training_eligible=False))
        decisions.append(dict(path=str(decision_path),sha256=digest(decision_path),native_step=step,stage=teacher.stage))
        current,samples,windows=consumer.run_prefix(adapter,program,index=index,
            past_samples=samples,past_windows=windows,sample_observer=observe)
        if cycles:_link_replacement(adapter,cycles[-1],current)
        cycles.append(current)
        for offset,w in enumerate(current['windows']):
            task.need(np.array_equal(np.asarray(w['actual_installed_consumed_EEF10_BASE7'])[:10],
                program.reference_path[offset]),'EVERY_TRAINABLE_COMMAND_ACTUALLY_INSTALLED_AND_CONSUMED')
            confirmed.append(dict(native_start_step=w['native_start_step'],native_end_step=w['native_end_step'],
                policy_EEF10=program.decoded_H40[offset].tolist(),
                installed_consumed_EEF10=program.reference_path[offset].tolist(),
                Teacher_decision_sha256=digest(decision_path),solver_gripper_confirmed=True))
        write_json(adapter.output/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json',dict(records=confirmed,
            period_s=.05,complete=False,unexecuted_suffix_included=False,future_achieved_labels=False))
    # Termination continues the last explicit Teacher grasp/hold command.
    raw=np.repeat(program.decoded_H40[-1:,:],40,axis=0)
    raw[:,:3]=np.asarray(samples[-1]['tcp'])[:3,3]
    raw[:,3:9]=task.rotation_matrix_to_rot6d_rows(np.asarray(samples[-1]['tcp'])[:3,:3])
    hold=task.FiftyMsProgram(raw,samples[-1]['tcp'],adapter.reference_history[-1]['pose'],adapter.base_pose_hold)
    current,samples,windows=consumer.run_prefix(adapter,hold,index=task.QUERIES,
        past_samples=samples,past_windows=windows,sample_observer=observe)
    _link_replacement(adapter,cycles[-1],current);cycles.append(current)
    scored=task.score(task_samples,teacher.cube[2,3])
    write_json(adapter.output/'TASK_LEARNING_SCORE.private.json',scored)
    write_json(adapter.output/'TASK_LEARNING_CAPTURES.private.json',dict(records=captures,complete=True,task_success=scored['task_success']))
    write_json(adapter.output/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json',dict(records=confirmed,
        period_s=.05,complete=True,unexecuted_suffix_included=False,future_achieved_labels=False,
        task_success=scored['task_success'],expert_training_eligible=False))
    return jsonable(dict(copy.deepcopy(report),schema='TASK_LEARNING_GRASP_CONSTRUCTION_NATIVE_ACTUAL_V1',
        actual_spec_phase=task.PHASE,native_endpoint_step=task.E,task_learning_grasp=task.protocol(),
        teacher_decisions=decisions,generated_cycles=cycles,windows=windows,
        initialization_all40_knots_consumed=True,all_40_knots_consumed=False,raw_knot_consumed=[False]*40,
        GR00T_raw_H40_consumed=False,complete_task_atoms=len(windows),task_native_ticks=task.E-2503,
        solver_target_confirmed_atoms=len(windows),gripper_confirmed_atoms=len(windows),
        new_task_score=scored,task_condition_success=scored['task_success'],
        decision_start_readiness=start_ready,expert_training_eligible=False,
        data_role='TASK_CONSTRUCTION_NOT_BC',GR00T_queries=0,VLA_queries=0,GR00T_updates=0,
        WM_updates=0,Actor_updates=0,full_task_success=False,target_task_success=False,
        new_grasp_lift_hold_task_success=scored['task_success'],source_task_phase_equivalence=False))


def complete_consumption(spec,result):
    task.declared(spec);cycles=result.get('generated_cycles')or[]
    if not(result.get('schema')=='TASK_LEARNING_GRASP_CONSTRUCTION_NATIVE_ACTUAL_V1'
        and len(cycles)==task.QUERIES+1 and len(result.get('teacher_decisions',[]))==task.QUERIES
        and result.get('expert_training_eligible')is False):return False
    for index,row in enumerate(cycles):
        c=consumer.cycle(spec,index);rows=row.get('windows')or[];offset=(c['start_step']-2503)//25
        if not(row['cycle']==c and len(rows)==row.get('complete_suffix_atoms')==c['atoms']
            and row.get('solver_target_confirmed_atoms')==row.get('gripper_confirmed_atoms')==c['atoms']
            and rows==result['windows'][offset:offset+c['atoms']]):return False
        for j,w in enumerate(rows):
            if not(w['native_start_step']==c['start_step']+25*j and w['native_end_step']==c['start_step']+25*(j+1)
                and w['actual_ticks']==25 and w.get('solver_policies_with_exact_installed_target')
                and w['gripper']['command_consumed']is True):return False
        if index<task.QUERIES and row.get('cancellation_status')!='SUPERSEDED_BY_ORIGINAL_NEW_REFERENCE_ACK':return False
    return True


def run(adapter):
    from .fusion_task_phase_xyz_feedback_native_v4 import run_feedback_program
    episode,source=episode_config(adapter.spec)
    return run_feedback_program(adapter,episode,source,recipe_module=common.prefix,actual_phase=task.PHASE,
        report_schema='TASK_LEARNING_GRASP_CALIBRATION_V1',goal_schema='TASK_LEARNING_GRASP_GOAL_V1',
        decision_schema='TASK_LEARNING_GRASP_PREFIX_DECISION_V1',feedback_report_schema='TASK_LEARNING_GRASP_PREFIX_FEEDBACK_V1',
        continuation=continue_native)
