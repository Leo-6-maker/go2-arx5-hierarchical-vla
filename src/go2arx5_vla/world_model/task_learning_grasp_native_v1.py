"""New contact-task construction on the original single-world native loop."""
import copy
from pathlib import Path
import numpy as np
from scripts.pi0292.durable import write as write_progress
from . import task_learning_grasp_v1 as task
from . import stable_task_scene_v1 as scenes
from . import stable_task_prefix_v1 as consumer
from . import stable_start_short_task_v1 as common
from . import target_teacher_prefix_diagnostic_v1 as previous
from .storage import digest,write_json,jsonable

SOURCE_MEMBERS=tuple(dict.fromkeys(previous.SOURCE_MEMBERS+(
    'src/go2arx5_vla/world_model/task_learning_grasp_v1.py',
    'src/go2arx5_vla/world_model/task_learning_preparation_v1.py',
    'src/go2arx5_vla/world_model/task_learning_scene_v1.py',
    'src/go2arx5_vla/world_model/task_learning_teacher_panel_v1.py',
    'src/go2arx5_vla/world_model/task_learning_expansion_v1.py',
    'src/go2arx5_vla/world_model/task_learning_collection_v1.py',
    'src/go2arx5_vla/world_model/task_learning_dev_v1.py',
    'src/go2arx5_vla/world_model/task_learning_recovery_v1.py',
    'src/go2arx5_vla/world_model/task_learning_teacher_v8.py',
    'src/go2arx5_vla/world_model/task_learning_teacher_v8_qualification.py',
    'src/go2arx5_vla/world_model/native_collection_performance_v1.py',
    'src/go2arx5_vla/world_model/native_state_snapshot.py',
    'src/go2arx5_vla/target_mujoco/state_snapshot.hpp',
    'src/go2arx5_vla/world_model/task_learning_grasp_native_v1.py')))


def scene_admission(spec):
    task.declared(spec)
    if spec.get('task_learning_role')==task.MATCHED_ROLE:
        from .task_learning_matched_native_v3_1 import native_scene_admission
        return native_scene_admission(spec)
    if spec.get('task_learning_role')==task.ENDPOINT_ROLE:
        from .task_learning_endpoint_native_v3 import native_scene_admission
        return native_scene_admission(spec)
    if spec.get('task_learning_role')=='TEACHER_V8_RECOVERY24_QUALIFICATION_NOT_BC_V1':
        from .task_learning_teacher_v8_qualification import native_scene_admission
        return native_scene_admission(spec)
    if spec.get('task_learning_role')==task.CONIC_ROLE:
        from .task_learning_conic_native_v3 import native_scene_admission
        return native_scene_admission(spec)
    if spec.get('task_learning_role')==task.CORE_LOCAL_ROLE:
        from .task_learning_local_policy_v1 import native_scene_admission
        return native_scene_admission(spec)
    from .task_learning_scene_v1 import admission
    proof=admission(spec)
    split='DEV'if spec.get('task_learning_role')in(task.DEV_REFERENCE_ROLE,task.STUDENT_DEV_ROLE)else'TRAIN'
    task.need(proof['split']==spec.get('data_role')==split and proof['parent']==spec['data_parent']
        and spec['scene_id']==proof['parent'],'NEW_NEAR_WORKSPACE_TRAIN_CONSTRUCTION_PILOT')
    panel=scenes.pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    role=spec.get('task_learning_role',task.CONSTRUCTION_ROLE)
    if role in(task.CONSTRUCTION_ROLE,task.QUALIFICATION_ROLE,task.EXPANSION_QUALIFICATION_ROLE):
        task.need(spec.get('task_learning_teacher_mode')in(None,'NORMAL_TEACHER_V7'),
            'HISTORICAL_CONSTRUCTION_AND_QUALIFICATION_NEVER_RECOVERY_RELABEL')
    qualification=role==task.QUALIFICATION_ROLE
    if qualification:
        from .task_learning_teacher_panel_v1 import validate as validate_panel
        rows=validate_panel(panel)
    elif role==task.EXPANSION_QUALIFICATION_ROLE:
        from .task_learning_expansion_v1 import validate as validate_panel
        rows=validate_panel(panel)
    elif role==task.STUDENT_DEV_ROLE:
        from .task_learning_student_v1 import validate as validate_panel
        rows=validate_panel(panel)
    elif role==task.DEV_REFERENCE_ROLE:
        from .task_learning_dev_v1 import validate as validate_panel
        rows=validate_panel(panel)
    elif role in(task.RECOVERY_QUALIFICATION_ROLE,task.COLLECTION_ROLE):
        from .task_learning_collection_v1 import validate as validate_panel
        task.need(panel.get('task_learning_role')==role,'PANEL_AND_NATIVE_COLLECTION_ROLE')
        rows=validate_panel(panel)
    else:
        task.need(panel.get('schema')=='TASK_LEARNING_GRASP_CONSTRUCTION_PANEL_V1'
            and panel.get('protocol')==task.protocol() and len(panel['assignments'])==1,'FROZEN_SINGLE_CONSTRUCTION_PILOT')
        rows=panel['assignments']
    matches=[r for r in rows if r['native_run']==Path(spec['registered_root']).name]
    task.need(len(matches)==1,'UNIQUE_ACTUAL_QUALIFICATION_OR_CONSTRUCTION_REQUEST')
    row=matches[0]
    if role==task.STUDENT_DEV_ROLE:
        from .task_learning_student_v1 import PURPOSE
        expected_purpose=PURPOSE
        task.need(spec.get('task_learning_teacher_mode')is None
            and spec.get('target_gr00t_policy')==panel['policies'][row['method']]
            and spec.get('target_gr00t_B1_global_step')==panel['B1_global_step']
            and spec.get('target_gr00t_query_seeds')==row['query_seeds'],
            'FROZEN_METHOD_AND_ALL_FIFTEEN_PREDECLARED_STUDENT_QUERIES')
    elif role==task.EXPANSION_QUALIFICATION_ROLE:
        from .task_learning_expansion_v1 import PURPOSE
        expected_purpose=PURPOSE
        task.need(spec.get('task_learning_teacher_mode')==row['Teacher_mode'],
            'NEXT_TRAIN_FEASIBILITY_UNCHANGED_TEACHER')
    elif role==task.DEV_REFERENCE_ROLE:
        from .task_learning_dev_v1 import purpose
        expected_purpose=purpose(role)
        task.need(spec.get('task_learning_teacher_mode')==row['Teacher_mode'],
            'DEV_REFERENCE_USES_UNCHANGED_NORMAL_TEACHER')
    elif role in(task.RECOVERY_QUALIFICATION_ROLE,task.COLLECTION_ROLE):
        from .task_learning_collection_v1 import purpose
        expected_purpose=purpose(role)
        task.need(spec.get('task_learning_teacher_mode')==row['Teacher_mode'],'EXACT_PREREGISTERED_NORMAL_OR_RECOVERY_MODE')
    else:expected_purpose='TEACHER_QUALIFICATION_NOT_BC'if qualification else'TASK_CONSTRUCTION_NOT_TEACHER_QUALIFICATION_OR_BC'
    task.need(row['native_run']==Path(spec['registered_root']).name and row['parent']==proof['parent']
        and row['model_sha256']==digest(spec['model_path']) and row['command']==task.COMMAND
        and row.get('diagnostic_control')==spec.get('task_learning_control')
        and row['purpose']==expected_purpose,'NEW_TASK_SCENE_RUN_JOIN')
    return proof


def validate(spec):
    task.need(spec.get('task_learning_control')in(None,task.HOLD_CONTROL),'DECLARED_NORMAL_OR_MATCHED_HOLD_CONTROL_ONLY')
    task.need(spec.get('task_learning_start_readiness')==task.start_readiness_contract(),
              'DECLARED_START_READINESS_BEFORE_ANY_NATIVE_STARTUP')
    scene_admission(spec)
    role=spec.get('task_learning_role',task.CONSTRUCTION_ROLE)
    if task.student_role(role):
        from .task_learning_student_v1 import EXPERIMENT
        if role==task.CORE_LOCAL_ROLE:
            from .task_learning_local_policy_v1 import EXPERIMENT
        if role==task.CONIC_ROLE:
            from .task_learning_conic_native_v3 import EXPERIMENT
        if role==task.ENDPOINT_ROLE:
            from .task_learning_endpoint_native_v3 import EXPERIMENT
        if role==task.MATCHED_ROLE:
            from .task_learning_matched_native_v3_1 import EXPERIMENT
        task.need(spec.get('experiment_role')==EXPERIMENT
            and spec.get('snapshot_binding_mode')=='PRE_REGISTERED_TASK_LEARNING_GRASP_V1'
            and spec.get('live_skill_geometry_scale')==1.
            and spec.get('live_skill_time_stretch_factor')==1
            and spec.get('task_learning_control')is None
            and spec.get('target_gr00t_gpu_root')and spec.get('target_gr00t_policy')
            and not any(spec.get(k)for k in ('target_gr00t_collection','paused_query_pair',
                'paused_query_gpu_spec','stable_task_candidate','stable_task_evaluation','program_reference_selection')),
            'TYPED_STUDENT_CURRENT_RGB_STATE_ONLY_WITHOUT_TEACHER_OR_WM')
        return
    if role=='TEACHER_V8_RECOVERY24_QUALIFICATION_NOT_BC_V1':
        from .task_learning_teacher_v8_qualification import EXPERIMENT
        expected_experiment=EXPERIMENT
    elif role==task.EXPANSION_QUALIFICATION_ROLE:
        from .task_learning_expansion_v1 import EXPERIMENT
        expected_experiment=EXPERIMENT
    elif role==task.DEV_REFERENCE_ROLE:
        from .task_learning_dev_v1 import experiment_role
        expected_experiment=experiment_role(role)
    elif role in(task.RECOVERY_QUALIFICATION_ROLE,task.COLLECTION_ROLE):
        from .task_learning_collection_v1 import experiment_role
        expected_experiment=experiment_role(role)
    else:expected_experiment='TASK_LEARNING_TEACHER_QUALIFICATION_NO_BC'if role==task.QUALIFICATION_ROLE else'TASK_LEARNING_TASK_CONSTRUCTION_NO_BC'
    task.need((role==task.CONSTRUCTION_ROLE or spec.get('task_learning_control')is None)
        and spec.get('experiment_role')==expected_experiment
        and spec.get('snapshot_binding_mode')=='PRE_REGISTERED_TASK_LEARNING_GRASP_V1'
        and spec.get('live_skill_geometry_scale')==1. and spec.get('live_skill_time_stretch_factor')==1
        and not any(spec.get(k)for k in ('target_gr00t_collection','target_gr00t_policy',
            'target_gr00t_gpu_root','paused_query_pair','paused_query_gpu_spec',
            'stable_task_candidate','stable_task_evaluation','program_reference_selection')),
        'NEW_CONTACT_TASK_WITHOUT_NEURAL_OR_PRIVILEGED_STUDENT_INPUTS')


def episode_config(spec):
    from .native_collection_performance_v1 import configuration
    configuration(spec)
    validate(spec)
    document=scenes.pinned(dict(path=spec['episode_manifest'],sha256=spec['episode_manifest_sha256']))
    episode=scenes.prefix_episode(document,spec)
    members=SOURCE_MEMBERS
    if task.student_role(spec.get('task_learning_role')):
        from .task_learning_live_pair_v1 import SOURCE_MEMBERS as model_members
        members=tuple(dict.fromkeys(members+model_members))
    episode,source=common.operational.recipe_config(spec,episode,actual_phase=task.PHASE,
        actual_kind=task.KIND,source_members=members)
    return episode,dict(source,stable_task_scene_reference=spec['stable_task_scene_reference'])


def episode_limit(spec,prior):
    episode_config(spec);common.audit_prefix_predecessor(spec,prior)
    if task.student_role(spec.get('task_learning_role')):
        from .task_learning_live_pair_v1 import preflight
        preflight(spec)
    return task.E


def validate_adapter_install(adapter,action):
    validate(adapter.spec);step=adapter.observe()['step']
    active=getattr(adapter,'stable_task_active_cycle',None)
    if active is None:
        from .task_learning_preparation_v1 import validate_preview
        return validate_preview(adapter,action)
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


def actual_feet(adapter,sample):
    """Read the original FK reader and native contact packet; never integrate."""
    import mujoco
    from scripts.pi0294.task_state import ActualState
    state=adapter.actual_state;packet=adapter.observe()
    task.need(type(state)is ActualState and packet['step']==sample['native_step']
        and state.data.time==sample['time_s']and np.array_equal(state.data.qpos,sample['qpos'])
        and np.array_equal(packet['qpos'],sample['qpos']),'ORIGINAL_SAME_PACKET_FOOT_FK_AND_CONTACTS')
    names=('FL','FR','RL','RR')
    positions=np.stack([state.data.geom_xpos[state.model.geom(name).id].copy()for name in names])
    support=set()
    for contact in packet['contacts']:
        pair=[mujoco.mj_id2name(state.model,mujoco.mjtObj.mjOBJ_GEOM,g)for g in(contact['geom0'],contact['geom1'])]
        if 'floor'in pair and contact['wrench'][0]>=task.standing_contract()['minimum_actual_floor_normal_force_N']:
            support.update(name for name in pair if name in names)
    return positions,sorted(support)


def continue_native(adapter,report,samples,windows):
    from .stable_task_closed_loop_v1 import _link_replacement
    from .stable_task_data_v1 import capture_input
    validate(adapter.spec);ready=report['task_local_preparation']
    task.need(adapter.observe()['step']==task.CALIBRATION_B and not adapter.trace_sealed
        and ready['original_steady_prefix_qualified']is True and ready['source_pose_alignment']is False
        and report['all_40_knots_consumed']is True,'ORIGINAL_STABLE_PREFIX_AND_TASK_LOCAL_HOLD')
    samples,windows=list(samples),list(windows)
    # New task reset intervention, after the unchanged original initialization.
    # Preserve native time and every consumption receipt; keep the old velocity gate.
    actual=np.asarray(samples[-1]['tcp']);raw=np.empty((40,10))
    raw[:,:3]=actual[:3,3];raw[:,3:9]=task.rotation_matrix_to_rot6d_rows(actual[:3,:3]);raw[:,9]=1.
    settling=task.FiftyMsProgram(raw,actual,adapter.reference_history[-1]['pose'],adapter.base_pose_hold)
    adapter.task_learning_stage='PREACTION_HOLD'
    reset_samples=[]
    def observe_reset(s):
        task.need(s.get('safety_valid')is True and not s.get('native_abort')and not s.get('illegal_contact'),
                  'ORIGINAL_NATIVE_SAFETY_DURING_DECLARED_PREACTION_HOLD')
        reset_samples.append(copy.deepcopy(s))
    reset_cycle,samples,windows=consumer.run_prefix(adapter,settling,index=-1,
        past_samples=samples,past_windows=windows,sample_observer=observe_reset)
    start_ready=task.require_start_readiness(adapter,samples[-1])
    feet,_=actual_feet(adapter,samples[-1])
    initial=dict(qpos=np.asarray(samples[-1]['qpos']).copy(),feet=feet)
    control=adapter.spec.get('task_learning_control')==task.HOLD_CONTROL
    student=task.student_role(adapter.spec.get('task_learning_role'))
    from .task_learning_recovery_v1 import NORMAL,UNDER_CLOSE,UnderCloseRecoveryTeacher,INTERVENTION
    from .task_learning_teacher_v8 import VERSION as V8_VERSION,NativeQualificationTeacher
    mode=NORMAL if student else adapter.spec.get('task_learning_teacher_mode',NORMAL)
    task.need(mode in(NORMAL,UNDER_CLOSE,V8_VERSION)and(not control or mode==NORMAL)
        and(mode!=V8_VERSION or adapter.spec.get('task_learning_role')=='TEACHER_V8_RECOVERY24_QUALIFICATION_NOT_BC_V1'),
        'EXPLICIT_TEACHER_MODE')
    if student:
        from .task_learning_student_v1 import StudentPolicy
        teacher=StudentPolicy(adapter,samples[-1],consumed_reference=lambda:(cycles[-1],adapter.live_skill_program))
    elif mode==V8_VERSION:
        teacher=NativeQualificationTeacher(adapter,samples[-1],lambda:(cycles[-1],adapter.live_skill_program))
    else:
        teacher=(task.HoldControl if control else UnderCloseRecoveryTeacher if mode==UNDER_CLOSE else task.Teacher)(samples[-1])
    intervention_decisions=0
    cycles=[reset_cycle];decisions=[];captures=[];confirmed=[]
    task_samples=[samples[-1]];standing=[]
    def observe(s):
        feet,supported=actual_feet(adapter,s)
        witness=dict(task.standing_metrics(s['qpos'][:7],feet,supported,initial),native_step=s['native_step'])
        if not standing or s['native_step']>standing[-1]['native_step']:standing.append(witness)
        if not witness['passes']:
            write_progress(adapter.output/'TASK_LEARNING_STANDING_ACTUAL.private.json',dict(records=standing,complete=False))
            task.need(False,'NEW_TASK_STANDING_SCORE_BOUND_EXCEEDED:'+repr(witness))
        task.need(all(s.get(k)is True for k in ('safety_valid','contact_valid','gripper_state_valid','object_present'))
            and not any(s.get(k)for k in ('native_abort','illegal_contact','joint_limit_penetration','reset','takeover')),
            'UNCHANGED_NATIVE_CONTACT_AND_STANDING_SAFETY')
        if s['native_step']>task_samples[-1]['native_step']:task_samples.append(copy.deepcopy(s))
    for index,step in enumerate(consumer.observation_steps(adapter.spec)):
        task.need(adapter.observe()['step']==samples[-1]['native_step']==step,'REAL_CONTINUOUS_TEACHER_CLOCK')
        observe(samples[-1])
        capture=capture_input(adapter,language=task.LANGUAGE,task_score=task.score_contract())
        captures.append(capture)
        write_progress(adapter.output/'TASK_LEARNING_CAPTURES.private.json',dict(records=captures,complete=False))
        if student:teacher.observation=capture
        from .native_collection_performance_v1 import measure
        with measure(adapter,'policy.student_inference' if student else 'policy.Teacher_decision', native_step=step):
            program,proposal=teacher.decide(samples[-1],adapter.reference_history[-1]['pose'],adapter.base_pose_hold)
        intervention_decisions+=proposal.get('label_role')==INTERVENTION
        adapter.task_learning_stage=teacher.stage
        decision_path=adapter.output/f'TASK_LEARNING_DECISION_{index:03d}.private.json'
        adaptation=program.evidence()
        if proposal.get('explicit_decoder_applied')is True:
            adaptation.update(original_raw_H40_unchanged=False,explicit_composed_H40_unchanged_by_transport=True,
                explicit_decoder_execution_version=proposal['explicit_decoder_execution_version'])
        write_json(decision_path,dict(proposal,native_step=step,observation=capture,
            original_actual_TCP=np.asarray(samples[-1]['tcp']).tolist(),adaptation=adaptation,
            privileged_Teacher_metadata_not_student_input=not control,
            diagnostic_control=task.HOLD_CONTROL if control else None,expert_training_eligible=False))
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
                **{'student_decision_sha256'if student else'control_decision_sha256'if control else'Teacher_decision_sha256':digest(decision_path)},
                solver_gripper_confirmed=True))
        write_progress(adapter.output/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json',dict(records=confirmed,
            period_s=.05,complete=False,unexecuted_suffix_included=False,future_achieved_labels=False))
        write_progress(adapter.output/'TASK_LEARNING_STANDING_ACTUAL.private.json',dict(records=standing,complete=False))
    # Termination continues the last explicit Teacher grasp/hold command.
    raw=np.repeat(program.decoded_H40[-1:,:],40,axis=0)
    raw[:,:3]=np.asarray(samples[-1]['tcp'])[:3,3]
    raw[:,3:9]=task.rotation_matrix_to_rot6d_rows(np.asarray(samples[-1]['tcp'])[:3,:3])
    hold=task.FiftyMsProgram(raw,samples[-1]['tcp'],adapter.reference_history[-1]['pose'],adapter.base_pose_hold)
    current,samples,windows=consumer.run_prefix(adapter,hold,index=task.QUERIES,
        past_samples=samples,past_windows=windows,sample_observer=observe)
    _link_replacement(adapter,cycles[-1],current);cycles.append(current)
    scored=task.score(task_samples,teacher.cube[2,3],standing=standing)
    write_progress(adapter.output/'TASK_LEARNING_STANDING_ACTUAL.private.json',dict(records=standing,complete=True))
    write_json(adapter.output/'TASK_LEARNING_SCORE.private.json',scored)
    write_progress(adapter.output/'TASK_LEARNING_CAPTURES.private.json',dict(records=captures,complete=True,task_success=scored['task_success']))
    write_progress(adapter.output/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json',dict(records=confirmed,
        period_s=.05,complete=True,unexecuted_suffix_included=False,future_achieved_labels=False,
        task_success=scored['task_success'],expert_training_eligible=False))
    return jsonable(dict(copy.deepcopy(report),schema=native_result_schema(adapter.spec),
        actual_spec_phase=task.PHASE,native_endpoint_step=task.E,
        task_learning_grasp=task.protocol(adapter.spec.get('task_learning_role',task.CONSTRUCTION_ROLE)),
        teacher_decisions=[]if control or student else decisions,control_decisions=decisions if control else[],
        **(dict(student_decisions=decisions,all_fifteen_generated_H40_consumed=True,
            student_commands_are_expert_labels=False)if student else{}),
        diagnostic_control=task.HOLD_CONTROL if control else None,generated_cycles=cycles,windows=windows,
        initialization_all40_knots_consumed=True,all_40_knots_consumed=False,raw_knot_consumed=[False]*40,
        GR00T_raw_H40_consumed=student and adapter.spec.get('target_gr00t_policy',{}).get('arm')!='C500_PLUS_CONIC' and not str(adapter.spec.get('target_gr00t_policy',{}).get('arm')).endswith('_PLUS_ENDPOINT_BASIS'),
        matched_smooth_H40_consumed=student and adapter.spec.get('task_learning_role')==task.MATCHED_ROLE and str(adapter.spec['target_gr00t_policy']['arm']).endswith('_PLUS_ENDPOINT_BASIS'),
        explicit_endpoint_basis_H40_consumed=student and adapter.spec.get('task_learning_role')==task.ENDPOINT_ROLE,
        explicit_conic_H40_consumed=student and adapter.spec.get('target_gr00t_policy',{}).get('arm')=='C500_PLUS_CONIC',complete_task_atoms=len(windows),task_native_ticks=task.E-2503,
        solver_target_confirmed_atoms=len(windows),gripper_confirmed_atoms=len(windows),
        new_task_score=scored,task_condition_success=scored['task_success'],
        decision_start_readiness=start_ready,expert_training_eligible=False,
        data_role=('TEACHER_QUALIFICATION_NOT_BC'if adapter.spec.get('task_learning_role')==task.QUALIFICATION_ROLE else
            'TASK_CONSTRUCTION_NOT_BC'if adapter.spec.get('task_learning_role',task.CONSTRUCTION_ROLE)==task.CONSTRUCTION_ROLE else
            adapter.spec['task_learning_role']),
        excluded_intervention_decisions=intervention_decisions,
        expert_policy_decisions=0 if control or student else task.QUERIES-intervention_decisions,
        recovery_evidence=dict(mode=mode,intervention_issued=getattr(teacher,'intervention_issued',False),
            actual_under_close_failed=getattr(teacher,'actual_under_close_failed',False),
            native_grasp_after_correction=getattr(teacher,'native_grasp_after_correction',False)),
        **(dict(teacher_v8_recovery_evidence=teacher.evidence())if mode==V8_VERSION else{}),
        GR00T_queries=task.QUERIES if student else 0,VLA_queries=task.QUERIES if student else 0,GR00T_updates=0,
        WM_updates=0,Actor_updates=0,full_task_success=False,target_task_success=False,
        new_grasp_lift_hold_task_success=False if control else scored['task_success'],source_task_phase_equivalence=False))


def native_result_schema(spec):
    role=spec.get('task_learning_role',task.CONSTRUCTION_ROLE)
    if role==task.MATCHED_ROLE:
        from .task_learning_matched_native_v3_1 import NATIVE_SCHEMA
        return NATIVE_SCHEMA
    if role==task.ENDPOINT_ROLE:
        from .task_learning_endpoint_native_v3 import NATIVE_SCHEMA
        return NATIVE_SCHEMA
    if role=='TEACHER_V8_RECOVERY24_QUALIFICATION_NOT_BC_V1':
        from .task_learning_teacher_v8_qualification import NATIVE_SCHEMA
        return NATIVE_SCHEMA
    if role==task.CONIC_ROLE:
        from .task_learning_conic_native_v3 import NATIVE_SCHEMA
        return NATIVE_SCHEMA
    if role==task.CORE_LOCAL_ROLE:
        from .task_learning_local_policy_v1 import NATIVE_SCHEMA
        return NATIVE_SCHEMA
    if role==task.EXPANSION_QUALIFICATION_ROLE:
        from .task_learning_expansion_v1 import NATIVE_SCHEMA
        return NATIVE_SCHEMA
    if role==task.STUDENT_DEV_ROLE:
        from .task_learning_student_v1 import NATIVE_SCHEMA
        return NATIVE_SCHEMA
    if role==task.DEV_REFERENCE_ROLE:
        from .task_learning_dev_v1 import native_schema
        return native_schema(role)
    if role in(task.RECOVERY_QUALIFICATION_ROLE,task.COLLECTION_ROLE):
        from .task_learning_collection_v1 import native_schema
        return native_schema(role)
    return 'TASK_LEARNING_TEACHER_QUALIFICATION_NATIVE_ACTUAL_V1'if role==task.QUALIFICATION_ROLE else'TASK_LEARNING_GRASP_CONSTRUCTION_NATIVE_ACTUAL_V1'


def complete_consumption(spec,result):
    task.declared(spec);cycles=result.get('generated_cycles')or[]
    control=spec.get('task_learning_control')==task.HOLD_CONTROL
    student=task.student_role(spec.get('task_learning_role'))
    decisions=result.get('student_decisions'if student else'control_decisions'if control else'teacher_decisions',[])
    expected_schema=native_result_schema(spec)
    if not(result.get('schema')==expected_schema
        and result.get('diagnostic_control')==(task.HOLD_CONTROL if control else None)
        and (not(control or student)or result.get('teacher_decisions')==[])
        and (not student or(result.get('GR00T_queries')==task.QUERIES
            and result.get('student_commands_are_expert_labels')is False
            and result.get('all_fifteen_generated_H40_consumed')is True))
        and len(cycles)==task.QUERIES+2 and len(decisions)==task.QUERIES
        and result.get('expert_training_eligible')is False):return False
    for position,row in enumerate(cycles):
        index=position-1
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
    from .fusion_trajectory_qualification_native import run_program
    from .task_learning_preparation_v1 import HoldProgram
    import json
    episode,source=episode_config(adapter.spec)
    # Operational/source identities were checked above; no source observation,
    # source pose or historical terminal error determines the new references.
    inputs={}
    def factory(destination,actual_episode,initial,actual,base):
        task.need(destination is inputs and actual_episode is episode and not inputs,
            'ONE_TASK_LOCAL_PREPARATION_BINDING')
        program=HoldProgram(initial,actual)
        inputs.update(raw=program.desired_path,provenance=dict(
            producer='ORIGINAL_INITIAL_TCP_REFERENCE_HOLD_NOT_EXPERT_ACTION',source_pose_alignment=False))
        return program
    def finalizer(report):
        prefix=json.loads((adapter.output/'TRAJECTORY_QUALIFICATION_PREFIX.private.json').read_text())
        task.need(prefix['qualified']is True and prefix['native_step']==3703,
            'UNCHANGED_ORIGINAL_STEADY_PREFIX_GATE')
        report['task_local_preparation']=dict(original_steady_prefix_qualified=True,
            original_steady_prefix_native_step=3703,source_pose_alignment=False,
            reference_hold_endpoint=task.CALIBRATION_B,startup_WBC_native_safety_or_dynamics_changed=False)
    return run_program(adapter,episode,inputs,program_factory=factory,
        preparation_end_step=task.CALIBRATION_B,report_finalizer=finalizer,continuation=continue_native,
        report_context=dict(schema='TASK_LEARNING_LOCAL_PREPARATION_NATIVE_V4',actual_spec_phase=task.PHASE,
            source_pose_alignment=False,post_decision_atoms=40,tail_hold_atoms=0,
            experiment_role='TASK_LOCAL_PREPARATION_NOT_SOURCE_EQUIVALENCE_OR_EXPERT_DATA'))
