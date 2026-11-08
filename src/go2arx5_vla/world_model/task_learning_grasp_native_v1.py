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
    'src/go2arx5_vla/world_model/task_learning_grasp_native_v1.py')))


def scene_admission(spec):
    task.declared(spec)
    from .task_learning_scene_v1 import admission
    proof=admission(spec)
    task.need(proof['split']==spec.get('data_role')=='TRAIN' and proof['parent']==spec['data_parent']
        and spec['scene_id']==proof['parent'],'NEW_NEAR_WORKSPACE_TRAIN_CONSTRUCTION_PILOT')
    panel=scenes.pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    qualification=spec.get('task_learning_role')==task.QUALIFICATION_ROLE
    if qualification:
        from .task_learning_teacher_panel_v1 import validate as validate_panel
        rows=validate_panel(panel)
    else:
        task.need(panel.get('schema')=='TASK_LEARNING_GRASP_CONSTRUCTION_PANEL_V1'
            and panel.get('protocol')==task.protocol() and len(panel['assignments'])==1,'FROZEN_SINGLE_CONSTRUCTION_PILOT')
        rows=panel['assignments']
    matches=[r for r in rows if r['native_run']==Path(spec['registered_root']).name]
    task.need(len(matches)==1,'UNIQUE_ACTUAL_QUALIFICATION_OR_CONSTRUCTION_REQUEST')
    row=matches[0]
    task.need(row['native_run']==Path(spec['registered_root']).name and row['parent']==proof['parent']
        and row['model_sha256']==digest(spec['model_path']) and row['command']==task.COMMAND
        and row.get('diagnostic_control')==spec.get('task_learning_control')
        and row['purpose']==('TEACHER_QUALIFICATION_NOT_BC'if qualification else'TASK_CONSTRUCTION_NOT_TEACHER_QUALIFICATION_OR_BC'),'NEW_TASK_SCENE_RUN_JOIN')
    return proof


def validate(spec):
    task.need(spec.get('task_learning_control')in(None,task.HOLD_CONTROL),'DECLARED_NORMAL_OR_MATCHED_HOLD_CONTROL_ONLY')
    task.need(spec.get('task_learning_start_readiness')==task.start_readiness_contract(),
              'DECLARED_START_READINESS_BEFORE_ANY_NATIVE_STARTUP')
    scene_admission(spec)
    qualification=spec.get('task_learning_role')==task.QUALIFICATION_ROLE
    task.need((not qualification or spec.get('task_learning_control')is None)
        and spec.get('experiment_role')==('TASK_LEARNING_TEACHER_QUALIFICATION_NO_BC'if qualification else'TASK_LEARNING_TASK_CONSTRUCTION_NO_BC')
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
    teacher=(task.HoldControl if control else task.Teacher)(samples[-1])
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
        program,proposal=teacher.decide(samples[-1],adapter.reference_history[-1]['pose'],adapter.base_pose_hold)
        adapter.task_learning_stage=teacher.stage
        decision_path=adapter.output/f'TASK_LEARNING_DECISION_{index:03d}.private.json'
        write_json(decision_path,dict(proposal,native_step=step,observation=capture,
            original_actual_TCP=np.asarray(samples[-1]['tcp']).tolist(),adaptation=program.evidence(),
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
                **{'control_decision_sha256'if control else'Teacher_decision_sha256':digest(decision_path)},
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
    return jsonable(dict(copy.deepcopy(report),schema='TASK_LEARNING_TEACHER_QUALIFICATION_NATIVE_ACTUAL_V1'
        if adapter.spec.get('task_learning_role')==task.QUALIFICATION_ROLE else'TASK_LEARNING_GRASP_CONSTRUCTION_NATIVE_ACTUAL_V1',
        actual_spec_phase=task.PHASE,native_endpoint_step=task.E,task_learning_grasp=task.protocol(),
        teacher_decisions=[]if control else decisions,control_decisions=decisions if control else[],
        diagnostic_control=task.HOLD_CONTROL if control else None,generated_cycles=cycles,windows=windows,
        initialization_all40_knots_consumed=True,all_40_knots_consumed=False,raw_knot_consumed=[False]*40,
        GR00T_raw_H40_consumed=False,complete_task_atoms=len(windows),task_native_ticks=task.E-2503,
        solver_target_confirmed_atoms=len(windows),gripper_confirmed_atoms=len(windows),
        new_task_score=scored,task_condition_success=scored['task_success'],
        decision_start_readiness=start_ready,expert_training_eligible=False,
        data_role='TEACHER_QUALIFICATION_NOT_BC'if adapter.spec.get('task_learning_role')==task.QUALIFICATION_ROLE else'TASK_CONSTRUCTION_NOT_BC',
        GR00T_queries=0,VLA_queries=0,GR00T_updates=0,
        WM_updates=0,Actor_updates=0,full_task_success=False,target_task_success=False,
        new_grasp_lift_hold_task_success=False if control else scored['task_success'],source_task_phase_equivalence=False))


def complete_consumption(spec,result):
    task.declared(spec);cycles=result.get('generated_cycles')or[]
    control=spec.get('task_learning_control')==task.HOLD_CONTROL
    decisions=result.get('control_decisions'if control else'teacher_decisions',[])
    expected_schema='TASK_LEARNING_TEACHER_QUALIFICATION_NATIVE_ACTUAL_V1'if spec.get('task_learning_role')==task.QUALIFICATION_ROLE else'TASK_LEARNING_GRASP_CONSTRUCTION_NATIVE_ACTUAL_V1'
    if not(result.get('schema')==expected_schema
        and result.get('diagnostic_control')==(task.HOLD_CONTROL if control else None)
        and (not control or result.get('teacher_decisions')==[])
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
