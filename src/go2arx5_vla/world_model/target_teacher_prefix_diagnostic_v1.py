"""Two privileged4mm decisions through the unchanged paused-prefix consumer.

This separate TRAIN-geometry diagnostic tests two-cycle reachability. It never
generates GR00T queries, expert labels, learned-policy qualification or a retry
of the failed B0 condition. Original admission, standing/contact guards, ACKs,
reference clocks and native2ms integration remain mandatory.
"""
import copy
from pathlib import Path
import numpy as np
from . import stable_start_short_task_v1 as common
from . import stable_task_scene_v1 as scene
from . import stable_task_prefix_v1 as consumer
from . import target_gr00t_collection_v1 as tasks
from .storage import digest,write_json,jsonable

PHASE='collect-target-teacher-prefix-diagnostic-v1'
KIND='TARGET_TEACHER_PREFIX_DIAGNOSTIC_V1'
B,E=34303,40503
SOURCE_MEMBERS=tuple(dict.fromkeys(common.SOURCE_MEMBERS+(
    'src/go2arx5_vla/world_model/stable_task_scene_v1.py',
    'src/go2arx5_vla/world_model/stable_task_prefix_v1.py',
    'src/go2arx5_vla/world_model/stable_task_closed_loop_v1.py',
    'src/go2arx5_vla/world_model/target_gr00t_collection_v1.py',
    'src/go2arx5_vla/world_model/target_teacher_prefix_diagnostic_v1.py',
    'scripts/wm02/target_teacher_prefix_diagnostic_v1.py')))


def need(ok,why):
    if not ok:raise ValueError('TARGET_TEACHER_PREFIX_'+why)


def protocol():
    return dict(schema='WM02_TARGET_TEACHER_TWO_PREFIX_DIAGNOSTIC_V1',
        method='PRIVILEGED_GEOMETRIC_TEACHER_4MM_EACH_FRESH_DECISION',
        prefix_atoms=120,policy_queries=2,final_hold_atoms=8,
        geometry_scale=1.,time_stretch_factor=2,physics_dt_s=.002,WBC_changes=False,
        new_neural_queries=0,new_updates=0,expert_training_eligible=False,
        data_purpose='DIAGNOSTIC_ONLY_NO_AUTO_TRAIN',
        desired_outcome_provenance='PRIVILEGED_CURRENT_SCENE_GEOMETRY',
        assistance='NATIVE_QUOTA_PAUSED_REAL_OBSERVATION',
        original_per_install_MPC_clock=True,teacher_fallback=False)


def declared(spec):
    need(spec.get('phase')==PHASE and spec.get('kind')==KIND
        and spec.get('target_teacher_prefix_diagnostic')==protocol()
        and spec.get('live_skill_start_step')==B and spec.get('max_ticks')==E
        and spec.get('physics_dt_s')==.002 and spec.get('WBC_changes')is False,
        'EXACT_NEW_TYPED_FINITE_DIAGNOSTIC')


def validate(spec):
    declared(spec);scene.scene(spec)
    need(spec.get('data_role')=='TRAIN'and spec.get('task_command')in tasks.COMMANDS
        and spec.get('experiment_role')=='TARGET_TEACHER_TWO_PREFIX_DIAGNOSTIC_NOT_BC'
        and spec.get('snapshot_binding_mode')=='PRE_REGISTERED_TARGET_TEACHER_TWO_PREFIX_V1'
        and spec.get('target_gr00t_start_readiness')==tasks.start_readiness_contract()
        and spec.get('live_skill_geometry_scale')==1.
        and spec.get('live_skill_time_stretch_factor')==2
        and not any(spec.get(k)for k in ('target_gr00t_collection','target_gr00t_policy',
            'target_gr00t_gpu_root','paused_query_pair','paused_query_gpu_spec',
            'stable_task_candidate','stable_task_evaluation','program_reference_selection')),
        'PRIVILEGED_DIAGNOSTIC_WITHOUT_MODEL_OR_EXPERT_RELABELLING')
    p=scene.pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    need(p.get('teacher_prefix_protocol')==protocol(),'PANEL_FREEZES_DIAGNOSTIC_METHOD')


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
    declared(adapter.spec);step=adapter.observe()['step']
    active=getattr(adapter,'stable_task_active_cycle',None)
    if active is None:
        need(step<B,'CALIBRATION_PRECEDES_FIRST_TEACHER_DECISION')
        return common.prefix.validate_install_preview(action,initial_tcp=adapter.trajectory_initial_tcp,
            program=adapter.trajectory_program,step=step,base=adapter.base_pose_hold,
            frozen_base=adapter.trajectory_initial_base,end_step=B)
    c=consumer.cycle(adapter.spec,active['index']);program=active['program']
    need(all(active[k]==v for k,v in c.items())and type(program)is TrajectoryAdaptation
        and c['start_step']<=step<c['end_step']and(step-c['start_step'])%25==0
        and program.geometry_scale==1. and program.time_stretch_factor==2
        and np.array_equal(action,program.preview((step-c['start_step'])//25))
        and np.array_equal(program.base_pose_hold,adapter.base_pose_hold)
        and np.array_equal(adapter.base_pose_hold,adapter.trajectory_initial_base),
        'EXACT_CURRENT_FULL_PREVIEW_AND_ORIGINAL_BASE_HOLD')
    return dict(status='ORIGINAL_CERTIFICATE_AND_ACK_STILL_REQUIRED',native_step=step,
        cycle_index=c['index'],physics_advance_authorized=False)


def continue_native(adapter,report,samples,windows):
    from .fusion_task_phase_live_skill_native_v8 import _noncontact
    from .fusion_task_reference_diagnostic_v1 import require_standing
    from .stable_task_closed_loop_v1 import _link_replacement
    from .stable_task_policy_v1 import build_for_delta
    validate(adapter.spec);ready=report['task_phase_initialization']['readiness']
    need(adapter.observe()['step']==B and not adapter.trace_sealed
        and all(v is True for k,v in ready['predicates'].items()if k!='full_grade')
        and report['all_40_knots_consumed']is True and len(windows)==(B-2503)//25,
        'ACTUAL_UNCHANGED_CALIBRATION_BEFORE_TEACHER')
    readiness=tasks.require_start_readiness(adapter,samples[-1])
    samples,windows=list(samples),list(windows)
    initial=dict(qpos=np.asarray(samples[-1]['qpos']).copy());cube=np.asarray(samples[-1]['object_pose']).copy()
    cycles=[];decisions=[]
    def observe(sample):_noncontact(sample,cube);require_standing(sample,initial)
    try:
        for index,step in enumerate(consumer.observation_steps(adapter.spec)):
            need(adapter.observe()['step']==samples[-1]['native_step']==step,'CURRENT_ACTUAL_TEACHER_CLOCK')
            observe(samples[-1])
            program,proposal=tasks.build(samples[-1]['tcp'],adapter.reference_history[-1]['pose'],
                adapter.base_pose_hold,cube,tasks.config(adapter.spec['task_command'],'TRAIN'))
            decision=jsonable(dict(proposal,cycle_index=index,native_step=step,
                producer=protocol()['method'],GR00T_queries=0,expert_training_eligible=False))
            write_json(adapter.output/f'TEACHER_PREFIX_DECISION_{index:02d}.private.json',decision)
            decisions.append(decision)
            current,samples,windows=consumer.run_prefix(adapter,program,index=index,
                past_samples=samples,past_windows=windows,sample_observer=observe)
            if cycles:_link_replacement(adapter,cycles[-1],current)
            cycles.append(current)
        program,hold=build_for_delta(samples[-1]['tcp'],adapter.reference_history[-1]['pose'],
            adapter.base_pose_hold,np.zeros(3))
        write_json(adapter.output/'TEACHER_PREFIX_TERMINATION_HOLD.private.json',hold)
        current,samples,windows=consumer.run_prefix(adapter,program,index=2,
            past_samples=samples,past_windows=windows,sample_observer=observe)
        _link_replacement(adapter,cycles[-1],current);cycles.append(current)
    except BaseException as error:
        write_json(adapter.output/'TEACHER_PREFIX_FAILED.private.json',dict(
            failure=type(error).__name__+': '+str(error),teacher_decisions=len(decisions),
            last_native_step=adapter.observe()['step'],GR00T_queries=0,expert_training_eligible=False))
        raise
    scored=tasks.score([s for s in samples if s['native_step']>=B],adapter.spec['task_command'])
    return jsonable(dict(copy.deepcopy(report),schema='WM02_TARGET_TEACHER_TWO_PREFIX_NATIVE_ACTUAL_V1',
        actual_spec_phase=PHASE,native_endpoint_step=E,data_role='TRAIN_DIAGNOSTIC_ONLY',
        data_parent=adapter.spec['data_parent'],decision_start_readiness=readiness,
        teacher_prefix_protocol=protocol(),teacher_decisions=decisions,generated_cycles=cycles,
        initialization_all40_knots_consumed=True,all_40_knots_consumed=False,
        raw_knot_consumed=[False]*40,GR00T_raw_H40_consumed=False,
        windows=windows,complete_task_atoms=len(windows),task_native_ticks=E-2503,
        solver_target_confirmed_atoms=len(windows),gripper_confirmed_atoms=len(windows),
        target_gr00t_task_score=scored,task_condition_success=scored['task_success'],
        GR00T_queries=0,VLA_queries=0,GR00T_updates=0,WM_updates=0,Actor_updates=0,
        TASK_ADAPTER_updates=0,teacher_fallbacks=0,expert_training_eligible=False,
        GR00T_native_interface_qualified=False,full_task_success=False,
        original_low_level_state_reset=False,full_MPC_hidden_state_restored=False,
        real_time_deployment=False,native_clock_rebased=False))


def complete_consumption(spec,complete):
    declared(spec);cycles=complete.get('generated_cycles')or[]
    if not(complete.get('schema')=='WM02_TARGET_TEACHER_TWO_PREFIX_NATIVE_ACTUAL_V1'
        and complete.get('teacher_prefix_protocol')==protocol()and len(cycles)==3
        and len(complete.get('teacher_decisions',[]))==2
        and [r.get('native_step')for r in complete['teacher_decisions']]==list(consumer.observation_steps(spec))
        and complete.get('expert_training_eligible')is False
        and complete.get('GR00T_native_interface_qualified')is False
        and complete.get('GR00T_queries')==complete.get('GR00T_updates')
            ==complete.get('WM_updates')==complete.get('Actor_updates')==0
        and complete.get('all_40_knots_consumed')is False):return False
    for i,row in enumerate(cycles):
        c=consumer.cycle(spec,i);windows=row.get('windows')or[]
        if not(row.get('cycle')==c and row.get('complete_suffix_atoms')==len(windows)==c['atoms']
            and row.get('solver_target_confirmed_atoms')==row.get('gripper_confirmed_atoms')==c['atoms']):return False
        offset=(c['start_step']-2503)//25
        if windows!=complete['windows'][offset:offset+c['atoms']]:return False
        for j,w in enumerate(windows):
            grip=w.get('gripper')or{}
            if not(w.get('native_start_step')==c['start_step']+25*j
                and w.get('native_end_step')==c['start_step']+25*(j+1)
                and w.get('actual_ticks')==25 and w.get('solver_policies_with_exact_installed_target')
                and grip.get('command_consumed')is True
                and (grip.get('submitted')or{}).get('sequence')==offset+j):return False
        if row.get('raw_knot_consumed')!=[int(k)<c['atoms']for k in row['adaptation']['clock']['knot_indices']]:return False
        if i<2:
            following=cycles[i+1]['installs'][0];replacement=row.get('cancellation')or{}
            if not(row.get('cancellation_status')=='SUPERSEDED_BY_ORIGINAL_NEW_REFERENCE_ACK'
                and replacement.get('native_step')==c['end_step']==following['native_step']
                and replacement.get('old_reference_sha256')==row['final_reference_sha256']==following['previous_reference_sha256']
                and replacement.get('new_reference_sha256')==following['wire_sha256']):return False
    return True


def run(adapter):
    from .fusion_task_phase_xyz_feedback_native_v4 import run_feedback_program
    episode,source=episode_config(adapter.spec)
    return run_feedback_program(adapter,episode,source,recipe_module=common.prefix,actual_phase=PHASE,
        report_schema='WM02_TEACHER_PREFIX_CALIBRATION_V1',goal_schema='WM02_TEACHER_PREFIX_GOAL_V1',
        decision_schema='WM02_TEACHER_PREFIX_DECISION_V1',feedback_report_schema='WM02_TEACHER_PREFIX_FEEDBACK_V1',
        continuation=continue_native)
