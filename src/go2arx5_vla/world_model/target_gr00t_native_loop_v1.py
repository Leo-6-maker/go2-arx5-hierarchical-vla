"""Actual GR00T observations and finite prefixes in one original native world.

The original owner, calibration, certificates, quota, solver/gripper joins and
trace lifecycle remain authoritative. There is no Teacher or WM action fallback.
"""
import copy,json,time
from pathlib import Path
import numpy as np

from . import target_gr00t_prefix_v1 as protocol
from . import stable_task_prefix_v1 as consumer
from . import stable_task_scene_v1 as scene
from . import stable_start_short_task_v1 as common
from . import target_gr00t_collection_v1 as tasks
from . import target_gr00t_live_pair_v1 as pair
from .storage import digest,write_json,jsonable

PHASE,KIND,B,E=protocol.PHASE,protocol.KIND,protocol.B,protocol.E
need=protocol.need
SOURCE_MEMBERS=tuple(dict.fromkeys(common.SOURCE_MEMBERS+(
    'src/go2arx5_vla/world_model/stable_task_scene_v1.py',
    'src/go2arx5_vla/world_model/stable_task_data_v1.py',
    'src/go2arx5_vla/world_model/stable_task_prefix_v1.py',
    'src/go2arx5_vla/world_model/stable_task_closed_loop_v1.py',
    'src/go2arx5_vla/world_model/target_gr00t_collection_v1.py',
    'src/go2arx5_vla/world_model/target_gr00t_prefix_v1.py',
    'src/go2arx5_vla/world_model/target_gr00t_live_pair_v1.py',
    'src/go2arx5_vla/world_model/target_gr00t_native_loop_v1.py',
    'scripts/wm02/target_gr00t_native_loop_check_v1.py')))


def validate(spec):
    protocol.declared(spec);scene.scene(spec)
    need(spec.get('task_command')in tasks.COMMANDS
        and spec.get('snapshot_binding_mode')=='PRE_REGISTERED_TARGET_GR00T_PREFIX_PAIR_V1'
        and spec.get('live_skill_geometry_scale')==1.
        and spec.get('live_skill_time_stretch_factor')==2
        and spec.get('target_gr00t_start_readiness')==tasks.start_readiness_contract()
        and not any(spec.get(k)for k in ('stable_task_candidate','stable_task_evaluation',
            'paused_query_pair','program_reference_selection','program_reference_intervention')),
        'EXPLICIT_NEW_GR00T_LOOP_NOT_EXTERNAL_ADAPTER_OR_WM')
    panel=scene.pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    need(panel.get('target_gr00t_prefix_protocol')==protocol.protocol()
        and panel.get('policies',{}).get(spec['target_gr00t_policy']['arm'])==spec['target_gr00t_policy'],
        'PANEL_FREEZES_POLICY_PACKAGE_AND_EXECUTION_BEFORE_DEV')
    if spec['target_gr00t_policy']['arm']=='B1':
        step=spec.get('target_gr00t_B1_global_step',2000)
        need(type(step)is int and step in(2000,4000,8000)
            and step==panel.get('B1_global_step',2000),
            'DECLARED_PANEL_ENDPOINT_NOT_POSTHOC_MODEL_SELECTION')
    rows=[r for r in panel['assignments']if r['native_run']==Path(spec['registered_root']).name]
    need(len(rows)==1 and rows[0]['arm']==spec['target_gr00t_policy']['arm']
        and rows[0]['seeds']==spec['target_gr00t_query_seeds'],
        'PREDECLARED_MODEL_AND_TWO_NOISE_SEEDS_FOR_THIS_CONDITION')
    if spec['data_role']=='TRAIN':
        need(spec.get('experiment_role')=='TARGET_GR00T_NATIVE_INTERFACE_QUALIFICATION_NOT_BC',
             'TRAIN_INTERFACE_CHECK_IS_NOT_EXPERT_OR_FORMAL_DEV')
    else:need(spec['data_role']=='DEV'and spec.get('experiment_role')=='TARGET_GR00T_NATIVE_PREFIX_DEV_V1',
              'INDEPENDENT_ORIGINAL_DEV_ROLE')
    if spec['data_role']=='DEV':require_interface_qualification(spec)
    return spec['target_gr00t_policy']


def require_interface_qualification(spec):
    from .stable_task_prefix_qualification_v1 import sealed_native
    descriptor=spec.get('target_gr00t_native_interface_qualification')
    need(isinstance(descriptor,dict),'REAL_NATIVE_INTERFACE_QUALIFICATION_REQUIRED_BEFORE_DEV')
    _,qualified,result=sealed_native(descriptor)
    need(qualified.get('phase')==PHASE and qualified.get('kind')==KIND
        and qualified.get('data_role')=='TRAIN'
        and qualified.get('experiment_role')=='TARGET_GR00T_NATIVE_INTERFACE_QUALIFICATION_NOT_BC'
        and all(qualified['source_files'].get(n)==spec['source_files'].get(n)
            and n in qualified['source_files']for n in SOURCE_MEMBERS)
        and complete_consumption(qualified,result.get('reference_qualification')or{}),
        'REAL_SAME_IMPLEMENTATION_TWO_QUERY_NATIVE_QUALIFICATION')
    need(all(spec['frozen_files'].get(n)==pin for n,pin in qualified['frozen_files'].items()
        if n!=qualified['model_path']), 'SAME_FROZEN_NATIVE_CONTROL_AND_SAFETY_AFTER_QUALIFICATION')


def episode_config(spec):
    validate(spec)
    document=scene.pinned(dict(path=spec['episode_manifest'],sha256=spec['episode_manifest_sha256']))
    episode=scene.prefix_episode(document,spec)
    episode,source=common.operational.recipe_config(spec,episode,actual_phase=PHASE,
        actual_kind=KIND,source_members=SOURCE_MEMBERS)
    return episode,dict(source,stable_task_scene_reference=spec['stable_task_scene_reference'])


def episode_limit(spec,previous):
    episode_config(spec);common.audit_prefix_predecessor(spec,previous)
    pair.preflight(spec)
    return E


def validate_adapter_install(adapter,action):
    from .fusion_trajectory_adapter import TrajectoryAdaptation
    protocol.declared(adapter.spec);step=adapter.observe()['step']
    active=getattr(adapter,'stable_task_active_cycle',None)
    if active is None:
        need(step<B,'CALIBRATION_PRECEDES_FIRST_STUDENT_QUERY')
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
        'EXACT_CURRENT_CYCLE_FULL_REFERENCE_AND_ORIGINAL_BASE_HOLD')
    return dict(status='ORIGINAL_CERTIFICATE_AND_ACK_STILL_REQUIRED',native_step=step,
        cycle_index=c['index'],physics_advance_authorized=False)


def _continue_native(adapter,report,samples,windows):
    from scripts.pi0292.relay_protocol import unseal
    from .stable_task_data_v1 import capture_input
    from .fusion_observation import shadow_inference_payload
    from .fusion_task_phase_live_skill_native_v8 import _noncontact
    from .fusion_task_reference_diagnostic_v1 import require_standing
    from .stable_task_closed_loop_v1 import _link_replacement
    from .stable_task_policy_v1 import build_for_delta
    validate(adapter.spec);ready=report['task_phase_initialization']['readiness']
    need(adapter.observe()['step']==B and not adapter.trace_sealed
        and all(v is True for k,v in ready['predicates'].items()if k!='full_grade')
        and report['all_40_knots_consumed']is True and len(windows)==(B-2503)//25,
        'ACTUAL_UNCHANGED_CALIBRATION_BEFORE_GR00T')
    readiness=tasks.require_start_readiness(adapter,samples[-1])
    samples,windows=list(samples),list(windows)
    initial=dict(qpos=np.asarray(samples[-1]['qpos']).copy());cube=np.asarray(samples[-1]['object_pose']).copy()
    cycles=[];observations=[];queries=[]
    def observe(sample):_noncontact(sample,cube);require_standing(sample,initial)
    for index,step in enumerate(consumer.observation_steps(adapter.spec)):
        need(adapter.observe()['step']==samples[-1]['native_step']==step,'FRESH_CURRENT_NATIVE_QUERY_CLOCK')
        observe(samples[-1])
        observation=capture_input(adapter,language=tasks.LANGUAGES[adapter.spec['task_command']],
            task_score=tasks.score_contract(adapter.spec['task_command']))
        need(observation['native_step']==step and abs(observation['native_time_s']-step*.002)<=1e-8,
             'CAPTURE_CURRENT_CYCLE_BEFORE_QUERY')
        observations.append(observation)
        action,pin,returned=pair.query_from_native(adapter,observation,index)
        queries.append(returned)
        write_json(adapter.output/f'TARGET_GR00T_QUERY_RETURN_{index:02d}.private.json',
            dict(returned,observation=observation,action_sha256=pin))
        record=json.loads(Path(observation['path']).read_bytes());body=unseal(record)
        payload=shadow_inference_payload(body['original_shadow_ABI'])
        frame=unseal(body['original_shadow_ABI'])['source_policy_view']['T_source_from_target']
        program,proof=protocol.load_generated_proposal(action,pin,
            query_state10=payload['state_eef10'],source_from_target=frame,
            actual_TCP=samples[-1]['tcp'],consumed_reference_TCP=adapter.reference_history[-1]['pose'],
            base_hold=adapter.base_pose_hold,actual_cube=cube)
        proof.update(query=returned,observation=observation,cycle_index=index,
            policy=adapter.spec['target_gr00t_policy'],desired_outcome_provenance='ACTUAL_GR00T_NOT_EXPERT_LABEL')
        write_json(adapter.output/f'TARGET_GR00T_PROPOSAL_{index:02d}.private.json',proof)
        current,samples,windows=consumer.run_prefix(adapter,program,index=index,
            past_samples=samples,past_windows=windows,sample_observer=observe)
        if cycles:_link_replacement(adapter,cycles[-1],current)
        cycles.append(current)
    # Explicit controller termination, not a third model query or Teacher skill.
    program,hold=build_for_delta(samples[-1]['tcp'],adapter.reference_history[-1]['pose'],
        adapter.base_pose_hold,np.zeros(3))
    write_json(adapter.output/'TARGET_GR00T_TERMINATION_HOLD.private.json',
        dict(hold,kind='DETERMINISTIC_TERMINATION_HOLD',GR00T_queries=0,teacher_fallback=False))
    current,samples,windows=consumer.run_prefix(adapter,program,index=2,
        past_samples=samples,past_windows=windows,sample_observer=observe)
    _link_replacement(adapter,cycles[-1],current);cycles.append(current)
    # Complete responses survive process exit; require actual worker0 at the
    # final finite boundary, rather than confusing a reply with full closeout.
    from .fusion_paused_query_bridge import _registered
    gpu_root=Path(adapter.spec['target_gr00t_gpu_root']);gpu_path=gpu_root/'EXECUTION_SPEC.json'
    gpu=json.loads(gpu_path.read_bytes())
    while json.loads((gpu_root/'control/LEASE.json').read_bytes())['status']=='ACTIVE':
        adapter.guard();adapter.spin(.15)
    _registered(gpu,gpu_path,closed=True)
    score=tasks.score([s for s in samples if s['native_step']>=B],adapter.spec['task_command'])
    return jsonable(dict(copy.deepcopy(report),schema='WM02_TARGET_GR00T_NATIVE_PREFIX_ACTUAL_V1',
        actual_spec_phase=PHASE,native_endpoint_step=E,data_role=adapter.spec['data_role'],
        data_parent=adapter.spec['data_parent'],actual_scene_sha256=digest(adapter.spec['model_path']),
        decision_start_readiness=readiness,target_gr00t_prefix_protocol=protocol.protocol(),
        initialization_all40_knots_consumed=report['all_40_knots_consumed'],
        initialization_adaptation=report.get('adaptation'),all_40_knots_consumed=False,
        raw_knot_consumed=[False]*40,GR00T_raw_H40_consumed=False,
        generated_cycles=cycles,policy_observations=observations,queries=queries,
        windows=windows,complete_task_atoms=len(windows),task_native_ticks=E-2503,
        solver_target_confirmed_atoms=len(windows),gripper_confirmed_atoms=len(windows),
        policy_queries=2,actual_VLA_inferences=2,actual_VLA_commands=2,
        VLA_queries=2,VLA_inferences=2,VLA_commands=2,
        unique_generated_action_chunks=2,VLA_action_installations=sum(len(c['installs'])for c in cycles[:2]),
        GR00T_updates=0,WM_updates=0,Actor_updates=0,TASK_ADAPTER_updates=0,TASK_ADAPTER_inferences=0,
        native_process_Processor_decodes=0,additional_action_decodes=0,teacher_fallbacks=0,
        raw_reference_is_GR00T=True,target_gr00t_policy=adapter.spec['target_gr00t_policy'],
        capability_score=score['original_approach_score'],target_gr00t_task_score=score,
        task_condition_score_contract=tasks.score_contract(adapter.spec['task_command']),
        task_condition_success=score['task_success'],expert_training_eligible=False,
        full_task_success=False,target_task_success=False,real_time_deployment=False,
        assistance='NATIVE_QUOTA_PAUSED_REAL_OBSERVATION',native_clock_rebased=False,
        full_MPC_hidden_state_restored=False,source_task_phase_equivalence=False,
        final_HOLD_cancel_status='REQUIRES_OUTER_ORIGINAL_CANCEL_AND_FINAL_TRACE'))


def continue_native(adapter,report,samples,windows):
    try:return _continue_native(adapter,report,samples,windows)
    except BaseException as error:
        replies=[]
        gpu_root=Path(adapter.spec['target_gr00t_gpu_root'])/'fit-01/queries'
        for i in range(2):
            p=gpu_root/f'{i:02d}'/'RESULT.json'
            if p.is_file():replies.append(dict(index=i,result_sha256=digest(p)))
        try:step=adapter.observe()['step']
        except Exception:step=None
        write_json(adapter.output/'TARGET_GR00T_LOOP_FAILED.private.json',dict(
            failure=type(error).__name__+': '+str(error),last_native_step=step,
            actual_paired_GPU_query_records=len(replies),paired_GPU_records=replies,
            query_records_are_not_native_consumption=True,teacher_fallbacks=0,
            GR00T_updates=0,WM_updates=0,Actor_updates=0,retry_authorized=False))
        raise


def complete_consumption(spec,complete):
    protocol.declared(spec)
    if not(complete.get('target_gr00t_prefix_protocol')==protocol.protocol()
        and complete.get('policy_queries')==complete.get('actual_VLA_inferences')==2
        and complete.get('actual_VLA_commands')==complete.get('unique_generated_action_chunks')==2
        and complete.get('teacher_fallbacks')==0 and complete.get('raw_reference_is_GR00T')is True
        and complete.get('target_gr00t_policy')==spec['target_gr00t_policy']
        and complete.get('GR00T_raw_H40_consumed')is False
        and complete.get('all_40_knots_consumed')is False
        and complete.get('raw_knot_consumed')==[False]*40
        and complete.get('native_process_Processor_decodes')==complete.get('additional_action_decodes')==0
        and complete.get('GR00T_updates')==complete.get('WM_updates')
            ==complete.get('Actor_updates')==complete.get('TASK_ADAPTER_updates')==0):return False
    observations=complete.get('policy_observations')or[]
    if [o.get('native_step')for o in observations]!=list(consumer.observation_steps(spec)):return False
    cycles=complete.get('generated_cycles')or[]
    if len(cycles)!=3:return False
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
        report_schema='WM02_TARGET_GR00T_LOOP_CALIBRATION_PREFIX_V1',
        goal_schema='WM02_TARGET_GR00T_LOOP_CALIBRATION_GOAL_V1',
        decision_schema='WM02_TARGET_GR00T_LOOP_CALIBRATION_DECISION_V1',
        feedback_report_schema='WM02_TARGET_GR00T_LOOP_CALIBRATION_FEEDBACK_V1',continuation=continue_native)
