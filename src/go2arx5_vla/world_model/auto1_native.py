"""AUTO1's thin binding to the existing native simulator and physical owner.

No scheduling or native safety replacement: each invocation owns one bounded
stage, retains the existing wrapper/ROS entry and returns actual evidence.
"""
from pathlib import Path
import json,os,time
from contextlib import contextmanager

PHASES=(('cold-1',50),('cold-2',50),('hold-2s-1',1000),('hold-2s-2',1000),('hold-5s-1',2500),('hold-5s-2',2500))

def validate_parent_loader(expected,actual):
    # glibc's loader reads LD_LIBRARY_PATH at exec, before importing rclpy.
    # Setting it later only for native children cannot repair the parent.
    if not expected or actual!=expected:raise PermissionError('AUTO1_NATIVE_PARENT_LOADER_ENV_NOT_BOUND_AT_EXEC')

def phase_limit(phase,previous,spec=None):
    if phase=='collect-target-task-learning-grasp-v1':
        from .task_learning_grasp_native_v1 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-teacher-prefix-diagnostic-v1':
        from .target_teacher_prefix_diagnostic_v1 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-gr00t-prefix-loop-v1':
        from .target_gr00t_native_loop_v1 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-stable-task-closed-loop-v1':
        from .stable_task_closed_loop_v1 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-stable-task-eval-v1':
        from .stable_task_evaluation_v1 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-stable-task-data-v1':
        from .stable_task_data_v1 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-stable-start-short-v1':
        from .stable_start_short_task_v1 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-c02-source-state-feedback-v2':
        from .fusion_c02_source_state_feedback_v2 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-c02-source-state-initialization-v1':
        from .fusion_c02_source_state_initialization_v1 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-task-phase-live-skill-v9':
        from .fusion_task_phase_live_skill_contract_v9 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-task-phase-live-skill-v8':
        from .fusion_task_phase_live_skill_contract_v8 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-task-phase-bias-fade-v8':
        from .fusion_task_phase_bias_fade_contract_v8 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-task-phase-pose-continuation-v7':
        from .fusion_task_phase_pose_continuation_contract_v7 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-task-phase-pose-feedback-v6':
        from .fusion_task_phase_pose_feedback_contract_v6 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-task-phase-xyz-feedback-continuation-v5':
        from .fusion_task_phase_xyz_feedback_continuation_contract_v5 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-task-phase-xyz-feedback-v4':
        from .fusion_task_phase_xyz_feedback_contract_v4 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-task-phase-bias-continuation-v3':
        from .fusion_task_phase_bias_continuation_contract_v3 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-task-phase-bias-calibration-v2':
        from .fusion_task_phase_bias_calibration_contract_v2 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-task-phase-initialization-v1':
        from .fusion_task_phase_initialization_contract import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-source-posture-query-v3':
        from .fusion_source_posture_paused_query_v3 import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-source-posture-qualification-v1':
        from .fusion_source_posture_qualification_contract import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-live-proposal-qualification-v3':
        from .fusion_live_proposal_qualification_contract import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-paused-query-v2':
        from .fusion_paused_query_native import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-trajectory-response-v1':
        from .fusion_trajectory_response_native_contract import episode_limit
        return episode_limit(spec,previous)
    if phase in('collect-target-fusion-trajectory-qualification-v1','collect-target-fusion-trajectory-slow-qualification-v2'):
        from .fusion_trajectory_qualification_contract import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-fusion-reference-qualification-v1':
        from .fusion_reference_qualification_contract import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-pilot2-supported':
        from .pilot2_supported import validate
        return validate(spec,previous)
    if phase=='collect-target-pilot1-supported-close':
        from .pilot1_supported_close import validate
        return validate(spec,previous)
    if phase=='collect-target-pilot1-contact':
        from .pilot1_contact import validate
        return validate(spec,previous)
    if phase=='collect-target-pilot1':
        from .pilot1_episode import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-next2-contact':
        from .next2_contact import validate
        return validate(spec,previous)
    if phase=='collect-target-next2':
        from .next2_episode import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-next1':
        from .next1_episode import episode_limit
        return episode_limit(spec,previous)
    if phase=='collect-target-night1':
        from .target_manifest import episode_limit
        return episode_limit(spec,previous)
    if phase.startswith('collect-target-')or phase in ('g2-hold-short','g2-hold-renew','g2-eef-x','g2-eef-y','g2-standing-hold-short-v2','g2-standing-hold-renew-v2','g2-standing-eef-x-v2','g2-standing-eef-y-v2','g2-standing-ttl-v2'):
        from .target_reference import dynamic_limit
        return dynamic_limit(phase,previous,spec)
    if phase=='g2-reference-qualification':
        if(previous is None or previous.get('phase')!='hold-5s-2'or previous.get('status')!='NATIVE_STAGE_PASSED'or not previous.get('shutdown_clean')or any(p['exit_code']!=0 for p in previous.get('owned_process_exits',[]))):raise PermissionError('COMPLETE_G1_REQUIRED_BEFORE_REFERENCE_QUALIFICATION')
        return 2503
    names=[n for n,_ in PHASES]
    if phase not in names:raise PermissionError('UNKNOWN_AUTO1_NATIVE_PHASE')
    index=names.index(phase)
    if index and(previous is None or previous.get('phase')!=names[index-1]or previous.get('status')!='NATIVE_STAGE_PASSED'or any(p.get('exit_code')!=0 for p in previous.get('owned_process_exits',[]))):raise PermissionError('PREVIOUS_NATIVE_STAGE_NOT_PASSED')
    return PHASES[index][1]

def validate_scope(spec,lease,registry,*,host,uid,now):
    root=Path(spec['campaign_root']);binding=Path(spec['campaign_binding'])
    from .storage import digest
    if spec.get('diagnostic_debugger')and(spec.get('standby_shutdown_diagnostic')is not True or spec['phase']!='cold-1'):raise PermissionError('DEBUGGER_ONLY_FOR_ZERO_TICK_STANDBY_DIAGNOSIS')
    wall_bound=1200 if spec.get('phase')in('collect-target-fusion-c02-source-state-feedback-v2','collect-target-fusion-c02-source-state-initialization-v1','collect-target-fusion-trajectory-slow-qualification-v2','collect-target-fusion-trajectory-response-v1','collect-target-fusion-paused-query-v2','collect-target-fusion-live-proposal-qualification-v3','collect-target-fusion-source-posture-qualification-v1','collect-target-fusion-source-posture-query-v3','collect-target-fusion-task-phase-initialization-v1','collect-target-fusion-task-phase-bias-calibration-v2','collect-target-fusion-task-phase-bias-continuation-v3','collect-target-fusion-task-phase-xyz-feedback-v4','collect-target-fusion-task-phase-xyz-feedback-continuation-v5','collect-target-fusion-task-phase-pose-feedback-v6','collect-target-fusion-task-phase-pose-continuation-v7','collect-target-fusion-task-phase-bias-fade-v8','collect-target-fusion-task-phase-live-skill-v8','collect-target-fusion-task-phase-live-skill-v9','collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1','collect-target-gr00t-prefix-loop-v1','collect-target-teacher-prefix-diagnostic-v1','collect-target-task-learning-grasp-v1')else 600
    if (spec['domain']!='WM02_AUTO1_TARGET_COMMISSION_V1'or spec['physics']is not True
            or spec['site_profile']!='4080-simulation'or spec['owner']!='EXISTING_LOCAL_PC'
            or host!='corelab-4080s'or uid!=2009 or spec['host']!=host or spec['uid']!=uid
            or lease.get('status')!='ACTIVE'or lease.get('scope')!='WM02_AUTO1_SIM_ONLY'
            or lease.get('owner')!=spec['owner']or lease.get('run_uuid')!=root.name
            or lease.get('binding_sha256')!=digest(binding)or now>=lease['deadline_epoch']
            or (root/'STOP.json').exists()or not 0<spec['wall_limit_s']<=wall_bound):raise PermissionError('AUTO1_REGISTERED_SCOPE_HOST_LEASE_OR_TIME_INVALID')
    if not any(e.get('scope')=='WM02_AUTO1_SIM_ONLY'and e.get('owner')==spec['owner']and e.get('path')==str(root)and e.get('local_lease')==str(root/'control/LEASE.json')for e in registry['entries']):raise PermissionError('AUTO1_ACTUAL_REGISTRY_BINDING_REQUIRED')

@contextmanager
def admitted_target(spec_path,run_id,output):
    from scripts.pi0292.durable import OwnerLock
    from .runtime import admitted_run
    from .storage import digest
    spec=json.loads(Path(spec_path).read_text());campaign=Path(spec['campaign_root'])
    previous=json.loads(Path(spec['previous_stage']).read_text())if spec.get('previous_stage')else None
    if previous is not None:
        if (digest(spec['previous_stage'])!=spec['previous_stage_sha256']or previous.get('campaign_uuid')!=campaign.name
                or previous.get('native_binary_sha256')!=native_prefix_predecessor(spec)or previous.get('observer_sha256')!=digest(spec['observer_library'])):raise PermissionError('PREVIOUS_STAGE_NOT_BOUND_TO_THIS_CANDIDATE')
    if type(spec['candidate_index'])is not int or spec['candidate_index']<1 or(not(spec.get('night1_binding')or spec.get('next1_binding')or spec.get('next2_binding')or spec.get('pilot1_binding'))and spec['candidate_index']>6):raise PermissionError('ACTUAL_CANDIDATE_IDENTITY_REQUIRED')
    limit=phase_limit(spec['phase'],previous,spec)
    if spec['max_ticks']!=limit:raise PermissionError('AUTO1_EXACT_PHASE_TICK_LIMIT_REQUIRED')
    if spec['phase'].startswith(('g2-standing-','collect-target-')):
        repair=Path(spec['repair_stop_result']);negative=json.loads(repair.read_text())
        if (spec.get('reference_binding_version')!='MEASURED_EEF_AND_BASE_POSE_HOLD_V2'
                or digest(repair)!=spec['repair_stop_result_sha256']
                or negative.get('campaign_uuid')!=campaign.name
                or negative.get('native_binary_sha256')!=standing_repair_ancestor(spec)
                or negative.get('phase')!='g2-hold-renew'
                or negative.get('actual_target_integrations')!=2853
                or negative.get('physics_stops')!=[dict(kind='stop',reason='ILLEGAL_CONTACT',steps=2853)]
                or not negative.get('shutdown_clean')):raise PermissionError('EVIDENCED_STANDING_REFERENCE_REPAIR_REQUIRED')
    if spec['phase'].startswith('collect-target-b-'):
        result=Path(spec['initial_WM_result']);trained=json.loads(result.read_text())
        if (digest(result)!=spec['initial_WM_result_sha256']or trained.get('status')!='TARGET_WM_ENGINEERING_ENTRY_PASSED'
                or trained.get('main_updates')!=200 or trained.get('continuation_probe_updates')!=1
                or digest(spec['initial_WM_checkpoint'])!=trained.get('checkpoint_sha256')):raise PermissionError('REAL_TARGET_WM200_AND_RESTORE_REQUIRED_BEFORE_BATCH_B')
    validate_scope(spec,json.loads((campaign/'control/LEASE.json').read_text()),json.loads(Path(spec['registry']).read_text()),host=os.uname().nodename,uid=os.getuid(),now=time.time())
    if spec.get('pilot1_binding'):
        from .pilot1_scope import validate
        validate(spec,json.loads(Path(spec['registry']).read_text()))
    if spec.get('next2_binding'):
        from .next2_scope import validate
        validate(spec,json.loads(Path(spec['registry']).read_text()))
    if spec.get('next1_binding'):
        from .next1_scope import validate
        validate(spec,json.loads(Path(spec['registry']).read_text()))
    if spec.get('night1_binding'):
        from .target_manifest import validate_night1
        validate_night1(spec,json.loads(Path(spec['registry']).read_text()))
    for path,pin in spec['frozen_files'].items():
        if digest(path)!=pin:raise PermissionError('FROZEN_TARGET_INPUT_CHANGED:'+path)
    old=Path(spec['historical_target_lease'])
    if digest(old)!=spec['historical_target_lease_sha256']or json.loads(old.read_text())['status']!='CLOSED':raise PermissionError('HISTORICAL_LEASE_CHANGED')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU_TARGET_COMMISSIONING_ONLY')
    validate_parent_loader(spec['native_environment']['LD_LIBRARY_PATH'],os.environ.get('LD_LIBRARY_PATH'))
    if str(Path(spec['xvfb_run']))not in spec['frozen_files']or not os.access(spec['xvfb_run'],os.X_OK):raise PermissionError('BOUND_EXISTING_XVFB_ENTRY_REQUIRED')
    if spec.get('corelab_world_binding_v3') is not None:
        from .corelab_world_owner_v3 import world_owner
        registry = json.loads(Path(spec['registry']).read_text())
        with world_owner(spec, registry):
            with admitted_run(spec_path, run_id, 'wm02-auto1-target', output, physics=True):
                yield spec
        return
    if spec.get('parallel_batch_spec'):
        from .target_parallel import inherited_batch_owner
        with inherited_batch_owner(spec_path,spec):
            with admitted_run(spec_path,run_id,'wm02-auto1-target',output,physics=True):yield spec
        return
    campaign_lock=OwnerLock(campaign/'OWNER.lock')
    try:
        native_lock=OwnerLock(spec['physical_owner_lock'])
        try:
            with admitted_run(spec_path,run_id,'wm02-auto1-target',output,physics=True):yield spec
        finally:native_lock.close()
    finally:campaign_lock.close()

def native_prefix_predecessor(spec):
    """Retain a historical prefix's real identity after the exact sensor repair.

    The observer, outer wrapper and enlarged trace remain the previously bound
    runtime libraries. This only admits a new construction/qualification run;
    no historical result is rewritten as a pass for the repaired executable.
    """
    from .storage import digest
    if spec.get('native_ui_build')or spec.get('native_ui_rate'):
        from .native_ui_rate_v1 import baseline_spec
        spec=baseline_spec(spec)
    actual=digest(spec['native_binary'])
    if not spec.get('sensor_handoff_previous_candidate_receipt'):return actual
    from .native_state_snapshot import VERSION
    if spec.get('phase')!='collect-target-task-learning-grasp-v1':
        raise PermissionError('SENSOR_SUCCESSOR_ONLY_FOR_FRESH_TASK_LEARNING_NATIVE_RUNS')
    standing_repair_ancestor(spec)
    build=json.loads(Path(spec['cold_history_build_spec']).read_text())
    path=Path(spec['sensor_handoff_previous_candidate_receipt'])
    prior=json.loads(path.read_text());root=path.parent
    if (build.get('robot_state_handoff')!=VERSION
            or digest(path)!=spec.get('sensor_handoff_previous_candidate_receipt_sha256')
            or root/'rl_sim_mujoco.cpp'!=Path(build['robot_state_handoff_predecessor_cpp'])
            or prior.get('status')!='INIT_CANDIDATE_COMPILED_AND_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED'
            or prior.get('candidate_sources',{}).get('rl_sim_mujoco.cpp')!=digest(root/'rl_sim_mujoco.cpp')
            or prior.get('candidate_sources',{}).get('rl_sdk.cpp')!=digest(root/'rl_sdk.cpp')
            or prior.get('binary_sha256')!=digest(root/'rl_sim_mujoco')
            or digest(root/'observer.so')!=digest(spec['observer_library'])
            or digest(root/'task_actuation_r1.so')!=digest(spec['outer_library'])):
        raise PermissionError('EXACT_OLD_PREFIX_BINARY_AND_UNCHANGED_RUNTIME_LIBRARIES_REQUIRED')
    return prior['binary_sha256']


def standing_repair_ancestor(spec):
    """Bind the preserved reference failure to a math-preserving runtime successor."""
    from .storage import digest
    if spec.get('native_ui_build')or spec.get('native_ui_rate'):
        from .native_ui_rate_v1 import baseline_spec
        spec=baseline_spec(spec)
    actual=digest(spec['native_binary'])
    if not spec.get('cold_history_build_receipt'):return actual
    for key in('cold_history_build_receipt','cold_history_build_spec'):
        if digest(spec[key])!=spec[key+'_sha256']:raise PermissionError('COLD_HISTORY_BUILD_PROOF_CHANGED')
    receipt=json.loads(Path(spec['cold_history_build_receipt']).read_text());build=json.loads(Path(spec['cold_history_build_spec']).read_text())
    pair=build.get('command_pair_coherence')
    handoff=build.get('robot_state_handoff')
    if handoff is not None:
        from .native_state_snapshot import VERSION,patch as sensor_patch
        root=Path(spec['cold_history_build_receipt']).parent
        old=Path(build['robot_state_handoff_predecessor_cpp'])
        old_sdk=Path(build['robot_state_handoff_predecessor_sdk'])
        header=Path(spec['source_root'])/'src/go2arx5_vla/target_mujoco/state_snapshot.hpp'
        runtime_trace=old.parent/'libwm02_init_trace.so'
        if (handoff!=VERSION or receipt.get('robot_state_handoff')!=VERSION
                or digest(old)!=build.get('robot_state_handoff_predecessor_cpp_sha256')
                or digest(old_sdk)!=build.get('robot_state_handoff_predecessor_sdk_sha256')
                or digest(root/'state_snapshot.hpp')!=digest(header)
                or digest(header)!=receipt.get('state_snapshot_header_sha256')
                or (root/'rl_sim_mujoco.cpp').read_text()!=sensor_patch(old.read_text())
                or (root/'rl_sdk.cpp').read_text()!=old_sdk.read_text()
                or receipt.get('robot_state_handoff_inverse_text_identical')is not True
                or receipt.get('robot_state_handoff_PD_waits')is not False
                or receipt.get('robot_state_handoff_sensor_filters')is not False
                or build.get('sensor_handoff_frozen_trace_dependency')!=str(runtime_trace)
                or build.get('frozen_files',{}).get(str(runtime_trace))!=digest(runtime_trace)
                or receipt.get('sensor_handoff_frozen_trace_dependency_sha256')!=digest(runtime_trace)):
            raise PermissionError('EXACT_LOCKFREE_SENSOR_SNAPSHOT_MATH_PRESERVING_PROOF_REQUIRED')
    clock=build.get('physical_callback_clock')
    if clock is not None:
        header=Path(spec['cold_history_build_receipt']).parent/'control_clock.hpp'
        if(clock not in ('FROZEN_PERIOD_PD_POLICY_PHYSICAL_CLOCK_V1','FROZEN_PERIOD_PD_POLICY_APPLIED_CLOCK_V2')or receipt.get('physical_callback_clock')!=clock
                or not header.is_file()or digest(header)!=receipt.get('control_clock_header_sha256')):raise PermissionError('EXACT_PHYSICAL_CALLBACK_CLOCK_SUCCESSOR_PROOF_REQUIRED')
        if clock=='FROZEN_PERIOD_PD_POLICY_APPLIED_CLOCK_V2':
            applied_header=header.parent/'applied_control_clock.hpp'
            if not applied_header.is_file()or digest(applied_header)!=receipt.get('applied_control_clock_header_sha256'):raise PermissionError('EXACT_APPLIED_CLOCK_SUCCESSOR_PROOF_REQUIRED')
    if pair is not None:
        header=Path(spec['cold_history_build_receipt']).parent/'command_pair.hpp'
        if(pair!='SHORT_PAIR_PUBLICATION_PD_TRYLOCK_V1'or receipt.get('command_pair_coherence')!=pair
                or not header.is_file()or digest(header)!=receipt.get('command_pair_header_sha256')):raise PermissionError('EXACT_COMMAND_PAIR_SUCCESSOR_PROOF_REQUIRED')
    if (receipt.get('status')!='INIT_CANDIDATE_COMPILED_AND_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED'
            or receipt.get('binary_sha256')!=actual
            or receipt.get('cold_history_init')!='ORIGINAL_BUFFER_RESET_FIRST_REAL_OBSERVATION_V1'
            or receipt.get('ongoing_control_math_unchanged')is not True
            or receipt.get('original_R1_math_assertions_pass')is not True
            or receipt.get('source_sha')!=build.get('source_sha')
            or receipt.get('original_SDK_sha256')!=digest(build['sdk_source'])
            or not build.get('ancestor_binary_sha256')):raise PermissionError('EXACT_COLD_HISTORY_SUCCESSOR_PROOF_REQUIRED')
    return build['ancestor_binary_sha256']

def task_observer_source(text):
    """Restore the exact pre-existing task pair rule omitted by the generic build."""
    old='    illegal_contact|=!allowed;'
    new='''    auto pair=[&](const char* x,const char* y) {
      return a && b && ((std::string(a)==x && std::string(b)==y) || (std::string(a)==y && std::string(b)==x));
    };
    bool task_pair=pair("task_table","task_cube_geom") || pair("task_left_pad","task_cube_geom") || pair("task_right_pad","task_cube_geom");
    allowed=allowed || task_pair;
    illegal_contact|=!allowed || (task_pair && d->contact[i].dist < -.004);'''
    if text.count(old)!=1:raise ValueError('Exact generic observer task-adaptation anchor required')
    return text.replace(old,new)
