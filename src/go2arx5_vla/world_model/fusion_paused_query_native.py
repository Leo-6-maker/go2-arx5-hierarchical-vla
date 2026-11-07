"""One admitted HOLD prefix, actual paused RGB and one separate GPU query.

No VLA proposal is installed. Both scopes must already exist under the original
owner before this native process acquires its original physical/campaign locks.
"""
import json
from pathlib import Path
import numpy as np

from .storage import digest
from .fusion_reference_qualification_contract import hold_preview
from .fusion_reference_qualification_native import require_original_install_ack, consumed_window
from .fusion_trajectory_qualification_native import actual_trace_binding, submit_gripper, confirm_gripper
from .fusion_observation import TargetShadowBinding, transform_hash
from .fusion_short_skill import score_contract, document_hash, task_document

PHASE='collect-target-fusion-paused-query-v2'
SCHEMA='WM02_FUSION_PAUSED_QUERY_EPISODE_V2'
INIT_STEP, END_STEP=2503,3703
SOURCE_MEMBERS=(
    'src/go2arx5_vla/world_model/fusion_paused_query_native.py',
    'src/go2arx5_vla/world_model/fusion_paused_query_bridge.py',
    'src/go2arx5_vla/world_model/fusion_paused_inference.py',
    'src/go2arx5_vla/world_model/fusion_paused_observation.py',
    'src/go2arx5_vla/world_model/fusion_short_skill.py',
    'src/go2arx5_vla/world_model/fusion_observation.py',
    'src/go2arx5_vla/world_model/target_sequential.py',
    'src/go2arx5_vla/world_model/auto1_native.py',
    'src/go2arx5_vla/world_model/fusion_reference_qualification_native.py',
    'src/go2arx5_vla/world_model/fusion_trajectory_qualification_native.py',
    'scripts/wm02/fusion_paused_query_check.py')


def need(value, reason):
    if not value:raise ValueError('PAUSED_QUERY_NATIVE_'+reason)


def episode_config(spec):
    need(spec.get('phase')==PHASE and spec.get('profile')=='wm02-auto1-target'
         and spec.get('domain')=='WM02_AUTO1_TARGET_COMMISSION_V1'
         and spec.get('physics') is True and bool(spec.get('pilot1_binding')),
         'ORIGINAL_AUTO1_P1_NATIVE_ENTRY')
    need(spec.get('max_ticks')==END_STEP and type(spec.get('max_ticks')) is int
         and type(spec.get('wall_limit_s')) in (int,float) and 0<spec['wall_limit_s']<=1200
         and spec.get('WBC_changes') is False and spec.get('physics_dt_s')==.002
         and spec.get('reference_binding_version')=='MEASURED_EEF_AND_BASE_POSE_HOLD_V2',
         'FINITE_HOLD_ONLY_3703_FROZEN_WBC_AND_CLOCK')
    need(not any(spec.get(k) for k in ('fusion_binding','night1_binding','next1_binding',
         'next2_binding','parallel_batch_spec','standby_shutdown_diagnostic',
         'reference_readiness_probe','diagnostic_debugger')), 'SINGLE_NEW_NATIVE_SCOPE_NO_BYPASS')
    source=Path(spec['source_root']);files=spec.get('source_files',{})
    need(all(name in files and digest(source/name)==files[name] for name in SOURCE_MEMBERS),
         'EXACT_NEW_SOURCE_CAPSULE')
    task_path=Path(spec['paused_target_task']);score_path=Path(spec['paused_score_contract'])
    need(digest(task_path)==spec['paused_target_task_file_sha256']
         and digest(score_path)==spec['paused_score_contract_file_sha256'], 'TASK_AND_SCORE_FILE_BYTES')
    task=json.loads(task_path.read_text());contract=json.loads(score_path.read_text())
    need(contract==score_contract() and task==task_document(digest(spec['model_path']))
         and document_hash(task)==spec['paused_target_task_sha256']
         and task['score_contract_sha256']==document_hash(contract), 'NEW_TASK_LANGUAGE_SCENE_SCORE_BINDING')
    path=Path(spec['episode_manifest'])
    need(digest(path)==spec['episode_manifest_sha256'], 'EPISODE_FILE_BYTES')
    episode=json.loads(path.read_text())
    exact=dict(schema=SCHEMA,phase=PHASE,purpose='LIVE_PAUSED_CAPTURE_QUERY_ONLY',
        run=spec['registered_root'],experiment_role='NON_ACTION_QUERY_DIAGNOSTIC',
        split='TRAIN',scene_id='F2T06',parent='FUSION_F2_T06_ROOT_XY_V1',
        decision_native_step=END_STEP,prehold_atoms=48,task_sha256=document_hash(task),
        VLA_action_installations=0,WM_corrections=0,max_queries=1,physics_dt_s=.002)
    need(episode==exact, 'EXACT_HOLD_ONLY_EPISODE_NO_VLA_MOTION')
    catalog_path=Path(spec['scene_allowlist']);catalog=json.loads(catalog_path.read_text())
    scenes=[x for x in catalog['scenes']if x['scene_id']==episode['scene_id']]
    need(digest(catalog_path)==spec['scene_allowlist_sha256'] and len(scenes)==1
         and scenes[0]['parent']==episode['parent'] and scenes[0]['split']=='TRAIN'
         and scenes[0]['model_path']==spec['model_path']
         and scenes[0]['model_sha256']==digest(spec['model_path'])
         and scenes[0]['static_check']=='PASSED_NO_INTEGRATION'
         and catalog['native_binary_sha256']==digest(spec['native_binary']), 'ORIGINAL_STATIC_T06_SCENE')
    build_path=Path(spec['trajectory_trace_capacity_build']);build=json.loads(build_path.read_text())
    capacity=spec.get('trajectory_trace_capacity',262144)
    need(type(capacity) is int and capacity in (262144,524288),'EXPLICIT_TRACE_CAPACITY')
    if capacity==524288:
        from .fusion_trajectory_qualification_contract import validate_trace_capacity_failure
        validate_trace_capacity_failure(spec)
    need(digest(build_path)==spec['trajectory_trace_capacity_build_sha256']
         and build.get('status')==f'TRACE_ONLY_{capacity}_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED'
         and build.get('trace_capacity')==capacity and build.get('new_target_integrations')==0
         and build.get('trace_library_sha256')==digest(spec['trace_library'])
         and build.get('native_binary_sha256')==digest(spec['native_binary'])
         and build.get('observer_sha256')==digest(spec['observer_library'])
         and all(build['source_files'][n]==files[n] for n in (
             'src/go2arx5_vla/target_mujoco/init_trace.cpp',
             'src/go2arx5_vla/target_mujoco/init_trace.hpp')), 'EXACT_EXISTING_TRACE_BUILD_NOT_NEW_PHYSICS_PERMISSION')
    need(transform_hash(spec['paused_source_from_target'])==spec['paused_frame_map_sha256'],
         'EXPLICIT_OWNER_FRAME_MAP_NOT_CAMERA_EQUIVALENCE')
    from .fusion_paused_query_bridge import preflight_pair
    preflight_pair(Path(spec['paused_query_gpu_spec']),spec)
    return episode,task


def episode_limit(spec,previous):
    episode_config(spec)
    path=Path(spec['previous_stage']);root=path.parent.parent
    need(digest(path)==spec['previous_stage_sha256'] and json.loads(path.read_text())==previous,
         'EXACT_NATIVE_PREDECESSOR_BYTES')
    ps=json.loads((root/'EXECUTION_SPEC.json').read_text())
    lease=json.loads((root/'control/LEASE.json').read_text());worker=json.loads((root/'WORKER_EXIT.private.json').read_text())
    report=previous.get('reference_qualification')or{};prior=report.get('episode')or{};trace=previous.get('trace_audit')or{}
    from .fusion_trajectory_qualification_contract import episode_contract,endpoint
    episode_contract(prior)
    need(lease.get('status')=='CLOSED' and lease.get('worker_exit_code')==worker.get('exit_code')==0
         and lease.get('spec_sha256')==digest(root/'EXECUTION_SPEC.json')
         and previous.get('status')=='NATIVE_STAGE_PASSED'
         and previous.get('phase')==ps.get('phase')=='collect-target-fusion-trajectory-slow-qualification-v2'
         and prior.get('geometry_scale')==1. and prior.get('time_stretch_factor')==2
         and prior.get('scene_id')=='F2T06' and report.get('all_40_knots_consumed') is True
         and previous.get('actual_target_integrations')==ps.get('max_ticks')==report.get('native_endpoint_step')
         and previous.get('actual_target_integrations')==endpoint(prior)
         and report.get('complete_task_atoms')==report.get('solver_target_confirmed_atoms')==report.get('gripper_confirmed_atoms')
         and previous.get('shutdown_clean') is True and previous.get('reached_exact_pause') is True
         and previous.get('failure') is None and previous.get('physics_stops')==[]
         and previous.get('original_joint_or_warning_samples')==0
         and previous.get('native_outer_guard_rejection_count')==0
         and previous.get('native_outer_guard_rejections')=={} and previous.get('trace_audit_error') is None
         and trace.get('status')=='RECORDED_TRACE_CHECKS_PASS' and trace.get('errors')==[]
         and trace.get('missing')==[] and (trace.get('writer')or{}).get('dropped')==0
         and trace.get('native_step_records')==previous['actual_target_integrations']
         and previous.get('candidate_index')==spec['candidate_index']
         and previous.get('campaign_uuid')==Path(spec['campaign_root']).name
         and previous.get('native_binary_sha256')==digest(spec['native_binary'])
         and previous.get('observer_sha256')==digest(spec['observer_library'])
         and previous.get('scene_sha256')==digest(spec['model_path'])
         and {p.get('name')for p in previous.get('owned_process_exits',[])}=={'sim','controller','solver','planner'}
         and all(p.get('exit_code')==0 for p in previous['owned_process_exits']),
         'GENUINE_FULL1_NATIVE_SAFETY_TRACE_AND_CLOSEOUT')
    rows=report.get('windows',[])
    need(len(rows)==report['complete_task_atoms']==(previous['actual_target_integrations']-INIT_STEP)//25,
         'ALL_FULL1_ATOMS_RETAINED')
    for i,row in enumerate(rows):
        grip=row.get('gripper')or{};submitted=grip.get('submitted')or{};memory=grip.get('consumed_wrapper')or{};values=memory.get('values')or{}
        need(row.get('native_start_step')==INIT_STEP+25*i and row.get('native_end_step')==INIT_STEP+25*(i+1)
             and row.get('actual_ticks')==25 and bool(row.get('solver_policies_with_exact_installed_target'))
             and grip.get('command_consumed') is True and submitted.get('sequence')==values.get('sequence')==i
             and memory.get('status')=='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY'
             and values.get('steps')==row['native_end_step']
             and values.get('opening')==submitted.get('opening') and values.get('expiry')==submitted.get('expiry'),
             'EVERY_FULL1_POSE_AND_GRIPPER_ATOM')
    return END_STEP


def validate_adapter_install(adapter,action):
    need(hasattr(adapter,'paused_query_initial_tcp'), 'ACTUAL_MEASURED_INITIAL_HOLD')
    need(np.asarray(action).shape==(40,10)
         and np.allclose(action,hold_preview(adapter.paused_query_initial_tcp),rtol=0,atol=1e-12)
         and np.allclose(adapter.base_pose_hold,adapter.paused_query_initial_base,rtol=0,atol=1e-10),
         'ONLY_EXACT_INITIAL_HOLD_NO_VLA_PROPOSAL_INSTALL')
    return dict(status='EXACT_INITIAL_HOLD_ONLY',native_execution_authorized=False)


def run(adapter):
    from scripts.pi0292.durable import write
    from scripts.pi0292.eval_contract import matrix10
    from .target_stability import motion,screen
    from .target_reference import require_live_window
    from .fusion_paused_query_bridge import capture_and_wait_query
    episode,task=episode_config(adapter.spec)
    adapter.prepare_dynamic();initial=adapter.measured_task_state()
    need(initial['native_step']==INIT_STEP,'REAL_INITIAL_2503')
    adapter.paused_query_initial_tcp=initial['tcp'].copy();adapter.paused_query_initial_base=list(adapter.base_pose_hold)
    trace=actual_trace_binding(adapter);samples=[initial];windows=[];acks=[]
    def install():
        ref=adapter.install_reference(hold_preview(initial['tcp']))
        acks.append(require_original_install_ack(ref));return adapter.observe()['step']
    installed=install()
    while adapter.observe()['step']<END_STEP:
        before=adapter.observe();step=before['step']
        if step==3503 or step-installed>=175:installed=install()
        clock=require_live_window(adapter.active_reference['source_mpc_s'],before['time_s'],25)
        opening=adapter.active_reference['action_EEF10'][(step-installed)//25][9]
        grip=submit_gripper(adapter,len(windows),opening);start=len(adapter.ros_evidence)
        after=adapter.advance_to(step+25);adapter.spin(.15);actual=adapter.measured_task_state();samples.append(actual)
        need(actual['native_step']==after['step']==step+25 and actual['safety_valid'] is True
             and actual['native_abort'] is False and adapter.latest['mode'].data=='WBC', 'ORIGINAL_NATIVE_SAFETY_AND_WBC')
        row=consumed_window(adapter,start,before,after,clock);row['gripper']=confirm_gripper(adapter,grip,actual)
        need(bool(row['solver_policies_with_exact_installed_target']) and row['gripper']['command_consumed'],
             'ACTUAL_EACH_HOLD_POSE_AND_GRIPPER_CONSUMPTION')
        windows.append(row);adapter.command_event(dict(phase='paused-query-prefix-consumed',**row))
        adapter.reference_history.append(dict(end_native_step=after['step'],
            pose=matrix10(adapter.active_reference['action_EEF10'])[(step-installed)//25]))
    last=samples[-9:]
    relative=motion(np.asarray([x['qpos'][:3]for x in last]),np.asarray([x['qpos'][3:7]for x in last]),
                    np.asarray([x['tcp'][:3,3]for x in last]))['relative_tcp']
    prefix=screen(np.asarray([x['time_s']for x in last]),relative)
    write(adapter.output/'PAUSED_QUERY_PREFIX.private.json',prefix)
    need(prefix['qualified'],'UNCHANGED_STEADY_PREFIX_SCREEN')
    binding=TargetShadowBinding(target_world_frame='MUJOCO_GO2_ARX5_WORLD',
        scene_sha256=digest(adapter.spec['model_path']),
        target_runtime_sha256=digest(Path(adapter.spec['registered_root'])/'EXECUTION_SPEC.json'),
        T_source_from_target=adapter.spec['paused_source_from_target'],
        expected_transform_sha256=adapter.spec['paused_frame_map_sha256'])
    query=capture_and_wait_query(adapter,binding=binding,task=task,
        task_sha256=document_hash(task),gpu_spec_path=Path(adapter.spec['paused_query_gpu_spec']),capture_index=0)
    need(adapter.observe()['step']==END_STEP,'NO_PHYSICS_DURING_CAPTURE_WAIT_QUERY')
    adapter.capture_loaded_identity();cancel=adapter.cancel_reference();adapter.seal_trace(END_STEP)
    report=dict(schema=SCHEMA,episode=episode,actual_trace_binding=trace,
        complete_task_atoms=len(windows),solver_target_confirmed_atoms=len(windows),
        gripper_confirmed_atoms=len(windows),windows=windows,install_ACKs=acks,
        prefix=prefix,paused_query=query,operator_cancel=cancel,native_endpoint_step=END_STEP,
        VLA_action_installations=0,WM_corrections=0,task_executed=False,
        full_task_success=False,GR00T_updates=0,WM_updates=0,Actor_updates=0)
    write(adapter.output/'PAUSED_QUERY_NATIVE.private.json',report);adapter.reference_qualification=report
