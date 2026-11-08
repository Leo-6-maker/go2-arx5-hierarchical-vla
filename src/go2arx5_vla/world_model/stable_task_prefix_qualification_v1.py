"""Read-only qualification from real new TRAIN and sealed candidate DEV.

The small timing probe uses the real learned export and deployable TRAIN input.
It neither integrates physics nor turns old diagnostics into training data.
"""
import time
import os
from pathlib import Path
import numpy as np
from . import stable_task_prefix_v1 as prefix
from . import stable_task_dataset_v1 as dataset
from . import stable_task_evaluation_v1 as evaluation
from . import stable_task_policy_v1 as policy
from .storage import digest

SCHEMA='WM02_STABLE_TASK_PREFIX_FROM_REAL_TRAIN_V1'


def sealed_native(descriptor):
    result,_=dataset.load(descriptor['path'],descriptor['sha256'])
    root=Path(descriptor['path']).parent.parent
    spec,spec_id=dataset.load(root/'EXECUTION_SPEC.json',maximum=16*1024**2)
    lease,_=dataset.load(root/'control/LEASE.json',maximum=1024**2)
    worker,_=dataset.load(root/'WORKER_EXIT.private.json',maximum=1024**2)
    trace=result.get('trace_audit')or{};writer=trace.get('writer')or{}
    prefix.need(lease.get('status')=='CLOSED'and lease.get('spec_sha256')==spec_id['sha256']
        and lease.get('worker_exit_code')==worker.get('exit_code')==0 and worker.get('failure')is None
        and result.get('status')=='NATIVE_STAGE_PASSED'and result.get('phase')==spec['phase']
        and result.get('native_binary_sha256')==dataset.NATIVE and result.get('failure')is None
        and result.get('shutdown_clean')is True
        and {p.get('name')for p in result.get('owned_process_exits',[])}=={'sim','controller','solver','planner'}
        and all(p.get('exit_code')==0 for p in result['owned_process_exits'])
        and result.get('actual_target_integrations')==spec['max_ticks']
        and result.get('original_joint_or_warning_samples')==result.get('native_outer_guard_rejection_count')==0
        and not result.get('physics_stops')and result.get('trace_audit_error')is None
        and trace.get('status')=='RECORDED_TRACE_CHECKS_PASS'
        and trace.get('native_step_records')==spec['max_ticks']
        and writer.get('dropped')==0 and writer.get('events')==writer.get('written'),
        'REAL_CLOSED_SAFE_COMPLETE_NATIVE_EVIDENCE')
    return root,spec,result


def verify_inputs(candidate,train_descriptor,dev_descriptor):
    trained=evaluation.validate_trained_candidate(candidate)
    manifest,_=dataset.load(trained['spec']['replay'],trained['spec']['replay_sha256'],maximum=32*1024**2)
    prefix.need(trained['result']['provenance']['dataset_sha256']==trained['spec']['replay_sha256']
        and manifest.get('schema')==dataset.SCHEMA
        and manifest.get('status')=='COMPLETE_ALL_EXPECTED_EPISODES_TERMINAL',
        'EXACT_REAL_TRAINING_DATASET_NOT_DIAGNOSTIC_REPLAY')
    root,spec,result=sealed_native(train_descriptor)
    prefix.need(spec.get('phase')==dataset.data.PHASE and spec.get('kind')==dataset.data.KIND
        and spec.get('data_role')=='TRAIN'and spec.get('task_command')=='APPROACH_RED_4MM_OPEN'
        and (result.get('reference_qualification')or{}).get('task_condition_success')is True,
        'REAL_SUCCESSFUL_NEW_TRAIN_APPROACH_TIMING')
    matches=[r for r in manifest['episodes']if r['root']==str(root)]
    prefix.need(len(matches)==1 and matches[0]['files']['native_result']==train_descriptor,
        'PREFIX_CALIBRATION_IS_ACTUAL_CANDIDATE_TRAIN_MEMBER')
    panel,_=dataset.load(manifest['panel']['path'],manifest['panel']['sha256'],maximum=16*1024**2)
    row=next(r for r in panel['assignments']if r['native_run']==root.name)
    rebuilt=dataset.read_episode(root,row,manifest['panel']['sha256'])
    prefix.need(rebuilt['files']==matches[0]['files'],'REBUILT_REAL_TRAIN_REFERENCE_AND_INPUT')
    _,dev_spec,dev_result=sealed_native(dev_descriptor)
    evaluation.validate(dev_spec)
    q=dev_result['reference_qualification']
    prefix.need(dev_spec['phase']==evaluation.PHASE and dev_spec['kind']==evaluation.KIND
        and dev_spec['data_role']=='DEV'and q.get('candidate_export')==candidate['export']
        and q.get('candidate_training_result')==candidate['training_result']
        and q.get('capability_reference_all40_consumed')is True
        and q.get('TASK_ADAPTER_inferences')==1 and q.get('TASK_ADAPTER_updates')==0
        and q.get('raw_reference_is_GR00T')is False,
        'SAME_LEARNED_CANDIDATE_ALREADY_REAL_DEV_EVALUATED')
    return rebuilt,root,result,dev_result


def qualify(candidate,train_descriptor,dev_descriptor):
    prefix.need(os.uname().nodename=='corelab-4080s'and os.getuid()==2009,'ACTUAL_TARGET_CPU_FOR_TIMING')
    started,cpu=time.monotonic(),time.process_time()
    example,root,result,dev_result=verify_inputs(candidate,train_descriptor,dev_descriptor)
    arrays=policy.load_export(candidate['export']);x=np.asarray(example['features'],np.float32)
    latencies=[]
    for _ in range(200):
        begin=time.perf_counter();delta=np.asarray(policy.adapter.numpy_predict(arrays,x),float)
        latencies.append(time.perf_counter()-begin)
    q=result['reference_qualification'];windows=q['generated_segment']['windows']
    prefix.need(len(windows)==176 and len(windows[:prefix.PREFIX_ATOMS])==120
        and windows[119]['native_end_step']==prefix.B+25*120
        and all(w['actual_ticks']==25 and w['solver_policies_with_exact_installed_target']
            and w['gripper']['command_consumed']is True for w in windows[:120]),
        'REAL_120_ATOM_TRAIN_PREFIX_TIMING_AND_CONSUMPTION')
    proposal,_=dataset.load(root/'native-01/STABLE_TASK_EXPERT_PROPOSAL.private.json',
        q['executed_reference_proposal_sha256'],maximum=16*1024**2)
    a=proposal['adaptation']
    program,_=policy.build_for_delta(proposal['actual_decision_TCP'],a['initial_reference_TCP'],
        a['measured_fixed_BASE7'],delta)
    return dict(schema=SCHEMA,status='REAL_TRAIN_PREFIX_AND_SEALED_CANDIDATE_DEV_CHECKED_NOT_DYNAMIC_PERMISSION',
        candidate=candidate,TRAIN_native_result=train_descriptor,DEV_native_result=dev_descriptor,
        timing_host=os.uname().nodename,timing_uid=os.getuid(),
        latency_scope='NUMPY_NETWORK_ONLY_REAL_PRECOMPUTED_FEATURES_CAMERA_AND_DISK_NOT_TIMED',
        proposed_protocol=prefix.protocol(),qualified_TRAIN_prefix_native_ticks=3000,
        TRAIN_reference_suffix_atoms=176,source_relative_TTL_s=.4,original_reference_renewal_atoms=7,
        original_atom_ticks=25,physics_dt_s=.002,actual_numpy_inference_calls=200,
        numpy_wall_latency_s=dict(median=float(np.median(latencies)),p95=float(np.percentile(latencies,95)),maximum=max(latencies)),
        candidate_preview_end_atom=program.end_atom,qualification_module_sha256=digest(Path(__file__)),
        DEV_task_success_retained=dev_result['reference_qualification']['task_condition_success'],
        native_integrations=0,optimizer_updates=0,teacher_fallbacks=0,native_execution_authorized=False,
        deployment='PAUSED_ASSISTED_ONLY_REAL_TIME_NOT_QUALIFIED',wall_s=time.monotonic()-started,CPU_s=time.process_time()-cpu)
