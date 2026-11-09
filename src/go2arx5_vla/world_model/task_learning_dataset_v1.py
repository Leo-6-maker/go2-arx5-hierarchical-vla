"""Continuous label arrays and fixed exposure after native-proof verification.

These pure transformations grant no BC eligibility. A separate sealed exporter
must verify every real capture, consumed command, native exit and population.
"""
from pathlib import Path
import numpy as np
from . import task_learning_grasp_v1 as task
from .task_learning_sequence_v1 import map_policy_frame


def _capture(root,spec,spec_sha,identity,decision):
    from PIL import Image
    from scripts.pi0292.relay_protocol import unseal
    from .stable_task_dataset_v1 import load
    from .fusion_observation import shadow_inference_payload
    from .storage import digest
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10
    path=Path(identity['path']);expected=decision['native_step']
    task.need(path.is_absolute()and path.resolve()==path and not path.is_symlink()
        and path.is_relative_to(root/'native-01/PAUSED_OBSERVATIONS'),'OWN_REAL_CONTINUOUS_CAPTURE')
    record,proof=load(path,identity['sha256'],maximum=16*1024**2);body=unseal(record)
    task.need(body['native_step']==identity['native_step']==expected
        and body['world']['execution_spec_sha256']==spec_sha
        and body['world']['scene_sha256']==digest(spec['model_path'])
        and body['world']['native_output']==str(root/'native-01')
        and body['task']['language']==task.LANGUAGE,'SAME_REAL_CAPTURE_CLOCK_WORLD_AND_LANGUAGE')
    for name,pin in body['files'].items():
        member=path.parent/name
        task.need(member.resolve().is_relative_to(path.parent)and digest(member)==pin,'RETAINED_REAL_MEDIA_AND_STATE_HASHES')
    shadow=unseal(body['original_shadow_ABI']);payload=shadow_inference_payload(body['original_shadow_ABI'])
    frame=np.asarray(shadow['source_policy_view']['T_source_from_target'],float)
    state=np.asarray(payload['state_eef10'],float)
    task.need(shadow['native_step']==expected
        and shadow['target_observation']['target_runtime_sha256']==spec_sha
        and shadow['target_observation']['native_state_sha256']==body['files']['STATE.private.bin']
        and np.array_equal(frame,np.asarray(spec['task_phase_source_from_target']))
        and np.allclose(matrix_from10(state),frame@np.asarray(decision['original_actual_TCP']),rtol=0,atol=1e-12),
        'CURRENT_MEASURED_TCP_WITH_ONE_DECLARED_POLICY_FRAME_NOT_FUTURE_ACHIEVED_STATE')
    images=[]
    for camera in('body_rgb','wrist_rgb'):
        p=path.parent/'images'/('000000-'+camera+'.png')
        with Image.open(p)as image:
            task.need(image.mode=='RGB'and image.size==(256,256),'UNCHANGED_REAL_DUAL_RGB256')
        images.append(p)
    return dict(state=state,frame=frame,images=images,proof=proof)


def audited_episode(root,row,panel_sha,collector_source):
    """Read one complete new TRAIN episode; construction and Q are rejected."""
    from .stable_task_dataset_v1 import load,NATIVE
    from .storage import digest
    from .auto1_native import native_prefix_predecessor
    from .task_learning_sequence_v1 import sequence
    from .task_learning_grasp_native_v1 import complete_consumption
    from .task_learning_collection_v1 import native_schema
    root=Path(root)
    spec,sp=load(root/'EXECUTION_SPEC.json',maximum=16*1024**2)
    task.need(spec.get('task_learning_role')==task.COLLECTION_ROLE
        and spec.get('source_sha')==collector_source
        and spec.get('registered_root')==str(root)and row['native_run']==root.name
        and spec.get('phase')==task.PHASE and spec.get('kind')==task.KIND
        and spec.get('stable_task_panel_sha256')==panel_sha
        and spec.get('data_role')==row['split']=='TRAIN'and spec.get('data_parent')==row['parent']
        and spec.get('scene_id')==row['parent']and spec.get('task_command')==row['command']==task.COMMAND
        and spec.get('task_learning_teacher_mode')==row['Teacher_mode']
        and row['purpose']=='CONTINUOUS_TEACHER_DEMONSTRATION',
        'ONLY_NEW_PREREGISTERED_CONTINUOUS_TRAIN_NEVER_DIAGNOSTIC_RELABEL')
    lease,lp=load(root/'control/LEASE.json',maximum=1024**2)
    worker,wp=load(root/'WORKER_EXIT.private.json',maximum=1024**2)
    task.need(lease.get('status')=='CLOSED'and lease.get('spec_sha256')==sp['sha256']
        and lease.get('worker_exit_code')==worker.get('exit_code')==0
        and worker.get('failure')is None,'ACTUAL_CLOSED_ZERO_EXIT_CONTINUOUS_TRAJECTORY')
    native,npin=load(root/'native-01/RESULT.json');trace=native.get('trace_audit')or{};writer=trace.get('writer')or{}
    children=native.get('owned_process_exits')or[]
    task.need(native.get('status')=='NATIVE_STAGE_PASSED'and native.get('phase')==task.PHASE
        and native.get('failure')is None and native.get('actual_target_integrations')==task.E
        and native.get('native_binary_sha256')==digest(spec['native_binary'])
        and native_prefix_predecessor(spec)==NATIVE
        and native.get('observer_sha256')==digest(spec['observer_library'])
        and native.get('source_sha')==collector_source and native.get('shutdown_clean')is True
        and len(children)==4 and {v['name']for v in children}=={'sim','solver','controller','planner'}
        and all(v.get('exit_code')==0 for v in children)and not native.get('physics_stops')
        and native.get('original_joint_or_warning_samples')==native.get('native_outer_guard_rejection_count')==0
        and native.get('trace_audit_error')is None and trace.get('status')=='RECORDED_TRACE_CHECKS_PASS'
        and trace.get('native_step_records')==task.E and writer.get('dropped')==0
        and writer.get('events')==writer.get('written'),'FULL_ORIGINAL_NATIVE_TRACE_AND_SAFE_CHILD_EXITS')
    q=native['reference_qualification']
    score,score_pin=load(root/'native-01/TASK_LEARNING_SCORE.private.json',maximum=1024**2)
    task.need(q.get('schema')==native_schema(task.COLLECTION_ROLE)
        and q.get('data_role')==task.COLLECTION_ROLE and q.get('expert_training_eligible')is False
        and q.get('task_learning_grasp')==task.protocol(task.COLLECTION_ROLE)
        and complete_consumption(spec,q)and score==q.get('new_task_score')
        and score.get('contract')==task.score_contract()and score.get('task_success')is True
        and q.get('GR00T_queries')==q.get('GR00T_updates')==0,
        'COMPLETE_REAL_EXPERT_TASK_WITH_ORIGINAL_STANDING_AND_GRASP_SCORE')
    ready,ready_pin=load(root/'native-01/TASK_LEARNING_START_READINESS.private.json',maximum=1024**2)
    task.need(ready.get('passes')is True and ready.get('native_step')==task.B
        and ready.get('contract')==task.start_readiness_contract(),'ACTUAL_ORIGINAL_START_READINESS')
    captures,cp=load(root/'native-01/TASK_LEARNING_CAPTURES.private.json',maximum=16*1024**2)
    confirmed,fp=load(root/'native-01/TASK_LEARNING_CONFIRMED_COMMANDS.private.json',maximum=16*1024**2)
    task.need(captures.get('complete')is True and captures.get('task_success')is True
        and confirmed.get('task_success')is True and len(q.get('teacher_decisions',[]))==15,
        'FIFTEEN_ACTUAL_CONTINUOUS_DECISIONS_COMPLETE_BEFORE_EXPORT')
    decisions=[];decision_proofs=[]
    for index,descriptor in enumerate(q['teacher_decisions']):
        path=root/'native-01'/f'TASK_LEARNING_DECISION_{index:03d}.private.json'
        task.need(descriptor['path']==str(path),'OWN_NUMBERED_REAL_TEACHER_DECISION')
        decision,dp=load(path,descriptor['sha256'],maximum=16*1024**2)
        task.need(decision['native_step']==descriptor['native_step']==task.B+1000*index
            and descriptor['stage']==decision['stage'],'SAME_ACTUAL_CONTINUOUS_DECISION_DESCRIPTOR')
        decision['_sha256']=dp['sha256'];decisions.append(decision);decision_proofs.append(dp)
    checked=sequence(decisions,confirmed,captures['records'])
    actual_windows=q['windows'][(task.B-2503)//25:(task.B-2503)//25+600]
    task.need(len(actual_windows)==600,'EXACT_TEACHER_INTERVALS_IN_ORIGINAL_NATIVE_WINDOW_SEQUENCE')
    for command,window in zip(confirmed['records'],actual_windows):
        task.need(command['native_start_step']==window['native_start_step']
            and command['native_end_step']==window['native_end_step']
            and np.array_equal(np.asarray(command['installed_consumed_EEF10']),
                np.asarray(window['actual_installed_consumed_EEF10_BASE7'])[:10]),
            'NOMINAL_TEACHER_TO_ORIGINAL_SOLVER_GRIPPER_CONSUMPTION_JOIN')
    task.need(checked['supervised_policy_decisions']==q.get('expert_policy_decisions')
        and checked['excluded_intervention_decisions']==q.get('excluded_intervention_decisions'),
        'INTERVENTION_EXCLUSION_MATCHES_ACTUAL_NATIVE_PRODUCER')
    observations=[_capture(root,spec,sp['sha256'],identity,decision)
        for identity,decision in zip(captures['records'],decisions)]
    frame=observations[0]['frame']
    task.need(all(np.array_equal(v['frame'],frame)for v in observations),'ONE_FIXED_POLICY_FRAME_FOR_EPISODE')
    packet=packet_arrays(checked,np.asarray([v['state']for v in observations]),frame)
    from .task_learning_recovery_v1 import UNDER_CLOSE
    recovery=q.get('recovery_evidence')or{}
    genuine_recovery=bool(row['Teacher_mode']==UNDER_CLOSE
        and all(recovery.get(k)is True for k in('intervention_issued','actual_under_close_failed','native_grasp_after_correction'))
        and checked['excluded_intervention_decisions']==1 and checked['supervised_policy_decisions']==14)
    return dict(row=row,packet=packet,captures=observations,optimization_anchor_schedule=exposure_schedule(packet),
        genuine_recovery_success=genuine_recovery,native_binary_sha256=native['native_binary_sha256'],
        proof=dict(spec=sp,lease=lp,worker=wp,native=npin,score=score_pin,start_readiness=ready_pin,
            capture_index=cp,confirmed_commands=fp,real_Teacher_decisions=decision_proofs),
        expert_training_eligible='PENDING_COMPLETE_FIXED_POPULATION_AND_ACTUAL_PROCESSOR_ADMISSION')


def packet_arrays(checked,current_policy_states10,frame):
    """States already use the original shadow policy frame; map labels once."""
    states=np.asarray(current_policy_states10,float)
    task.need(states.shape==(15,10)and np.isfinite(states).all()
        and np.all((states[:,9]>=0)&(states[:,9]<=1)), 'FIFTEEN_CURRENT_ACTUAL_POLICY_STATES')
    task.need(np.array_equal(checked['native_command_anchor'],np.arange(0,600,40))
        and checked['policy_absolute10'].shape==checked['installed_audit_only'].shape==(600,10)
        and checked['valid_window'].shape==checked['stages'].shape==(15,),
        'FIFTEEN_REAL_ANCHORS_IN_ONE_EPISODE_NOT_DENSE_FAKE_QUERIES')
    mapped=map_policy_frame(checked,frame)
    return dict(actual_state10=states.copy(),committed_absolute10=mapped['policy_absolute10'],
        installed_consumed_50ms_audit_only=mapped['installed_audit_only'],
        native_command_anchor=checked['native_command_anchor'].copy(),valid_window=checked['valid_window'].copy(),
        expert_stage_audit_only=checked['stages'].copy(),instruction=np.asarray(task.LANGUAGE),
        label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'),
        command_period_s=np.asarray(.05),native_dt_s=np.asarray(.002))


def exposure_schedule(packet):
    """32 visits per episode; 8 boundary visits, 24 other-stage visits."""
    valid=np.asarray(packet['valid_window']);stages=np.asarray(packet['expert_stage_audit_only'])
    task.need(valid.shape==stages.shape==(15,)and valid.dtype==np.bool_,'EXPLICIT_REAL_DECISION_MASK')
    eligible=np.flatnonzero(valid).tolist();critical=[]
    for stage in('CLOSE','LIFT','HOLD'):
        found=[i for i in eligible if stages[i]==stage]
        if found:critical.append(found[0])
    other=[i for i in eligible if i not in critical]
    task.need(bool(critical)and bool(other),'REAL_BOUNDARY_AND_OTHER_PHASE_COVERAGE')
    return [critical[i%len(critical)]for i in range(8)]+[other[i%len(other)]for i in range(24)]


def fit_statistics(episodes):
    from scripts.pi0292.train_adapter import TAG
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10,pose10
    from scripts.pi0294.multitask.source_export import statistics
    states=[];actions=[];relative=[]
    for episode in episodes:
        row=episode['row']
        if row['split']!='TRAIN':continue
        task.need(row['purpose']=='CONTINUOUS_TEACHER_DEMONSTRATION',
            'CONSTRUCTION_QUALIFICATION_OR_OLD_SHORT_DATA_NEVER_NEW_TRAIN_STATISTICS')
        p=episode['packet']
        for i in np.flatnonzero(p['valid_window']):
            atom=int(p['native_command_anchor'][i]);s=p['actual_state10'][i:i+1];a=p['committed_absolute10'][None,atom:atom+40]
            task.need(a.shape==(1,40,10),'NO_PARTIAL_TAIL_OR_CROSS_EPISODE_H40')
            states.append(s);actions.append(a)
            relative.append(pose10(np.linalg.inv(matrix_from10(s))[:,None]@matrix_from10(a),a[...,9])[...,:9])
    task.need(bool(states),'NONEMPTY_UNIQUE_TRUE_TRAIN_DECISIONS_BEFORE_OVERSAMPLING')
    s,a,r=np.concatenate(states),np.concatenate(actions),np.concatenate(relative)
    return {TAG:dict(state=dict(eef_9d=statistics(s[:,:9],0),gripper_position=statistics(s[:,9:],0)),
        action=dict(eef_9d=statistics(a[...,:9],(0,1)),gripper_position=statistics(a[...,9:],(0,1))),
        relative_action=dict(eef_9d=statistics(r,(0,1))))}
