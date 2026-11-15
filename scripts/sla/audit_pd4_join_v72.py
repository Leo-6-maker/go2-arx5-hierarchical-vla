"""Read-only JOIN10 evidence audit, including renewals and partial prefixes.

Uses saved responses, native journals and receipts. It never loads a model,
calls a certifier, integrates physics, changes a score or dispatches a worker.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np

from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model import fusion_trajectory_adapter as envelope
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix, rotation_matrix_to_rot6d_rows
from scripts.pi0291_arx5_data.native_dataset import pose10
from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import unseal
from scripts.sla.reference_join10_v71 import connect, METHOD
from scripts.sla.audit_query_feasibility_v7 import geometry_detail

SPEC_SHA256='803b92a4825b8b462acd980734da0a9e4cf3fd2bc9769f2ceb0670e98be637ca'
RUNTIME_SHA='3232af4a42f0c1436980c5cd17a24a62ba2cb9ba'
CERTIFIER_SHA='edfb96620f91b6e2e583168bb1e157e8b8a9b5dab884917985af3d68d8a82e93'
CONSUMER_VERSION='SLA_PD4_REFERENCE_JOIN10_H40_P10_V72'


def need(ok,reason):
    if not ok:raise ValueError('JOIN10_EVIDENCE_'+reason)


def read(path):
    return json.loads(Path(path).read_bytes())


def close(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and np.allclose(a,b,rtol=0,atol=1e-12)


def preview(candidate,elapsed):
    return candidate[np.minimum(np.arange(elapsed,elapsed+40),39)]


def paths(raw,frame,actual,history):
    """Original single frame/grip decode, then the fixed archived method."""
    target=pose10(np.linalg.inv(frame)@matrix10(raw),raw[:,9])
    desired=target.copy()
    desired[:,9]=(envelope.WORKING_CLOSED_M+
        (envelope.WORKING_OPEN_M-envelope.WORKING_CLOSED_M)*target[:,9])/envelope.MECHANICAL_TRAVEL_M
    reference=history[-1]
    baseline=desired.copy()
    baseline[:,:3]=reference[:3,3]+target[:,:3]-actual[:3,3]
    baseline[:,3:9]=rotation_matrix_to_rot6d_rows(
        reference[:3,:3]@actual[:3,:3].T@rot6d_rows_to_matrix(target[:,3:9]))
    candidate=connect(baseline,history)
    candidate_desired=candidate.copy()
    candidate_desired[:,:3]+=actual[:3,3]-reference[:3,3]
    candidate_desired[:,3:9]=rotation_matrix_to_rot6d_rows(
        actual[:3,:3]@reference[:3,:3].T@rot6d_rows_to_matrix(candidate[:,3:9]))
    return target,desired,baseline,candidate,candidate_desired


def verify_adaptation(value,baseline,candidate,history,step):
    need(value['schema']==CONSUMER_VERSION+'_TRANSPORT'and value['method']==METHOD,
         'INDEPENDENT_FIXED_CONSUMER_IDENTITY')
    need(value['history_native_steps']==[step-25,step]and close(value['previous_reference_poses'],history),
         'TWO_ACTUALLY_COMPLETED_REFERENCE_POSES')
    need(close(value['original_reference_H40'],baseline)and close(value['candidate_reference_H40'],candidate),
         'REPRODUCIBLE_ORIGINAL_AND_CANDIDATE_FULL40')
    saved=np.asarray(value['candidate_reference_H40']);original=np.asarray(value['original_reference_H40'])
    need(np.array_equal(saved[9:],original[9:])and np.array_equal(saved[:,9],original[:,9]),
         'SUFFIX_AND_GRIP_BYTE_EXACT')
    need(value['installed_reference_changed']and value['original_raw_H40_unchanged']
         and value['suffix_k9_through39_byte_exact']and value['grip_byte_exact']
         and not value['expert_training_eligible']and not value['native_execution_authorized'],
         'ADAPTER_NOT_RAW_OR_EXPERT_OR_EXECUTION_GRANT')


def verify_install(event,candidate,step):
    elapsed=(event['native_step']-step)//25
    need(event['native_step']==step+25*elapsed and elapsed in(0,7),'ORIGINAL_P10_INSTALL_CLOCK')
    need(close(event['action_EEF10'],preview(candidate,elapsed)),'FULL40_RENEWAL_IS_SAME_CANDIDATE')
    if event['phase']=='reference-certificate-rejected':
        need(event['submitted_or_executed']is False and bool(event['reason']),'REJECTED_PREVIEW_NOT_INSTALLED')
        return dict(elapsed=elapsed,status='REJECTED',reason=event['reason'])
    c=event['native_reference_certificate']
    need(c['status']=='PASS_CLONED_NATIVE_REFERENCE_CERTIFICATE_NOT_DYNAMIC_WBC_PROOF'
         and c['certifier_revision']=='FIVE_UPDATE_FINAL_FK_CHECK_V1'
         and c['maximum_IK_updates_per_point']==5 and c['executed_reference_history_points']==2
         and c['points']==1051 and c['position_tolerance_m']==1e-5
         and c['rotation_tolerance_rad']==1e-4 and c['physics_integrations']==0
         and not c['actual_state_modified']and not c['Cartesian_reference_values_changed']
         and c['max_joint_speed_rad_s']<=.7*(1+1e-10)
         and c['max_joint_acceleration_rad_s2']<=1.*(1+1e-10),'ORIGINAL_COMPLETE_H40_CERTIFICATE')
    return dict(elapsed=elapsed,status='PASS_ORIGINAL_FULL40',wire_sha256=event['wire_sha256'],certificate=c)


def verify_consumption(windows,candidate,installs,step):
    """Count confirmed cycle windows even when an outer aggregate was not written."""
    need(len(windows)<=10,'AT_MOST_P10_CONFIRMED_POINTS')
    accepted={event['wire_sha256']:event for event in installs if event['phase']=='proposed'}
    for k,w in enumerate(windows):
        need((w['native_start_step'],w['native_end_step'],w['actual_ticks'])==
             (step+25*k,step+25*(k+1),25),'REAL_25_TICK_CONSUMPTION_CLOCK')
        need(close(w['actual_installed_consumed_EEF10_BASE7'][:10],candidate[k]),'CONSUMED_CANDIDATE_POINT')
        event=accepted.get(w['reference_sha256'])
        need(event is not None,'CONSUMPTION_LINKS_TO_ACCEPTED_CERTIFIED_REFERENCE')
        elapsed=(event['native_step']-step)//25
        need(elapsed==(0 if k<7 else 7)and close(w['full_final_installed_H40'],preview(candidate,elapsed)),
             'CONSUMED_REFERENCE_RETAINS_FULL_SUFFIX')
        need(bool(w['solver_policies_with_exact_installed_target'])and w['gripper']['command_consumed']is True,
             'ORIGINAL_SOLVER_AND_GRIPPER_CONFIRMATION')
    return len(windows)


def audit_condition(group,assignment,outcome):
    root=group/'runs'/assignment['native_run'];native=root/'native-01'
    gpu=root.with_name(root.name+'-gpu')/'fit-01'
    need(outcome['native_run']==root.name,'FIXED_CONDITION_ORDER')
    need(digest(native/'RESULT.json')==outcome['result_sha256']and
         digest(gpu/'RESULT.private.json')==outcome['GPU_result_sha256'],'PINNED_NATIVE_AND_GPU_TERMINALS')
    result=read(native/'RESULT.json');generated=read(gpu/'RESULT.private.json')
    need(result['native_binary_sha256']=='39e5136603f51c092b2f3e595245468dbb0c12dbca79818b0df08da8dd4af7dd',
         'ORIGINAL_NATIVE_BINARY_BYTES')
    records=generated['records'];captures=read(native/'TASK_LEARNING_CAPTURES.private.json')['records']
    need(len(records)==len(captures)==outcome['actual_SLA_queries'],'ACTUAL_QUERIES_NOT_RESERVED_SLOTS')
    events=[json.loads(line)for line in(native/'COMMANDS.private.jsonl').read_text().splitlines()]
    preparation=read(native/'STABLE_TASK_CYCLES/-1/RESULT.private.json')
    past=list(preparation['windows']);rows=[];confirmed=0;student_integrations=0
    aggregate_file=native/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json'
    aggregate=read(aggregate_file)['records']if aggregate_file.exists()else[]
    for i,(query,capture)in enumerate(zip(records,captures,strict=True)):
        step=5703+250*i;need(capture['native_step']==step and query['seed']==assignment['query_seeds'][i],'FIXED_QUERY_CLOCK_AND_NOISE')
        need(query['noise']['key_seed_u32']==assignment['query_seeds'][i]%2**32
             and query['noise']['shape']==[1,40,32],'SAME_ORIGINAL_PRNG_KEY_AND_ACTION_SHAPE')
        action=gpu/'queries'/f'{i:02d}'/'ACTION.npz'
        need(digest(action)==query['action_sha256']and digest(capture['path'])==query['observation_sha256'],
             'REAL_RESPONSE_AND_CURRENT_OBSERVATION_BYTES')
        need(digest(Path(capture['path']).parent/'STATE.private.bin')==capture['state_sha256'],'NATIVE_STATE_BYTES')
        observation=unseal(read(capture['path']));shadow=unseal(observation['original_shadow_ABI'])
        actual=matrix10(shadow['target_observation']['actual_state_eef10'])
        frame=np.asarray(shadow['source_policy_view']['T_source_from_target'])
        previous=[w for w in past if w['native_end_step']<=step][-2:]
        need([w['native_end_step']for w in previous]==[step-25,step],'CONTIGUOUS_COMPLETED_REFERENCE_HISTORY')
        history=np.stack([matrix10(w['actual_installed_consumed_EEF10_BASE7'][:10])for w in previous])
        with np.load(action,allow_pickle=False)as packet:
            raw=packet['physical_H40'].copy();normalized=packet['normalized']
            need(raw.shape==(40,10)and normalized.shape==(1,40,32)and np.isfinite(raw).all()
                 and np.isfinite(normalized).all(),'ORIGINAL_FINITE_NATIVE32_RAW40')
        target,desired,baseline,candidate,candidate_desired=paths(raw,frame,actual,history)
        raw_geometry=[geometry_detail(desired,actual),geometry_detail(baseline,history[-1])]
        candidate_geometry=[geometry_detail(candidate_desired,actual),geometry_detail(candidate,history[-1])]
        decision_path=native/f'TASK_LEARNING_DECISION_{i:03d}.private.json'
        cycle_file=native/'STABLE_TASK_CYCLES'/f'{i:02d}'/'RESULT.private.json'
        cycle=read(cycle_file)if cycle_file.exists()else None
        windows=cycle['windows']if cycle else[]
        installs=[event for event in events if step<=event.get('native_step',-1)<step+250
                  and event.get('phase')in('proposed','reference-certificate-rejected')and'action_EEF10'in event]
        if decision_path.exists():
            decision=read(decision_path)
            need(decision['action_sha256']==query['action_sha256']and close(decision['raw_policy_H40'],raw)
                 and close(decision['mapped_target_H40'],target),'RAW_AND_MAPPED_OUTPUT_RETAINED')
            need(not decision['teacher_fallback']and decision['command_clips']==0
                 and not decision['endpoint_basis_applied']and not decision['expert_segment_generator_applied'],
                 'NO_TEACHER_BASIS_OR_CLIPS')
            verify_adaptation(decision['reference_connection_evidence'],baseline,candidate,history,step)
            verify_adaptation(decision['adaptation'],baseline,candidate,history,step)
            need(all(g['passes']for g in raw_geometry+candidate_geometry),'BOTH_ORIGINAL_FULL40_EEF_GATES')
            need(cycle is not None and close(cycle['raw_H40'],target)
                 and close(cycle['full_reference_path'],candidate),'NATIVE_CYCLE_OWNS_SAME_CANDIDATE')
            certificates=[verify_install(event,candidate,step)for event in installs]
            count=verify_consumption(windows,candidate,installs,step)
            need(bool(installs),'EVERY_DECIDED_QUERY_HAS_NATIVE_INSTALL_ATTEMPT')
            if len(windows)>7:need([v['elapsed']for v in certificates]==[0,7],'P10_RENEWAL_CHECKED')
            linked=[w for w in aggregate if w.get('student_decision_sha256')==digest(decision_path)]
            # The outer aggregate is appended only after a whole prefix returns.
            # A failed renewal may leave seven authentic windows in cycle only.
            need(len(linked)in(0,count),'NO_FOREIGN_AGGREGATE_CONSUMPTION')
            for k,w in enumerate(linked):
                need(close(w['policy_EEF10'],target[k])and close(w['installed_consumed_EEF10'],candidate[k]),
                     'AGGREGATE_RAW_AND_CANDIDATE_MATCH_CONFIRMED_WINDOW')
            status='P10_CONSUMED'if count==10 else'PARTIAL_P10_CONSUMED'if count else'REJECTED_BEFORE_CONSUMPTION'
        else:
            rejection=read(native/'TASK_LEARNING_STUDENT_REJECTION.private.json')
            need(rejection['index']==i and rejection['action_sha256']==query['action_sha256']
                 and not rejection['proposal_executed']and not windows and not installs,'EEF_REJECTION_WITHOUT_EFFECT')
            if 'JOIN10_FULL_H40_GATE'in rejection['failure']:
                need(all(g['passes']for g in raw_geometry)and not all(g['passes']for g in candidate_geometry),
                     'CANDIDATE_EEF_FAILURE_RETAINED')
                verify_adaptation(rejection['reference_connection_evidence'],baseline,candidate,history,step)
                status='CANDIDATE_EEF_REJECT'
            else:
                need('SLA_DENSE_ORIGINAL_FULL_H40_GATE'in rejection['failure']and
                     not all(g['passes']for g in raw_geometry),'ORIGINAL_RAW_EEF_FAILURE_RETAINED')
                status='ORIGINAL_RAW_EEF_REJECT'
            count=0;certificates=[]
        ticks=cycle.get('actual_native_integrations',25*count)if cycle else 0
        need(type(ticks)is int and ticks>=25*count,'INTEGRATIONS_INCLUDE_UNCONFIRMED_FAILURE_COST')
        confirmed+=count;student_integrations+=ticks
        rows.append(dict(query=i,native_step=step,status=status,confirmed_points=count,
            native_integrations=ticks,unconfirmed_ticks=ticks-25*count,
            certificates=certificates,raw_geometry=raw_geometry,candidate_geometry=candidate_geometry,
            maximum_translation_change_m=float(np.linalg.norm(candidate[:,:3]-baseline[:,:3],axis=1).max()),
            action_sha256=query['action_sha256']))
        if cycle:past.extend(cycle['windows'])
    tail=native/'STABLE_TASK_CYCLES/60/RESULT.private.json'
    tail_ticks=read(tail)['actual_native_integrations']if tail.exists()else 0
    integrations=result['actual_target_integrations']
    return dict(native_run=root.name,method=assignment['method'],layout=assignment['layout'],
        technical_valid=outcome['technical_valid'],task_success=outcome['task_success'],
        first_failure=outcome['first_failure'],queries=rows,model_generations=len(records),
        SLA_parent_loads=generated['actual_SLA_parent_loads'],confirmed_student_points=confirmed,
        student_integrations=student_integrations,termination_hold_integrations=tail_ticks,
        native_integrations=integrations,initialization_integrations=5703,
        unaccounted_integrations=integrations-5703-student_integrations-tail_ticks,
        raw_native_result_sha256=outcome['result_sha256'],GPU_result_sha256=outcome['GPU_result_sha256'])


def audit(group,completed_only=False):
    group=Path(group);need(digest(group/'EXECUTION_SPEC.json')==SPEC_SHA256,'FIXED_NEW_SYSTEM_SPEC')
    spec=read(group/'EXECUTION_SPEC.json');need(spec['source_sha']==RUNTIME_SHA,'SEALED_RUNTIME_IDENTITY')
    need(digest(Path(spec['source_root'])/'scripts/pi0294/task_path_model.py')==CERTIFIER_SHA,
         'ORIGINAL_CERTIFIER_SOURCE_UNCHANGED')
    unit=dict(line.split('=',1)for line in subprocess.check_output(['systemctl','--user','show',
        'wm02-'+group.name,'--property=MainPID,SubState,Result'],text=True,timeout=15).splitlines())
    terminal=unit['MainPID']=='0'
    need(terminal or completed_only,'GROUP_STILL_LIVE_USE_COMPLETED_ONLY')
    assignments=read(group/'PLAN.private.json')['assignments']
    outcomes=read(group/'OUTCOMES.private.json')if(group/'OUTCOMES.private.json').exists()else[]
    need(len(assignments)==8 and len(outcomes)<=8,'FIXED_EIGHT_CONDITION_DENOMINATOR')
    if terminal:
        lease=read(group/'control/LEASE.json');worker=read(group/'WORKER_EXIT.private.json')
        need(lease['status']=='CLOSED'and lease['worker_exit_code']==worker['exit_code'],
             'PARENT_ACTUALLY_CLOSED_NOT_JUST_MISSING_HANDLE')
        result=read(group/'RESULT.private.json');need(result['outcomes']==outcomes,'FINAL_OUTCOME_LIST')
    rows=[];errors=[]
    for assignment,outcome in zip(assignments,outcomes):
        try:
            for suffix in('','-gpu'):
                actor=group/'runs'/(assignment['native_run']+suffix)
                lease=read(actor/'control/LEASE.json');worker=read(actor/'WORKER_EXIT.private.json')
                need(lease['status']=='CLOSED'and lease['worker_exit_code']==worker['exit_code'],
                     'BOTH_ACTORS_ACTUALLY_TERMINAL')
            rows.append(audit_condition(group,assignment,outcome))
        except Exception as error:
            errors.append(dict(native_run=assignment['native_run'],error=type(error).__name__+': '+str(error)))
    return dict(schema='SLA_PD4_JOIN10_NATIVE_CHAIN_READONLY_AUDIT_V72',epoch=time.time(),
        source_sha=RUNTIME_SHA,spec_sha256=SPEC_SHA256,group_terminal=terminal,
        registered_conditions=8,terminal_outcomes=len(outcomes),audited_conditions=len(rows),
        passed=bool(rows)and not errors,audit_complete=terminal and len(rows)==8 and not errors,
        scope='ONLY_RECORDED_TERMINAL_OUTCOMES; NO_LIVE_CONDITION_COMPLETION_INFERRED',
        errors=errors,conditions=rows,
        original_outcomes_not_rescored=outcomes,new_optimizer_updates=0,new_training_visits=0,
        completed_outcome_physical_requests=sum(o['native_world_attempted']for o in outcomes),
        completed_outcome_native_integrations=sum(o['actual_native_integrations']for o in outcomes),
        completed_outcome_generations=sum(o['actual_SLA_queries']for o in outcomes),
        verified_confirmed_student_points=sum(row['confirmed_student_points']for row in rows),
        new_model_generations=0,new_certifier_calls=0,new_physical_requests=0,new_integrations=0,
        old_RP8_passed=False,full_WBC_state_restored=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--completed-only',action='store_true');args=parser.parse_args()
    result=audit(args.group,args.completed_only)
    with Path(args.output).open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items()if k not in('conditions','original_outcomes_not_rescored')}))
    return 0 if result['passed']else 1


if __name__=='__main__':raise SystemExit(main())
