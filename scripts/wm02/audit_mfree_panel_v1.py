"""Read a closed mobile panel: real denominators and fullH40 rejection metrics.

No model load, sampling, training, simulator construction, certification change
or integration. Prefix/suffix summaries diagnose the unchanged full40 gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np
from scripts.pi0294.task_state import parse_state
from scripts.pi0292.eval_contract import matrix10
from go2arx5_vla.trajectory.so3 import matrix_to_quat
from go2arx5_vla.world_model import fusion_trajectory_adapter as envelope

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def per_knot(physical):
    dt=.05;pose=matrix10(physical);velocity=np.diff(physical[:,:3],axis=0)/dt
    acceleration=np.diff(np.vstack((np.zeros(3),velocity,np.zeros(3))),axis=0)/dt
    relative=pose[1:,:3,:3]@np.swapaxes(pose[:-1,:3,:3],-1,-2)
    q=matrix_to_quat(relative);q=np.where(q[:,3:]<0,-q,q);sine=np.linalg.norm(q[:,:3],axis=1,keepdims=True)
    angular=q[:,:3]*(2*np.arctan2(sine,q[:,3:]))/np.where(sine>0,sine,1.)/dt
    angular_acceleration=np.diff(np.vstack((np.zeros(3),angular,np.zeros(3))),axis=0)/dt
    values=dict(translation_speed=(np.linalg.norm(velocity,axis=1),envelope.MAX_TRANSLATION_SPEED_M_S),
        angular_speed=(np.linalg.norm(angular,axis=1),envelope.MAX_ANGULAR_SPEED_RAD_S),
        finger_speed=(abs(np.diff(physical[:,9]))*(envelope.WORKING_OPEN_M-envelope.WORKING_CLOSED_M)/dt,envelope.MAX_FINGER_SPEED_M_S),
        translation_acceleration=(np.linalg.norm(acceleration,axis=1),envelope.MAX_TRANSLATION_ACCELERATION_M_S2),
        angular_acceleration=(np.linalg.norm(angular_acceleration,axis=1),envelope.MAX_ANGULAR_ACCELERATION_RAD_S2))
    result={}
    for name,(value,limit)in values.items():
        bad=np.flatnonzero(value>limit*(1+1e-10));peak=int(np.argmax(value))
        result[name]=dict(maximum=float(value[peak]),limit=limit,ratio=float(value[peak]/limit),
            peak_k=peak,first_violation_k=int(bad[0])if len(bad)else None,
            prefix_maximum=float(value[:10].max()),suffix_maximum=float(value[10:].max()),
            endpoint_rest_padding_in_acceleration=name.endswith('acceleration'))
    return result


def journal_count(path):
    count=0;previous=0
    with Path(path).open('rb')as handle:
        while size:=handle.read(4):
            if len(size)!=4:raise ValueError('TRUNCATED_CLOSED_NATIVE_LENGTH')
            length=struct.unpack('<I',size)[0];raw=handle.read(length)
            if len(raw)!=length:raise ValueError('TRUNCATED_CLOSED_NATIVE_PACKET')
            packet=parse_state(raw);count+=1
            if packet['step']!=previous+1 or abs(packet['time_s']-packet['step']*.002)>1e-8:
                raise ValueError('REAL_CONTIGUOUS_2MS_NATIVE_CLOCK')
            previous=packet['step']
    return count


def run(root):
    root=Path(root);lease=read(root/'control/LEASE.json');spec=read(root/'EXECUTION_SPEC.json');result=read(root/'RESULT.private.json')
    if lease['status']!='CLOSED':raise ValueError('CLOSED_PANEL_ONLY')
    roots={Path(p).parent.name:Path(p).parent for p in spec['cases']};rows=[]
    for outcome in result['outcomes']:
        case=roots[outcome['root']];path=case/'RESULT.private.json';r=read(path);out=case/'native-01'
        if sha(path)!=outcome['result_sha256']:raise ValueError('UNCHANGED_NATIVE_CASE_RESULT')
        count=journal_count(out/'state.bin.journal')
        if count!=r['actual_native_integrations']:raise ValueError('TRUE_NATIVE_INTEGRATION_DENOMINATOR')
        samples=[json.loads(line)for line in(out/'MOBILE_SAMPLES.private.jsonl').read_text().splitlines()]if(out/'MOBILE_SAMPLES.private.jsonl').exists()else[]
        task=[v for v in samples if v['native_start_step']>=5703]
        initial=next((v for v in samples if v['native_end_step']==5703),None)
        start=None;longest=0.;maximum=0.
        for v in task:
            lift=v['object_pose'][2][3]-initial['object_pose'][2][3];maximum=max(maximum,lift)
            if v['native_grasp']and lift>=.020:
                if start is None:start=v['native_end_step']
                longest=max(longest,(v['native_end_step']-start)*.002)
            else:start=None
        detail=r.get('detail')or{};rejection=detail.get('valid_policy_rejection');diagnostic=None
        if rejection:
            query=int(rejection['query']);query_root=out/'queries'/f'{query:02d}';response=read(query_root/'RESPONSE.private.json')
            if sha(response['action'])!=response['action_sha256']:raise ValueError('UNCHANGED_REAL_MODEL_OUTPUT')
            with np.load(response['action'],allow_pickle=False)as z:physical=z['physical_H40'].copy()
            diagnostic=dict(query=query,action_sha256=response['action_sha256'],original_reason=rejection['reason'],metrics=per_knot(physical),
                coordinates='ORIGINAL_DECODED_POLICY_FRAME_RIGID_INVARIANT_RATE_NORMS',
                prefix_is_not_a_replacement_gate=True,full40_check_preserved=True)
        query_files=list((out/'queries').glob('*/RESPONSE.private.json'))if(out/'queries').exists()else[]
        if len(query_files)!=r['model_generations']:raise ValueError('REAL_GENERATION_RESPONSE_COUNT')
        rows.append(dict(condition=r['condition'],result_sha256=sha(path),source_sha=r['source_sha'],physical_requests=1,
            native_integrations=count,task_integrations=25*len(task),partial_task_ticks=count-5703-25*len(task)if initial else None,
            actual_model_generations=len(query_files),confirmed_points=len(task),
            ever_grasp=any(v['native_grasp']for v in task),grasp_points=sum(v['native_grasp']for v in task),
            maximum_lift_m=maximum,longest_grasp_lift_hold_s=longest,
            task_success=bool(detail.get('task_success')),task_and_required_motion_success=bool(r['metrics'].get('task_and_required_motion_success')),
            unsafe=r['unsafe'],technical_failure=r['technical_failure'],failure=r['failure'],
            terminal_reason=detail.get('terminal_reason'),at_30_seconds=detail.get('at_30_seconds'),
            actual_base_translation_m=r['metrics'].get('actual_base_translation_m'),
            EEF_translation_RMS_m=r['metrics'].get('EEF_translation_RMS_m'),
            exact_solver_target_confirmed_points=r['metrics'].get('exact_solver_target_confirmed_atoms',0),
            first_rejection=diagnostic,trace_status=(r.get('trace_audit')or{}).get('status')))
    return dict(schema='MFREE_CLOSED_NATIVE_PANEL_READONLY_AUDIT_V1',epoch=time.time(),
        original_group_result_sha256=sha(root/'RESULT.private.json'),planned=len(spec['cases']),terminal=len(rows),
        physical_requests=sum(v['physical_requests']for v in rows),native_integrations=sum(v['native_integrations']for v in rows),
        model_generations=sum(v['actual_model_generations']for v in rows),confirmed_points=sum(v['confirmed_points']for v in rows),
        task_successes=sum(v['task_success']for v in rows),grasp_conditions=sum(v['ever_grasp']for v in rows),
        unsafe=sum(v['unsafe']for v in rows),technical_failure=sum(v['technical_failure']for v in rows),
        unstarted=len(spec['cases'])-len(rows),outcomes=rows,
        new_model_loads=0,new_model_generations=0,new_optimizer_updates=0,new_training_visits=0,new_physical_requests=0,new_integrations=0,
        derivative_clock_s=.05,acceleration_k0='REST_TO_FIRST_SEGMENT',acceleration_k39='LAST_SEGMENT_TO_REST_ORIGINAL_CONTRACT',
        controller_selected_policy_identity='NOT_EXPORTED_DERIVED_TARGET_WITNESSES_ONLY',
        WBC_changes=False,safety_changes=False,old_RP8_passed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args()
    print(json.dumps(run(args.root),allow_nan=False))
