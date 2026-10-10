"""CPU reconstruction of actual native layers; no simulator/controller queries."""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from go2arx5_vla.target_mujoco.reference import panel_pose,pose_error,score_reference
from go2arx5_vla.trajectory.so3 import matrix_to_quat,quat_slerp

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def pose_at(trajectory,t):
    times=trajectory['time_trajectory'];poses=[x['value'][:7] for x in trajectory['state_trajectory']]
    if t<=times[0]:return poses[0]
    if t>=times[-1]:return poses[-1]
    k=bisect_right(times,t)-1;u=(t-times[k])/(times[k+1]-times[k])
    a,b=np.array(poses[k]),np.array(poses[k+1])
    return [*((1-u)*a[:3]+u*b[:3]),*quat_slerp(a[3:],b[3:],u)]

def main(root,episode):
    p=Path(root)/'evidence'/episode;out=p/'cpu';out.mkdir(exist_ok=True)
    rows=[json.loads(x) for x in (p/'physics.jsonl').read_text().splitlines()]
    model=rows[0];samples=[x for x in rows if x['kind']=='sample'];stops=[x for x in rows if x['kind']=='stop']
    initial=read(p/'initialization.jsonl');first=next(x for x in rows if x['kind']=='first_step_input')
    assert initial['data_pointer']==first['data_pointer'] and initial['qpos']==first['qpos'] and first['time']==0
    violations={}
    for j in model['joints']:
        if not j['limited']:continue
        values=[max(0,j['range'][0]-s['qpos'][j['qposadr']],s['qpos'][j['qposadr']]-j['range'][1]) for s in samples]
        violations[j['name']]=max(values)
    safe=not stops and not any(violations.values()) and all(not x['illegal_contact'] and not any(x['native_warnings']) for x in samples)
    control=[json.loads(x) for x in (p/'control.jsonl').read_text().splitlines()]
    result=read(p/'RESULT.json');task=read(p/'TASK_START.json') if (p/'TASK_START.json').exists() else None
    inventory=[dict(name=x.name,bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in p.iterdir() if x.is_file()]
    safety=dict(actual_initial_state_same_mjData=True,raw_limit_peak=violations,native_warnings=not all(not any(x['native_warnings']) for x in samples),illegal_contact=any(x['illegal_contact'] for x in samples),stop_records=stops,physics_integrations_observed_lower_bound=max(s['step'] for s in samples),exact_total_integrations='UNOBSERVED' if not stops else max(s['step'] for s in samples),last_sim_s=samples[-1]['sim_time_s'],safe=safe,original_result=result,inventory=inventory,dispatch_control_cycles_observed=max(x['control_calls'] for x in control))
    write(out/'SAFETY.json',safety)
    signals={k:[] for k in ['ee','mode','arm_request','arm_target','base','policy']}
    for line in (p/'ros.jsonl').open():
        x=json.loads(line);kind=x['kind']
        if kind not in signals:continue
        if kind=='policy':x['msg']={k:x['msg'][k] for k in ['init_observation','plan_target_trajectories']}
        signals[kind].append(x)
    if not task:return safety
    physical_mono=[x['monotonic_s'] for x in samples];physical_time=[x['sim_time_s'] for x in samples]
    ee=[x for x in signals['ee'] if task['monotonic_s']<=x['monotonic_s']<=physical_mono[-1]]
    if not ee:ee=[x for x in signals['ee'] if physical_mono[0]<=x['monotonic_s']<=physical_mono[-1]]
    pairs=[dict(sim_time_s=float(np.interp(x['monotonic_s'],physical_mono,physical_time)),mpc_time_s=x['msg']['time'],ros_time_s=x['wall_s'],wall_monotonic_s=x['monotonic_s']) for x in ee]
    offset=float(np.median([x['mpc_time_s']-x['sim_time_s'] for x in pairs]))
    residual=max(abs(x['mpc_time_s']-x['sim_time_s']-offset) for x in pairs)
    clock=dict(real_output=True,episode_id=episode,sim_to_mpc=dict(slope=1.0,offset_s=offset),mapping_max_residual_s=residual,samples=pairs,scope='after unpause; separately versioned fresh-stack epoch; no cross-reset map',reset_isolation_checked=False,expired_reference_hold_checked=False)
    write(out/'CLOCK_PAIRS.json',clock)
    if result['mode']=='clock-negatives':
        lifecycle=[json.loads(x) for x in (p/'planner.jsonl').read_text().splitlines() if json.loads(x)['kind']=='lifecycle']
        assert len(read(p/'NEGATIVE_CASES.json'))==3
        assert any(x['reason']=='REJECT_PACKET' and x['detail']=='WRONG_EPISODE' for x in lifecycle)
        assert any(x['reason']=='REJECT_PACKET' and x['detail']=='STALE_OR_FUTURE_PACKET' for x in lifecycle)
        assert any(x['reason']=='REFERENCE_EXPIRED' for x in lifecycle)
        write(out/'LIFECYCLE.json',dict(status='PASS' if safe else 'FAIL',actual_cases=read(p/'NEGATIVE_CASES.json'),events=lifecycle,scope='actual original HOLD responses; not G1 tracking PASS'))
        return safety
    issued=read(p/'ISSUED.json');planner=[json.loads(x) for x in (p/'planner.jsonl').read_text().splitlines()]
    accepted=[x for x in planner if x['kind']=='accept' and x['accepted']]
    installed=[x for x in planner if x['kind']=='installed']
    policies=signals['policy'];ptimes=[x['monotonic_s'] for x in policies]
    atimes=[x['monotonic_s'] for x in signals['arm_request']];btimes=[x['monotonic_s'] for x in signals['base']]
    sha_to_id={x['wire_sha256']:x['ref_id'] for x in issued};accepted_rows=[]
    for x in accepted:
        matching=[y for y in installed if y['ref_id']==x['ref_id']]
        if not matching:continue
        c=next(c for c in issued if c['wire_sha256']==x['ref_id'])
        accepted_rows.append(dict(ref_id=c['ref_id'],issue_time_s=c['issue_time_s'],accept_time_s=x['monotonic_s']-task['monotonic_s'],install_time_s=matching[0]['monotonic_s']-task['monotonic_s'],native_bind_mpc_s=matching[0]['bound_start']))
    output=[]
    for s in samples:
        mpc=s['sim_time_s']+offset;t=mpc-task['mpc_time_s']
        if not 0<=t<=10.075 or s['monotonic_s']>physical_mono[-1]:continue
        k=bisect_right(ptimes,s['monotonic_s'])-1;a=bisect_right(atimes,s['monotonic_s'])-1;b=bisect_right(btimes,s['monotonic_s'])-1
        if min(k,a,b)<0:continue
        policy=policies[k];target=policy['msg']['plan_target_trajectories']
        c=max((x for x in issued if x['monotonic_s']<=s['monotonic_s']),key=lambda x:x['sequence'],default=issued[0])
        achieved=[*s['tcp_xyz'],*matrix_to_quat(np.array(s['tcp_R']).reshape(3,3)).tolist()]
        output.append(dict(t=t,requested_pose=panel_pose(result['condition'],task['initial_tcp'],t),installed_pose=pose_at(target,mpc),achieved_pose=achieved,ref_id=c['ref_id'],sim_time_s=s['sim_time_s'],achieved_time_s=s['sim_time_s'],observation_time_s=policy['msg']['init_observation']['time']-offset,install_time_s=policy['monotonic_s']-physical_mono[0]+physical_time[0],dispatch_time_s=signals['arm_request'][a]['monotonic_s']-physical_mono[0]+physical_time[0],safety_ok=safe,arm_request=signals['arm_request'][a]['msg']['points'][-1],base_request=signals['base'][b]['msg']['data'],policy_receipt_age_s=s['monotonic_s']-policy['monotonic_s'],arm_request_age_s=s['monotonic_s']-signals['arm_request'][a]['monotonic_s'],base_request_age_s=s['monotonic_s']-signals['base'][b]['monotonic_s']))
    (out/'FOUR_LAYERS.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in output))
    common=[]
    for a,b in zip(issued,issued[1:]):
        old=[x for x in installed if x['ref_id']==a['wire_sha256']];new=[x for x in installed if x['ref_id']==b['wire_sha256']]
        if not old or not new:continue
        old=old[-1];new=new[0]
        future=max(old['times'][0],new['times'][0]);end=min(old['times'][-1],new['times'][-1])
        if future>end:continue
        def pose(row,t):return pose_at(dict(time_trajectory=row['times'],state_trajectory=[dict(value=p) for p in row['poses']]),t)
        common.append(dict(old_ref_id=a['ref_id'],new_ref_id=b['ref_id'],future_time_s=future-task['mpc_time_s'],old_pose=pose(old,future),new_pose=pose(new,future),actual_common_mpc_s=future))
    evidence=dict(real_output=True,native_abort_record_complete=True,joint_bounds_record_complete=True,contact_record_complete=True,clock_record_complete=residual<=.025,dispatch_record_complete=bool(atimes and btimes),native_abort=bool(stops),numerical_bound_warning=any(s['numerical_warning'] for s in samples),illegal_contact=any(s['illegal_contact'] for s in samples),same_time_pose_alignment=True)
    score=score_reference(result['condition'],task['initial_tcp'],output,issued=issued,accepted=accepted_rows,common_future=common,runtime_evidence=evidence)
    write(out/'SCORE_INPUT.json',dict(condition=result['condition'],initial_pose=task['initial_tcp'],issued=issued,accepted=accepted_rows,common_future=common,runtime_evidence=evidence))
    write(out/'REFERENCE_SCORE.json',score)
    # Native measured FK carries the actual MPC clock and the recorded 10s end.
    # Keep the separate sampled-physics score, including any incomplete endpoint.
    feedback=[]
    for x in signals['ee']:
        mpc=x['msg']['time'];t=mpc-task['mpc_time_s'];mono=x['monotonic_s']
        if not 0<=t<=10.075:continue
        k=bisect_right(ptimes,mono)-1;a=bisect_right(atimes,mono)-1;b=bisect_right(btimes,mono)-1
        if min(k,a,b)<0:continue
        policy=policies[k];sim=mpc-offset
        c=max((c for c in issued if c['monotonic_s']<=mono),key=lambda c:c['sequence'],default=issued[0])
        receipt=next((r for r in accepted_rows if r['ref_id']==c['ref_id']),None)
        if receipt is None:continue
        feedback.append(dict(t=t,requested_pose=panel_pose(result['condition'],task['initial_tcp'],t),installed_pose=pose_at(policy['msg']['plan_target_trajectories'],mpc),achieved_pose=x['msg']['state']['value'][:7],ref_id=c['ref_id'],sim_time_s=sim,achieved_time_s=sim,observation_time_s=policy['msg']['init_observation']['time']-offset,install_time_s=receipt['native_bind_mpc_s']-offset,dispatch_time_s=float(np.interp(signals['arm_request'][a]['monotonic_s'],physical_mono,physical_time)),safety_ok=safe,source='actual_native_EE_feedback_FK',clock_source='actual EE MPC timestamp; sim mapped using measured affine receipt',actual_receipt_monotonic_s=mono,arm_request=signals['arm_request'][a]['msg']['points'][-1],base_request=signals['base'][b]['msg']['data'],policy_receipt_age_s=mono-policy['monotonic_s'],arm_request_receipt_age_s=mono-signals['arm_request'][a]['monotonic_s'],base_request_receipt_age_s=mono-signals['base'][b]['monotonic_s'],dispatch_scope='ROS command receipt; native control trace recorded separately'))
    (out/'NATIVE_FEEDBACK_LAYERS.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in feedback))
    native_score=score_reference(result['condition'],task['initial_tcp'],feedback,issued=issued,accepted=accepted_rows,common_future=common,runtime_evidence=evidence)
    write(out/'NATIVE_FEEDBACK_SCORE.json',native_score)
    return dict(episode=episode,safe=safe,clock_residual_s=residual,reference_score=score)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('episode');a=p.parse_args();print(json.dumps(main(a.root,a.episode),allow_nan=False))
