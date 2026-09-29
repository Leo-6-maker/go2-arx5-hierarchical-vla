"""Fixed-order complete Motion2 eligibility; shared PI0259 solver, no model calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from pi0260_design import ROOT
from pi0260_route import route
from pi0255_motion3 import sha,write,scene
from pi0259_kinematic import reference,source_geometry,full_fk,homogeneous,solve,continuation
from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.safety import TrajectorySafetyGateV1


def admission():
    frozen=json.loads((ROOT/'state/design_freeze.json').read_text())
    h0=json.loads((ROOT/'state/H0_ACCEPTED.json').read_text())
    assert route(frozen['registry_sha256'],sha(ROOT/'design/registry.json'),h0['status'],h0['status'])=='ADMIT_NEXT_STAGE'
    assert sha(ROOT/'state/H0_ACCEPTED.json')==frozen['H0_receipt_sha256']
    reg=json.loads((ROOT/'design/registry.json').read_text());reg['renderer']['selected_rig']=frozen['selected_rig']
    return reg


def geometry(reg,seq):
    folder=ROOT/'proposals'/seq['family_id']/seq['sequence_id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'geometry_result.json').exists():return json.loads((folder/'geometry_result.json').read_text())
    assert not (folder/'started.json').exists(),'unfinished geometry needs exact attempt reconciliation'
    assert len(list((ROOT/'proposals').glob('*/*/started.json')))<664
    write(folder/'started.json',dict(sequence=seq,source_sha256=sha(__file__),registry_sha256=sha(ROOT/'design/registry.json')))
    family=next(f for f in reg['families'] if f['family_id']==seq['family_id'])
    kin,collision,base,root,velocity=source_geometry(reg);q=np.array(reg['arm_default']);data=scene(reg,family)
    times=np.arange(round(reg['experts'][seq['task']]['nominal_end_s']/.05)+1)*.05
    dp,dr=reference(reg,family,seq['task'],seq['command'],times)
    def target(t):
        p,r=reference(reg,family,seq['task'],seq['command'],[t]);return p[0],r[0]
    def validate(q0,q,dt):
        joints=dict(base);joints.update(zip(kin.joint_names,q));poses=full_fk(collision.urdf,joints,root)
        independent=poses['x5_link6']@homogeneous(np.eye(3),kin.tcp_offset_xyz)
        assert np.max(abs(independent-root@kin.tcp_pose(q)))<1e-9
        hits=collision.check(poses,data['obstacle'],reg['geometry']['obstacle_half_extents_m'])
        hard=bool(np.all(q>=kin.joint_limits[:,0]) and np.all(q<=kin.joint_limits[:,1]))
        speed=bool(np.all(abs(q-q0)/dt<=velocity+1e-10));finite=bool(np.isfinite(independent).all())
        return dict(valid=hard and speed and finite and not hits,hard_ok=hard,velocity_ok=speed,finite=finite,contacts=hits)
    start=validate(q,q,.05);ledger=[];qs=[q.copy()];prefix_time=0.;prefix_q=q.copy()
    if start['valid']:
        for i in range(1,len(times)):
            okay,t,q=continuation(target,lambda p,r,q:solve(kin,root,p,r,q,reg['solver']),validate,float(times[i-1]),q,float(times[i]),ledger)
            prefix_time=t;prefix_q=q.copy()
            if not okay:break
            qs.append(q.copy())
    completed=len(qs);all_q=np.full((len(times),6),np.nan);all_q[:completed]=qs
    fk=np.full((len(times),4,4),np.nan);fk[:completed]=[root@kin.tcp_pose(x) for x in qs]
    qualified=(np.arange(len(times))<completed)&bool(start['valid'])
    end=reg['experts'][seq['task']]['motion_end_s'];active=np.arange(0,len(times),2);active=active[times[active]<end]
    windows=np.array([bool(np.all(qualified[np.minimum(i+np.arange(40),len(times)-1)])) for i in active])
    np.savez_compressed(folder/'geometry.npz',synthetic_reference_time=times,joints=all_q,FK_world=fk,
        desired_position=dp,desired_rotation=dr,sample_qualified=qualified,active_indices=active,
        active_geometry_qualified=windows,initial_TCP_world=np.array(reg['initial_TCP_world']),root_transform=root,
        last_accepted_internal_time=prefix_time,last_accepted_internal_joints=prefix_q)
    write(folder/'solver_ledger.json',dict(start_checks=start,attempts=ledger))
    result=dict(sequence=seq,completed_samples=completed,total_samples=len(times),last_accepted_time=prefix_time,
        status='COMPLETE_GEOMETRY_PASS' if qualified.all() else 'VALID_GEOMETRY_FAIL',internal_solves=len(ledger),
        payload_sha256=sha(folder/'geometry.npz'),physics_steps=0)
    write(folder/'geometry_result.json',result);print(json.dumps(dict(sequence=seq['sequence_id'],status=result['status'],samples=completed)),flush=True)
    return result


def raw_gate(reg,seq):
    folder=ROOT/'proposals'/seq['family_id']/seq['sequence_id'];path=folder/'raw_gate.json'
    if path.exists():return json.loads(path.read_text())
    with np.load(folder/'geometry.npz') as z:a={k:z[k].copy() for k in z.files}
    assert np.isfinite(a['joints']).all(),'incomplete geometry cannot enter raw eligibility'
    gate=TrajectorySafetyGateV1();anchor=np.asarray(reg['initial_TCP_world'])[:3,3];rows=[]
    indices=sorted(set(range(0,len(a['joints']),2))|{round(t/.05) for t in reg['renderer']['calibration_times'][seq['task']]})
    for i in indices:
        t=float(a['synthetic_reference_time'][i]);ii=np.minimum(i+np.arange(40),len(a['joints'])-1)
        chunk=TrajectoryChunkV1(seq['sequence_id']+'-'+str(i),'KINEMATIC_RENDERED_REFERENCE',t,t+np.arange(40)*.05,
            a['desired_position'][ii]-anchor,Rotation.from_matrix(a['desired_rotation'][ii]).as_quat(),
            'NATIVE_WBC_RELATIVE_REFERENCE','x5_link6_gripper_center',t,t+1.95,gripper=np.full(40,.5))
        verdict=gate.evaluate(chunk,now=t,achieved_position=a['FK_world'][i,:3,3]-anchor,
            achieved_orientation=Rotation.from_matrix(a['FK_world'][i,:3,:3]).as_quat()).as_dict()
        rows.append(dict(index=i,time=t,verdict=verdict,active=t<reg['experts'][seq['task']]['motion_end_s']))
    active=[r for r in rows if r['active']];end=reg['experts'][seq['task']]['motion_end_s']
    good=lambda r:r['verdict']['decision']!='REJECT'
    fraction=sum(map(good,active))/len(active)
    quarters=[any(good(r) and k*end/4<=r['time']<(k+1)*end/4 for r in active) for k in range(4)]
    required=[r for r in rows if any(abs(r['time']-t)<1e-8 for t in reg['renderer']['calibration_times'][seq['task']])]
    result=dict(status='PASS' if fraction>=.95 and all(quarters) and all(map(good,required)) else 'VALID_RAW_GATE_FAIL',
        rows=rows,active_windows=len(active),fraction=fraction,quartiles=quarters,required_states_pass=all(map(good,required)))
    write(path,result);return result


def counters():
    results=[json.loads(p.read_text()) for p in (ROOT/'proposals').glob('*/*/geometry_result.json')]
    raw=[json.loads(p.read_text()) for p in (ROOT/'proposals').glob('*/*/raw_gate.json')]
    frames=list((ROOT/'proposals').glob('*/frames/*.npz'))
    return dict(calibration_image_pairs=84,new_kinematic_sequence_attempts=len(list((ROOT/'proposals').glob('*/*/started.json'))),
        completed_geometry_results=len(results),IK_solver_calls=sum(r['internal_solves'] for r in results),
        active_raw_windows=sum(r['active_windows'] for r in raw),production_image_pairs=len(frames),
        model_queries=0,optimizer_updates=0,physics_steps=0,WBC_actor_calls=0)


def run(split):
    import fcntl
    (ROOT/'state').mkdir(exist_ok=True);lock=(ROOT/'state/data_owner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reg=admission();quotas={'DEVELOPMENT':6,'TRAIN':40,'KIN-VAL':10};assert split in quotas
    if split!='DEVELOPMENT':assert json.loads((ROOT/'state/DEV_ACCEPTED.json').read_text())['status']=='PASS'
    families=[f for f in reg['families'] if f['split']==split];accepted=[];outcomes=[]
    for family in families:
        folder=ROOT/'proposals'/family['family_id'];folder.mkdir(parents=True,exist_ok=True)
        if len(accepted)>=quotas[split]:
            outcomes.append(dict(family=family['family_id'],status='NOT_EVALUATED_AFTER_FIXED_QUOTA'));continue
        done=folder/'outcome.json'
        if done.exists():result=json.loads(done.read_text())
        else:
            seqs=[s for s in reg['sequences'] if s['family_id']==family['family_id']]
            geometry_results=[geometry(reg,s) for s in seqs]
            complete=all(x['status']=='COMPLETE_GEOMETRY_PASS' for x in geometry_results)
            result=dict(family=family['family_id'],split=split,geometry=geometry_results,status='VALID_GEOMETRY_FAIL',
                raw_gate='NOT_RUN_GEOMETRY_FAIL',observation='NOT_RUN_GEOMETRY_FAIL')
            if complete:
                gates=[raw_gate(reg,s) for s in seqs];result['raw_gate']=[dict(sequence=s['sequence_id'],status=g['status'],fraction=g['fraction'],quartiles=g['quartiles']) for s,g in zip(seqs,gates)]
                result['status']='VALID_RAW_GATE_FAIL';result['observation']='NOT_RUN_RAW_GATE_FAIL'
                if all(g['status']=='PASS' for g in gates):
                    base='/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/stage13c/runtime_env'
                    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+base+'/public_native/isaacgym/python',
                        LD_LIBRARY_PATH=base+'/public_native/python_env/lib:/usr/lib/x86_64-linux-gnu',CUDA_VISIBLE_DEVICES='GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad')
                    log=folder/'render.log';assert not log.exists(),'unfinished render requires exact same-state recovery'
                    with log.open('xb') as stream:
                        child=subprocess.run([base+'/native_py38_4080/bin/python','-u',str(ROOT/'runtime/pi0260_render_family.py'),family['family_id']],env=env,stdout=stream,stderr=subprocess.STDOUT)
                    # Native teardown can fail after all immutable frames/results have been saved.
                    assert (folder/'observation.json').exists(),('RENDER_INCOMPLETE',child.returncode)
                    obs=json.loads((folder/'observation.json').read_text());result.update(observation=obs,renderer_exit_code=child.returncode,
                        status='ELIGIBLE' if obs['status']=='PASS' else 'VALID_OBSERVATION_FAIL')
            write(done,result)
        outcomes.append(result)
        if result['status']=='ELIGIBLE':accepted.append(family['family_id'])
        count=counters();assert count['active_raw_windows']<=60000 and count['production_image_pairs']<=120000
        write(ROOT/'state/data_progress.json',dict(split=split,accepted=accepted,evaluated=len(outcomes),counters=count))
        print(json.dumps(dict(family=family['family_id'],status=result['status'],accepted=len(accepted),counters=count)),flush=True)
    write(ROOT/'state'/('selection_'+split+'.json'),dict(split=split,selected=accepted,outcomes=outcomes,
        status='DATA_REPLAY_PENDING' if len(accepted)==quotas[split] else 'PI0260_ELIGIBLE_SCENE_SHORTFALL',counters=counters()))


if __name__=='__main__':run(sys.argv[1])
