"""Fixed recorded/open-clear source continuations. No neural model or task search."""
import argparse,importlib.util,json,os,sys,types
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0279')
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0279/source')
OLD_RAW=RAW.parents[1]/'PI0278/R2'
RELEASE='4afdb1401650353092428dcb69d0cb353ee74164'
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2/code')
from evaluation_card import read,sha,write,freeze


def teacher(t0,j):
    """World wire reference; fixed actual fork anchor, original gripper mapping."""
    j=np.asarray(j);s=np.clip((j-7)/24,0,1);h=10*s**3-15*s**4+6*s**5
    value=np.tile(np.r_[t0[:3,3],t0[:2,:3].reshape(6),1.],(j.size,1))
    value[:,2]+=.08*h
    return value.astype(np.float32)


def differences(actual,expected):
    errors={}
    for name in ('qpos','qvel'):
        a,b=actual[name],expected[name]
        errors[name+'_arm']=float(np.max(np.abs(a[...,:7]-b[...,:7])))
        errors[name+'_fingers']=float(np.max(np.abs(a[...,7:]-b[...,7:])))
    errors['TCP_xyz']=float(np.max(np.abs(actual['tcp'][...,:3,3]-expected['tcp'][...,:3,3])))
    errors['TCP_rotation']=float(np.max((Rotation.from_matrix(actual['tcp'][...,:3,:3]).inv()*Rotation.from_matrix(expected['tcp'][...,:3,:3])).magnitude()))
    for cube in ('cubeA','cubeB'):
        a,b=actual[cube+'_pose'],expected[cube+'_pose'];errors[cube+'_xyz']=float(np.max(np.abs(a[...,:3]-b[...,:3])))
        qa,qb=a[...,3:].astype(np.float64),b[...,3:].astype(np.float64);dot=np.abs(np.sum(qa*qb,axis=-1)/(np.linalg.norm(qa,axis=-1)*np.linalg.norm(qb,axis=-1)))
        errors[cube+'_rotation']=float(np.max(2*np.arccos(np.clip(dot,0,1))))
        for name in ('velocity','angular'):errors[cube+'_'+name]=float(np.max(np.abs(actual[cube+'_'+name]-expected[cube+'_'+name])))
    agreement={k:bool(np.array_equal(actual[k],expected[k])) for k in ('time_s','is_cubeA_grasped','is_cubeA_on_cubeB','is_cubeA_static','success')}
    limits={k:(1e-4 if 'rotation' in k or 'velocity' in k or 'angular' in k or k.startswith('qvel') else 1e-5) for k in errors}
    return dict(errors=errors,limits=limits,agreement=agreement,pass_=all(errors[k]<=limits[k] for k in errors) and all(agreement.values()))


def native():
    admission=read(ROOT/'design/ADOPTION.json');assert admission['release']==RELEASE
    for row in admission['source_files']:assert sha(row['path'])==row['sha256']
    run=types.ModuleType('run');run.ROOT=ROOT;run.sha=sha;run.write=write;sys.modules['run']=run
    path=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0276/code/source_worker.py')
    spec=importlib.util.spec_from_file_location('pi0279_native',path);source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    source.ROOT=ROOT;source.RAW=RAW
    caps=read(ROOT/'authority/REVIEW_AND_EXECUTION_LIMITS.json')['new_hard_caps']
    source.LIMITS={k:caps[v] for k,v in dict(starts='physical_starts',physics='physics_steps',control='control_calls',rgb='RGB_pairs',teacher='teacher_controls').items()}
    freeze(ROOT/'evidence/ACTUAL_NATIVE_IMPORT.json',dict(file=source.__file__,sha256=sha(source.__file__),new_neural_queries=0,release=RELEASE))
    return source


@lru_cache(maxsize=4)
def original(seed):
    folder=OLD_RAW/'episodes'/f'TRAIN_PHYSICS-{seed}-COVER934096'/'attempt-000'
    rec=read(folder/'RESULT.json');assert rec['status']=='VALID_MODEL_PHYSICS_COMPLETE' and rec['control_calls']==640
    for name in ('physics.npz','commands.npz','installed.npz','observations.npz','initial.npz','QUERIES.json','contacts.jsonl'):assert sha(folder/name)==rec['files'][name]
    with np.load(folder/'commands.npz') as z:commands=z['action'].copy()
    with np.load(folder/'installed.npz') as z:actions=z['action'].copy()
    with np.load(folder/'physics.npz') as z:physics={k:z[k].copy() for k in z.files}
    queries=read(folder/'QUERIES.json');assert len(queries)==80 and commands.shape==(640,10)
    return folder,rec,commands,actions,physics,queries


def branch(source,seed,fork,kind):
    slot=f'{kind}-{seed}-f{fork:03d}';old=source.retained(slot)
    if old:return old
    starts=list((RAW/'episodes').glob('*/attempt-*/START.json'))
    if any(p.parent.parent.name==slot for p in starts):
        assert len(starts)-len({p.parent.parent.name for p in starts})<4
    folder,parent,commands,actions,physics,queries=original(seed)
    ep=source.Episode(slot,seed,'pd_ee_pose');gates=[];transport=[];prefix=None;t0=None
    status='VALID_PILOT_COMPLETE'
    try:
        with np.load(folder/'initial.npz') as z:initial={k:z[k] for k in z.files}
        actual=source.flatten(ep.initial);assert actual.keys()==initial.keys()
        initial_errors={k:float(np.max(np.abs(actual[k]-initial[k]))) for k in actual};assert max(initial_errors.values())<=1e-6
        write(ep.folder/'INITIAL_FIXED_PHYSICS_MATCH.json',dict(parent=str(folder),differences=initial_errors))
        for k in range(640):
            if k==fork:
                ids=np.arange(fork*5-40,fork*5+1)
                current={name:np.stack([x[name] for x in ep.states])[ids] for name in physics}
                prefix=differences(current,{name:a[ids] for name,a in physics.items()})
                write(ep.folder/'PREFIX_RECONSTRUCTION.json',dict(prefix,comparison_native_indices=ids.tolist(),parent=str(folder),fork_control=fork))
                if not prefix['pass_']:raise RuntimeError('FIXED_RECONSTRUCTION_CONTRACT_MISMATCH')
                if kind=='OPEN_CLEAR':
                    paired=source.retained(f'RECORDED-{seed}-f{fork:03d}');assert paired
                    with np.load(paired[0]/'physics.npz') as z:
                        pairing=differences(current,{name:z[name][ids] for name in physics})
                    write(ep.folder/'PAIR_PREFIX_RECONSTRUCTION.json',dict(pairing,recorded=str(paired[0]),comparison_native_indices=ids.tolist()))
                    if not pairing['pass_']:raise RuntimeError('FIXED_PAIRED_RECONSTRUCTION_CONTRACT_MISMATCH')
                    assert all(np.isfinite(np.asarray(v)).all() for v in ep.states[-1].values())
                    supported=all(x['is_cubeA_on_cubeB'] for x in ep.states[-21:])
                    if not supported:status='VALID_PILOT_INELIGIBLE';break
                    t0=source.matrix(ep.env.agent.tcp.pose)
                    assert np.isfinite(t0).all();write(ep.folder/'TEACHER_ANCHOR.json',dict(actual_world_TCP=t0.tolist(),fork_control=fork,fork_s=fork*.05,anchor_fixed_once=True))
            observation=ep.observe()
            if k%8==0:
                if kind=='OPEN_CLEAR' and k>=fork:
                    chunk=teacher(t0,np.arange(k-fork,k-fork+40));origin='FIXED_OPEN_CLEAR_TEACHER'
                else:
                    q=queries[k//8];out=Path(q['output']);assert sha(out)==read(out.with_suffix('.receipt.json'))['output_sha256']
                    with np.load(out) as z:chunk=z['decoded'][0].copy()
                    assert np.array_equal(chunk[:8],commands[k:k+8]);origin='ORIGINAL_COMPLETE_H40'
                gate=source.pose_gate(chunk,observation['state_v1'],k*.05,slot)
                gates.append(dict(anchor=k,origin=origin,decision=gate,full_H40=chunk.tolist()))
                if origin=='ORIGINAL_COMPLETE_H40':assert gate['decision']==queries[k//8]['gate']['decision']
                if gate['decision']=='REJECT' or np.min(chunk[:,9])<-.1 or np.max(chunk[:,9])>1.1:
                    status='VALID_PILOT_GATE_FAIL';break
            is_teacher=kind=='OPEN_CLEAR' and k>=fork
            label=teacher(t0,np.array([k-fork]))[0] if is_teacher else commands[k]
            from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
            target=np.eye(4);target[:3,3]=label[:3];target[:3,:3]=rot6d_rows_to_matrix(label[None,3:9])[0]
            action=source.native_action(target,ep.root,float(label[9]))
            if not is_teacher:assert np.array_equal(action,actions[k]),'ORIGINAL_INSTALLED_ACTION_BYTE_IDENTITY'
            ep.step(action,phase=3 if is_teacher else -1,command=label,teacher=is_teacher)
            assert np.max(np.abs(source.matrix(ep.arm._target_pose)-np.linalg.inv(ep.root)@target))<1e-5
            transport.append(dict(control_index=k,raw=float(label[9]),sent=float(np.clip(label[9],0,1))))
        write(ep.folder/'FULL_H40_GATE.json',gates);write(ep.folder/'GRIPPER_TRANSPORT.json',transport)
        write(ep.folder/'PARENT.json',dict(path=str(folder),result_sha256=sha(folder/'RESULT.json'),seed=seed,fork=fork,kind=kind,new_neural_queries=0))
        rec=ep.finish(status)
    except BaseException as err:
        write(ep.folder/'FULL_H40_GATE.json',gates);write(ep.folder/'GRIPPER_TRANSPORT.json',transport)
        ep.finish('ENGINEERING_INCOMPLETE',repr(err));raise
    with np.load(ep.folder/'physics.npz') as z:
        ids=np.arange(2800,3001,5)
        measured={}
        if len(z['time_s'])>ids[-1]:
            measured=dict(opening_m_min=float((z['qpos'][ids,-2:].sum(axis=1)-.00004).min()),clearance_z_m_min=float((z['tcp'][ids,2,3]-t0[2,3]).min()) if t0 is not None else None)
    positive=kind=='OPEN_CLEAR' and status=='VALID_PILOT_COMPLETE' and rec['persistent_success'] and ep.control==640 and rec['ik_holds']==0 and measured['opening_m_min']>=.075 and measured['clearance_z_m_min']>=.06
    write(ep.folder/'SUFFIX_QUALIFICATION.json',dict(seed=seed,fork=fork,kind=kind,positive=positive,measured=measured,persistent_success=rec['persistent_success'],status=status,full_execution=ep.control==640,result_sha256=sha(ep.folder/'RESULT.json'),new_neural_queries=0))
    return ep.folder,rec


def verdict():
    rows=[]
    for seed in range(10000,10004):
        for fork in (160,320,480):
            own={}
            for kind in ('RECORDED','OPEN_CLEAR'):
                candidates=list((RAW/'episodes'/f'{kind}-{seed}-f{fork:03d}').glob('attempt-*/SUFFIX_QUALIFICATION.json'));assert len(candidates)==1
                own[kind]=read(candidates[0]);own[kind]['path']=str(candidates[0].parent)
            assert own['RECORDED']['full_execution']
            # The whole recorded control must reproduce its original registered verdict.
            original_success=original(seed)[1]['persistent_success'];assert own['RECORDED']['persistent_success']==original_success,'RECORDED_CONTROL_DID_NOT_REPRODUCE'
            rows.append(dict(seed=seed,fork=fork,branches=own,conversion=not own['RECORDED']['persistent_success'] and own['OPEN_CLEAR']['positive']))
    n=sum(x['branches']['OPEN_CLEAR']['positive'] for x in rows)
    per={str(s):sum(x['branches']['OPEN_CLEAR']['positive'] for x in rows if x['seed']==s) for s in range(10000,10004)}
    gate=n>=10 and min(per.values())>=2 and any(x['conversion'] and x['seed'] in (10000,10001) for x in rows)
    value=dict(status='PASS_FIXED_PAIRED_TEACHER_DATA_GATE' if gate else 'VALID_TEACHER_HYPOTHESIS_FAIL',rows=rows,positive=n,denominator=12,per_state=per,launch_allowed=gate,source_continuations_not_model_success=True,new_neural_queries=0,selected_checkpoint='NONE',deployment='NONE')
    freeze(ROOT/'evidence/PILOT_VERDICT.json',value);print(json.dumps(value),flush=True)


def audit():
    """Saved terminal states/contact identities only; no new simulator or model."""
    from collections import Counter
    rows=[]
    for seed in range(10000,10004):
        folder,rec,commands,actions,z,queries=original(seed);ids=np.arange(2800,3001,5)
        pairs=Counter();contact_samples=set();max_impulse={}
        with (folder/'contacts.jsonl').open() as stream:
            for line in stream:
                sample=json.loads(line)
                if not 2800<=sample['physics_sample']<=3000:continue
                for c in sample['contacts']:
                    pair=' | '.join(sorted(c['bodies']));pairs[pair]+=1;contact_samples.add(sample['physics_sample'])
                    for point in c['points']:
                        if 'impulse' in point:max_impulse[pair]=max(max_impulse.get(pair,0.),float(np.linalg.norm(point['impulse'])))
        rows.append(dict(seed=seed,path=str(folder),result_sha256=sha(folder/'RESULT.json'),persistent_success=rec['persistent_success'],
            native_41={k:int(z[k][ids].sum()) for k in ('is_cubeA_on_cubeB','is_cubeA_static','is_cubeA_grasped','success')},
            actual_opening_m_range=[float((z['qpos'][ids,-2:].sum(axis=1)-.00004).min()),float((z['qpos'][ids,-2:].sum(axis=1)-.00004).max())],
            command_gripper_range=[float(commands[560:601,9].min()),float(commands[560:601,9].max())],
            cubeA_speed_m_s_max=float(np.linalg.norm(z['cubeA_velocity'][2800:3001],axis=1).max()),
            cubeA_angular_rad_s_max=float(np.linalg.norm(z['cubeA_angular'][2800:3001],axis=1).max()),
            named_contact_pairs=dict(pairs),named_contact_max_impulse=max_impulse,contact_native_samples=len(contact_samples),
            fork_eligibility={str(f):bool(z['is_cubeA_on_cubeB'][5*f-20:5*f+1].all()) for f in (160,320,480)}))
    freeze(ROOT/'evidence/SAVED_TERMINAL_AUDIT.json',dict(rows=rows,new_physics=0,new_neural_queries=0,
        cause='UNIQUE_CONTACT_OR_SUPPORT_CAUSE_NOT_ESTABLISHED',contact_counts_not_causal_proof=True))
    print(json.dumps(rows),flush=True)


def check():
    t=np.eye(4);t[:3,3]=[.2,-.1,.3];x=teacher(t,np.arange(80))
    assert np.array_equal(x[:8,:3],np.tile(t[:3,3].astype(np.float32),(8,1))) and np.all(x[:,9]==1)
    assert np.isclose(x[31,2],.38) and np.all(x[31:,2]==x[31,2]) and np.all(np.diff(x[:,2])>=0)
    assert np.all(x[:,0]==np.float32(.2)) and np.all(x[:,1]==np.float32(-.1))
    assert np.array_equal(teacher(t,np.arange(8,48)),x[8:48]);assert all(640-f-39>0 for f in (160,320,480))
    q=Rotation.from_euler('XYZ',[.13,.24,.36]).as_quat()[[3,0,1,2]].astype(np.float32)
    row=dict(qpos=np.zeros((2,9)),qvel=np.zeros((2,9)),tcp=np.tile(t,(2,1,1)))
    for cube in ('cubeA','cubeB'):
        row[cube+'_pose']=np.tile(np.r_[0.,0.,0.,q],(2,1));row[cube+'_velocity']=np.zeros((2,3));row[cube+'_angular']=np.zeros((2,3))
    for name in ('time_s','is_cubeA_grasped','is_cubeA_on_cubeB','is_cubeA_static','success'):row[name]=np.zeros(2)
    assert differences(row,row)['pass_']
    changed={k:v.copy() for k,v in row.items()};changed['qpos'][0,0]=1.01e-5
    assert not differences(changed,row)['pass_']
    print('PASS_FIXED_TEACHER_TIMING_CLEARANCE_PREFIX_AND_H40_FUNCTION')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['check','run','audit','verdict']);a=p.parse_args()
    if a.stage=='check':check()
    elif a.stage=='audit':audit()
    elif a.stage=='verdict':verdict()
    else:
        import fcntl
        (ROOT/'runtime').mkdir(exist_ok=True)
        with (ROOT/'runtime/source_owner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);source=native()
            for seed in range(10000,10004):
                for fork in (160,320,480):
                    for kind in ('RECORDED','OPEN_CLEAR'):
                        folder,rec=branch(source,seed,fork,kind);print('ACTUAL_FIXED_BRANCH',seed,fork,kind,rec['status'],rec['persistent_success'],flush=True)
            verdict()
