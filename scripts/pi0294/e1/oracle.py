"""Original Teacher continuations from verified actual student-state forks."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.s1_preflight import clone,predict,bind_install,observed_state
from scripts.pi0294.episode import _robot_contact
from scripts.pi0294.score import score
from scripts.pi0294.native_dataset import sha
from go2arx5_vla.task_data.codec import pose10
from go2arx5_vla.target_sim.scorer import BOOL_FIELDS
from .collision import reconstruct_case
from .scope import validate,Ledger


def atomic_npz(path,**arrays):
    path=Path(path);temp=path.with_name(path.name+'.pending.'+str(os.getpid()))
    with temp.open('xb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    temp.chmod(0o600);os.replace(temp,path)


def physical(state):
    return {('tcp_world' if k=='tcp' else k):copy.deepcopy(state[k]) for k in
        ['time_s','tcp','object_pose','object_linear_velocity','object_angular_velocity','robot_object_contact',*BOOL_FIELDS]}


def fork(root,case,anchor,original_engine,original_teacher,ledger,*,full):
    identifier=case['attempt_id']+'-S'+str(anchor).zfill(3)+('-FULL' if full else '-SHORT')
    folder=root/'forks'/identifier;folder.mkdir(mode=0o700)
    ledger.charge('recovery_forks',1)
    engine,teacher=clone(original_engine),copy.deepcopy(original_teacher)
    begin=engine.steps;end=16000 if full else min(begin+1000,16000)
    samples=[];states=[];commands=[];contacts=[];contact_times=[];highrate=[];acks=[];forecasts=[];maximum=0.
    status='ENGINEERING_INCOMPLETE';error=None;native_fail=False
    try:
        while engine.steps<end:
            ledger.guard()
            state=engine.snapshot();state['robot_object_contact']=_robot_contact(engine,state);teacher.observe(state)
            images,_=engine.render(state)
            plan=teacher.plan(state);poses,opening=plan['poses'][:8],plan['opening'][:8];nonce=str(uuid.uuid4())
            with ledger.steps('shadow_integrations',200,identifier+'-SHADOW-'+str(engine.steps)) as meter:
                def charge(count,*,invocation):
                    if not invocation:meter['actual']+=count
                forecast,expected=predict(engine,poses,opening,nonce=nonce,guard=ledger.guard,charge=charge)
            forecasts.append(forecast);write(folder/'S1_PREFLIGHTS.json',forecasts)
            if forecast['decision']!='PASS':status='VALID_ORACLE_S1_REJECT';break
            bind_install(engine,poses,opening,forecast,nonce)
            tick_before=engine.steps;prefix_rows=[];prefix_commands=[]
            write(folder/'PENDING_PREFIX.json',dict(status='RESERVED_BEFORE_FIRST_INTEGRATION',tick_before=tick_before,forecast_sha256=hashlib.sha256(json.dumps(forecast,sort_keys=True).encode()).hexdigest()))
            with ledger.steps('live_and_recovery_integrations',200,identifier+'-ACTUAL-'+str(tick_before)) as live:
                for point in range(8):
                    if point:
                        state=engine.snapshot();state['robot_object_contact']=_robot_contact(engine,state);teacher.observe(state)
                        images,_=engine.render(state)
                    index=len(commands);rgb=[];rgb_hashes=[]
                    for camera in ['body_rgb','wrist_rgb']:
                        image=folder/camera/f'{index:04d}.png';image.parent.mkdir(exist_ok=True,mode=0o700)
                        Image.fromarray(images[camera]).save(image);image.chmod(0o600);rgb.append(str(image.relative_to(folder)));rgb_hashes.append(sha(image))
                    states.append(pose10(state['tcp'],state['gripper_fraction']));samples.append(dict(physical=physical(state),rgb_paths=rgb,rgb_sha256=rgb_hashes,expert_stage=teacher.stage))
                    command=pose10(poses[point],float(opening[point]));commands.append(command);prefix_commands.append(command)
                    for k in range(25):
                        previous=engine.steps
                        try:engine.step(poses[point],float(opening[point]))
                        except Exception:
                            native_fail=True;raise
                        finally:live['actual']+=engine.steps-previous
                        actual=engine.snapshot();row=observed_state(engine);prefix_rows.append(row)
                        maximum=max(maximum,float(np.max(np.abs(row-expected[point*25+k]))))
                        if maximum>1e-10:raise ValueError('E1_FORK_LIVE_SHADOW_STATE_MISMATCH')
                        highrate.append(np.r_[engine.data.time,engine.data.qpos,engine.data.qvel,engine.data.ctrl,engine.data.xfrc_applied[engine.ids['tool']],float(not actual['safety_valid'])])
                        contact_times.append(actual['time_s']);contacts.append([float(_robot_contact(engine,actual)),float(actual['native_grasp']),
                            float(np.linalg.norm(actual['object_pose'][:2,3]-teacher.initial[:2,3])),float(actual['object_pose'][2,3]),float(not actual['safety_valid'])])
            prefix=folder/('prefix-'+str(len(acks)).zfill(3)+'.npz')
            atomic_npz(prefix,observed_state=np.asarray(prefix_rows),expert_commands=np.asarray(prefix_commands),tick_before=tick_before,tick_after=engine.steps)
            acks.append(dict(tick_before=tick_before,tick_after=engine.steps,integrations=200,commands=8,physical_receipt_sha256=sha(prefix)))
            write(folder/'PREFIX_ACKS.json',acks)
            write(folder/'PENDING_PREFIX.json',dict(status='ACKNOWLEDGED',tick_before=tick_before,tick_after=engine.steps,physical_receipt_sha256=sha(prefix)))
        if engine.steps==end:status='FULL_ORIGINAL_CLOCK_COMPLETE' if full else 'SAFE_TWO_SECOND_CONTINUATION'
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        if native_fail or 'STATE_MISMATCH' in str(exc):
            ledger.stop_physics('ACTUAL_FORK_SAFETY_OR_LIVE_SHADOW_INTEGRITY',error)
            status='STOP_ACTUAL_SAFETY_OR_INTEGRITY'
        else:status='ENGINEERING_INCOMPLETE'
    finally:
        final=engine.snapshot();final['robot_object_contact']=_robot_contact(engine,final)
        current=[s['physical'] for s in samples]+[physical(final)]
        arrays={k:np.asarray([s[k] for s in current]) for k in current[0]}
        arrays['contact_time_s']=np.asarray(contact_times);arrays['contact_audit']=np.asarray(contacts).reshape(-1,5)
        atomic_npz(folder/'actual_physics.npz',**arrays)
        atomic_npz(folder/'actual_highrate.npz',rows=np.asarray(highrate))
        result_score=None
        if full and status=='FULL_ORIGINAL_CLOCK_COMPLETE':
            old=Path(read(root/'E1_INPUTS.private.json')['E0_root'])/'episodes'/case['attempt_id']
            with np.load(old/'physics.npz',allow_pickle=False) as z:prefix={k:z[k].copy() for k in z.files}
            before=prefix['time_s']<begin*.002-1e-8
            joined={k:np.concatenate([prefix[k][before],arrays[k]],axis=0) for k in current[0]}
            joined['contact_time_s']=np.concatenate([prefix['contact_time_s'][:begin],arrays['contact_time_s']])
            joined['contact_audit']=np.concatenate([prefix['contact_audit'][:begin],arrays['contact_audit']])
            atomic_npz(folder/'original_clock_joined_physics.npz',**joined)
            result_score=score(joined,case['card'],complete=True)
            write(folder/'SCORE.json',result_score)
        n=len(commands);valid=max(0,n-39)
        # Only actual executed expert commands form future H40 labels.
        atomic_npz(folder/'expert_labels.npz',actual_state10=np.asarray(states).reshape(-1,10),
            committed_absolute10=np.asarray(commands).reshape(-1,10),valid_window=np.arange(n)<valid,
            rgb_paths=np.asarray([s['rgb_paths'] for s in samples]),rgb_sha256=np.asarray([s['rgb_sha256'] for s in samples]),
            instruction=np.asarray(case['card']['instruction']),label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'))
        result=dict(id=identifier,task=case['card']['task'],source_student_episode=case['episode_uuid'],
            anchor_sequence=anchor,anchor_tick=begin,full_recovery=full,original_clock_end=engine.data.time,
            status=status,error=error,actual_integrations=engine.steps-begin,actual_expert_commands=n,valid_H40_windows=valid,
            actual_ACKs=len(acks),S1_rejections=sum(r['decision']=='REJECT' for r in forecasts),live_native_failures=int(native_fail),
            maximum_live_shadow_difference=maximum,task_success=bool(result_score and result_score['persistent_success']),
            score=result_score,origin='E0_STATE_AGGREGATION_ONLY',not_new_M1_online_DAgger=True,
            student_prefixes_before_anchor=begin//200,expert_state_samples_not_student_state=True,
            labels_sha256=sha(folder/'expert_labels.npz'),engine_history_copied=True,original_32s_clock_preserved=True)
        write(folder/'RESULT.json',result);engine.close()
    return result


def run(root):
    root=Path(root);manifest=validate(root,'4080-simulation',physical=True);ledger=Ledger(root,'core')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:3]));start=time.process_time();lock=OwnerLock(root/'E1_SOURCE_OWNER.lock')
    old=Path(read(root/'E1_INPUTS.private.json')['E0_root']);registry=read(old/'evidence/model-M2048/REGISTRY.json')
    cases=sorted([c for c in registry['cases'].values() if c['card']['split']=='TRAIN'],key=lambda c:c['panel_index'])
    states={};results=[]
    try:
        for case in cases:
            replay,anchors=reconstruct_case(root,case,ledger,save=False,diagnose=False)
            if replay['maximum_actual_highrate_difference']>1e-10:raise ValueError('ORACLE_REPLAY_HISTORY_MISMATCH')
            states[case['attempt_id']]=anchors
        for case in cases:
            for anchor,(engine,teacher) in sorted(states[case['attempt_id']].items()):
                if (root/'SOURCE_PHYSICS_STOP.json').exists():break
                results.append(fork(root,case,anchor,engine,teacher,ledger,full=False))
                write(root/'evidence/ORACLE_RECOVERY_PROGRESS.json',dict(results=results))
        for case in cases:
            if (root/'SOURCE_PHYSICS_STOP.json').exists():break
            anchor=max(states[case['attempt_id']]);engine,teacher=states[case['attempt_id']][anchor]
            results.append(fork(root,case,anchor,engine,teacher,ledger,full=True))
            write(root/'evidence/ORACLE_RECOVERY_PROGRESS.json',dict(results=results))
        tasks={}
        for task in ['T1','T2']:
            chosen=[r for r in results if r['task']==task];full=[r for r in chosen if r['full_recovery']]
            valid=[r for r in chosen if r['actual_ACKs'] and not r['live_native_failures'] and r['maximum_live_shadow_difference']<=1e-10]
            episodes={r['source_student_episode'] for r in valid if r['student_prefixes_before_anchor']>0}
            qualified=bool(any(r['task_success'] for r in full) and len(episodes)>=2 and not (root/'SOURCE_PHYSICS_STOP.json').exists())
            tasks[task]=dict(status='RECOVERY_QUALIFIED_FOR_E1' if qualified else 'NOT_QUALIFIED',
                short_attempted=sum(not r['full_recovery'] for r in chosen),full_attempted=len(full),full_successes=sum(r['task_success'] for r in full),
                distinct_student_episode_anchors=len(episodes),complete_executed_H40_windows=sum(r['valid_H40_windows'] for r in valid),
                new_M1_online_DAgger=False)
        value=dict(status='ACTUAL_SOURCE_ORACLE_RESULTS_NOT_TARGET_WBC_QUALIFICATION',runtime_source_sha=manifest['code']['vla_commit'],tasks=tasks,results=results,CPU_s=time.process_time()-start)
        write(root/'evidence/ORACLE_RECOVERY_RESULTS.json',value)
        return value
    finally:
        for anchors in states.values():
            for engine,_ in anchors.values():engine.close()
        ledger.cpu(time.process_time()-start);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();v=run(a.root);print(json.dumps(dict(status=v['status'],tasks=v['tasks'],CPU_s=v['CPU_s'])),flush=True)
