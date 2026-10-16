"""Independent initial static-IK history audit; no new live or recovery forks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
from PIL import Image

from scripts.pi0292.durable import read,write
from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0294.scene import make_engine,qualify_engine
from scripts.pi0294.teacher import make_teacher
from scripts.pi0294.episode import _robot_contact
from scripts.pi0294.s1_preflight import data_state
from scripts.pi0294.native_dataset import sha
from .scope import validate,Ledger


def capture(engine,teacher):
    python={k:v for k,v in engine.__dict__.items() if k not in ['model','data','ik_data','renderer']}
    return dict(live=data_state(engine.model,engine.data),ik=data_state(engine.model,engine.ik_data),
        python=hashlib.sha256(pickle.dumps(python)).hexdigest(),teacher=hashlib.sha256(pickle.dumps(teacher)).hexdigest())


def replay(root,case,ledger,*,qualifier):
    old=Path(read(root/'E1_INPUTS.private.json')['E0_root']);folder=old/'episodes'/case['attempt_id'];card=read(folder/'CARD.json')
    engine=make_engine(card,read(root/'INPUT_BINDINGS.private.json')['asset_root']);first=engine.snapshot();teacher=make_teacher(card,first['object_pose'],engine)
    observations=sorted((folder/'relay/observations').glob('*.json'));last=len(observations)-1;anchors={0,last//2,last};states={};maximum=0.
    with np.load(folder/'highrate.npz',allow_pickle=False) as z:actual=z['rows'].copy()
    try:
        for sequence in range(last+1):
            state=engine.snapshot();state['robot_object_contact']=_robot_contact(engine,state)
            if sequence==0 and qualifier:
                sample=read(folder/'samples.json')[0];images={}
                for camera in ['body_rgb','wrist_rgb']:
                    item=sample['policy_obs'][camera];p=folder/item['path']
                    if sha(p)!=item['sha256']:raise ValueError('ORIGINAL_INITIAL_RENDER_CHANGED')
                    with Image.open(p) as image:images[camera]=np.asarray(image).copy()
                q=qualify_engine(engine,rendered=(images,sample['geometry']),guard=ledger.guard)
                if q['status']!='STATIC_PASS_PENDING_LIVE_CONTACT':raise ValueError('ORIGINAL_INITIAL_STATIC_QUALIFIER_NOT_REPRODUCED')
            teacher.observe(state)
            if sequence in anchors:states[sequence]=capture(engine,teacher)
            command=folder/'relay/commands'/f'{sequence:03d}.json'
            if not command.exists() or not (folder/'relay/acks'/f'{sequence:03d}.json').exists():break
            action=np.asarray(unseal(read(command)['action'])['payload']['trajectory_eef10']);poses=matrix10(action)
            key='INITIAL-HISTORY-'+('EXACT' if qualifier else 'OLD')+'-'+case['attempt_id']+'-'+str(sequence)
            with ledger.steps('shadow_integrations',200,key) as meter:
                for i in range(200):
                    ledger.guard()
                    if i and i%25==0:
                        s=engine.snapshot();s['robot_object_contact']=_robot_contact(engine,s);teacher.observe(s)
                    before=engine.steps
                    try:engine.step(poses[i//25],float(action[i//25,9]))
                    finally:meter['actual']+=engine.steps-before
                    s=engine.snapshot();row=np.r_[engine.data.time,engine.data.qpos,engine.data.qvel,engine.data.ctrl,engine.data.xfrc_applied[engine.ids['tool']],float(not s['safety_valid'])]
                    error=float(np.max(np.abs(row-actual[engine.steps-1])));maximum=max(maximum,error)
                    if error>1e-10 or engine.safety:raise ValueError('INITIAL_STATIC_HISTORY_REPLAY_ACTUAL_MISMATCH')
        return states,maximum
    finally:engine.close()


def run(root):
    root=Path(root);runtime=validate(root,'4080-simulation');ledger=Ledger(root,'core');start=time.process_time()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    old=Path(read(root/'E1_INPUTS.private.json')['E0_root']);cases=[c for c in read(old/'evidence/model-M2048/REGISTRY.json')['cases'].values() if c['card']['split']=='TRAIN']
    records=[]
    try:
        for case in sorted(cases,key=lambda c:c['panel_index']):
            before,old_error=replay(root,case,ledger,qualifier=False);exact,new_error=replay(root,case,ledger,qualifier=True)
            for sequence,state in exact.items():
                previous=before[sequence];live=float(np.max(np.abs(state['live']-previous['live'])));ik=float(np.max(np.abs(state['ik']-previous['ik'])))
                same=live==ik==0 and state['python']==previous['python'] and state['teacher']==previous['teacher']
                saved=root/'evidence/TRAIN_STATES'/(case['attempt_id']+'-S'+str(sequence).zfill(3))/'integration.npz'
                with np.load(saved,allow_pickle=False) as z:
                    saved_live=float(np.max(np.abs(z['live']-previous['live'])));saved_ik=float(np.max(np.abs(z['ik']-previous['ik'])))
                if saved_live or saved_ik:raise ValueError('EARLIER_CAPTURE_NOT_REPRODUCED')
                record=dict(attempt_id=case['attempt_id'],task=case['card']['task'],sequence=sequence,tick=sequence*200,
                    old_actual_replay_maximum_difference=old_error,exact_static_history_actual_replay_maximum_difference=new_error,
                    live_integration_state_difference=live,IK_integration_state_difference=ik,python_history_identical=state['python']==previous['python'],
                    teacher_history_identical=state['teacher']==previous['teacher'],full_state_identical=same,
                    previously_captured_state_identical_to_old_replay=True,correction_bank_uses_this_anchor=sequence>0)
                records.append(record)
                if sequence>0 and not same:ledger.stop_physics('NONINITIAL_CORRECTION_STATE_HISTORY_MISMATCH',record)
        critical=any(r['sequence']>0 and not r['full_state_identical'] for r in records)
        result=dict(status='STOP_NONINITIAL_CORRECTION_HISTORY_MISMATCH' if critical else 'PASS_ALL_USED_NONINITIAL_CORRECTION_STATES_EXACT_STATIC_HISTORY',
            runtime_source_sha=runtime['code']['vla_commit'],records=records,
            initial_state_queries_with_nonidentical_unused_IK_history=sum(r['sequence']==0 and not r['full_state_identical'] for r in records),
            used_noninitial_states=sum(r['sequence']>0 for r in records),no_new_live_or_recovery_forks=True,CPU_s=time.process_time()-start)
        write(root/'evidence/INITIAL_STATIC_HISTORY_AUDIT.json',result);return result
    except BaseException as exc:
        ledger.stop_physics('INITIAL_STATIC_HISTORY_AUDIT_INTEGRITY_UNRESOLVED',type(exc).__name__+': '+str(exc));raise
    finally:ledger.cpu(time.process_time()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();v=run(a.root);print(json.dumps({k:x for k,x in v.items() if k!='records'}),flush=True)
