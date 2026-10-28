"""Verify saved Oracle ACKs, original-clock scores and actual paired-camera video."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.score import score
from .audit import video
from .scope import validate,Ledger


def audit(root,folder,cases):
    out=root/'evidence/fork-audits'/folder.name
    if out.exists():return read(out/'AUDIT.private.json')
    out.mkdir(parents=True,mode=0o700);result=read(folder/'RESULT.json');checks={}
    def check(name,value):checks[name]=bool(value)
    acknowledgments=read(folder/'PREFIX_ACKS.json')
    begin=result['anchor_tick'];tick=begin;commands=[]
    for i,ack in enumerate(acknowledgments):
        p=folder/f'prefix-{i:03d}.npz'
        check('ACK_hash_'+str(i),sha(p)==ack['physical_receipt_sha256'])
        check('ACK_clock_'+str(i),ack['tick_before']==tick and ack['tick_after']==tick+200 and ack['integrations']==200 and ack['commands']==8)
        with np.load(p,allow_pickle=False) as z:
            check('ACK_saved_clock_'+str(i),int(z['tick_before'])==tick and int(z['tick_after'])==tick+200)
            commands.extend(z['expert_commands'].copy())
        tick+=200
    with np.load(folder/'expert_labels.npz',allow_pickle=False) as z:
        check('actual_ACK_command_labels_exact',np.array_equal(np.asarray(commands).reshape(-1,10),z['committed_absolute10']))
        check('complete_H40_no_padded_tail',int(z['valid_window'].sum())==max(0,len(commands)-39))
        paths=z['rgb_paths'].copy();digests=z['rgb_sha256'].copy()
    check('actual_labels_hash',sha(folder/'expert_labels.npz')==result['labels_sha256'])
    check('ACK_totals',len(commands)==result['actual_expert_commands'] and tick-begin==result['actual_integrations'])
    with np.load(folder/'actual_highrate.npz',allow_pickle=False) as z:high=z['rows'].copy()
    check('actual_highrate_count_clock',len(high)==tick-begin and np.isfinite(high).all() and np.allclose(high[:,0],np.arange(begin+1,tick+1)*.002,atol=1e-8,rtol=0))
    check('no_native_safety_flags',not np.any(high[:,-1]))
    for row,hashes in zip(paths,digests):
        for name,digest in zip(row,hashes):
            p=(folder/str(name)).resolve()
            if not p.is_relative_to(folder.resolve()) or sha(p)!=str(digest):raise ValueError('ACTUAL_FORK_RGB_HASH')
    check('native_S1_parity',result['maximum_live_shadow_difference']<=1e-10)
    case=cases[result['source_student_episode']];old=Path(read(root/'E1_INPUTS.private.json')['E0_root'])/'episodes'/case['attempt_id']
    if result['full_recovery'] and result['status']=='FULL_ORIGINAL_CLOCK_COMPLETE':
        with np.load(folder/'original_clock_joined_physics.npz',allow_pickle=False) as z:joined={k:z[k].copy() for k in z.files}
        check('original_32s_clock_exact',len(joined['time_s'])==641 and np.allclose(joined['time_s'],np.arange(641)*.05,atol=1e-8,rtol=0))
        check('independent_original_score_exact',score(joined,case['card'],complete=True)==read(folder/'SCORE.json')==result['score'])
    restored=begin>0
    check('history_adjudication_present',read(root/'evidence/ORACLE_HISTORY_ADJUDICATION.json')['status']=='C0_ALL_USED_CORRECTION_ANCHORS_EXACT_ORIGINAL_STATIC_HISTORY')
    views=out/'actual-frames';origins=[]
    if result['full_recovery']:
        for i in range(begin//25):origins.append((old,i,'ORIGINAL_E0_STUDENT'))
    origins.extend((folder,i,'ACTUAL_ORACLE_EXPERT') for i in range(len(commands)))
    for camera in ['body_rgb','wrist_rgb']:
        target=views/camera;target.mkdir(parents=True,mode=0o700)
        for i,(source,index,_) in enumerate(origins):
            src=source/camera/f'{index:04d}.png';dst=target/f'{i:04d}.png'
            try:os.link(src,dst)
            except OSError:shutil.copy2(src,dst)
    movie=video(views,out/'actual-original-clock-body-wrist.mp4',len(origins)) if origins else None
    if movie:movie.update(logical_id='E1_ORACLE/'+folder.name,relative_path=str((out/'actual-original-clock-body-wrist.mp4').relative_to(root)),
        original_student_prefix_frames=begin//25 if result['full_recovery'] else 0,actual_expert_frames=len(commands),
        first_native_time_s=0. if result['full_recovery'] else begin*.002,last_native_frame_time_s=(tick-25)*.002,
        final_32s_camera_fabricated=False,restoration_qualified=restored)
    value=dict(status='PASS_INDEPENDENT_SAVED_FORK_AUDIT' if all(checks.values()) else 'FAIL_PRESERVED_FORK_AUDIT',
        id=folder.name,checks=checks,result=result,video=movie,
        restoration='EXACT_NONINITIAL_HISTORY' if restored else 'INVALID_INITIAL_STATIC_IK_HISTORY_NOT_USED_FOR_LABELS',
        task_qualification_evidence_eligible=restored,source_safety_failure=bool(result['live_native_failures'] or np.any(high[:,-1])))
    write(out/'AUDIT.private.json',value);return value


def run(root):
    root=Path(root);validate(root,'4080-simulation');ledger=Ledger(root,'core');ledger.guard(closeout=True)
    if read(ledger.path)['reservations']:raise PermissionError('NO_FORK_AUDIT_WITH_UNSETTLED_PHYSICS')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));start=time.process_time();child=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        old=Path(read(root/'E1_INPUTS.private.json')['E0_root']);cases=read(old/'evidence/model-M2048/REGISTRY.json')['cases'];rows=[]
        for folder in sorted((root/'forks').iterdir()):
            ledger.guard(closeout=True)
            if (folder/'RESULT.json').exists():rows.append(audit(root,folder,cases))
        value=dict(status='PASS_ALL_SAVED_FORK_EVIDENCE' if len(rows)==16 and all(x['status']=='PASS_INDEPENDENT_SAVED_FORK_AUDIT' for x in rows) else 'PRESERVED_FORK_AUDIT_GAPS',
            forks=len(rows),initial_restoration_invalid=sum(not r['task_qualification_evidence_eligible'] for r in rows),rows=rows)
        write(root/'evidence/FORK_EVIDENCE_AUDIT_INDEX.private.json',value);return value
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN);ledger.cpu(time.process_time()-start+after.ru_utime+after.ru_stime-child.ru_utime-child.ru_stime)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();v=run(a.root);print(json.dumps(dict(status=v['status'],forks=v['forks'])),flush=True)
