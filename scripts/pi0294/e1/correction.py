"""Actual executed source correction labels; no rejected plans or padded tails."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha,NativeDataset
from .artifacts import atomic_npz
from .scope import validate,Ledger,STATS


def window_key(state,commands,rgb_sha):
    return hashlib.sha256(np.asarray(state).tobytes()+np.asarray(commands).tobytes()+json.dumps(list(rgb_sha)).encode()).hexdigest()


def export_recovery(root,statistics):
    root=Path(root);m=validate(root,'4080-simulation',physical=True);ledger=Ledger(root,'core');ledger.guard();started=time.process_time()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    if sha(statistics)!=STATS:raise ValueError('E1_CORRECTION_USES_ORIGINAL_STATS_ONLY')
    oracle=read(root/'evidence/ORACLE_RECOVERY_RESULTS.json');qualified={t for t,v in oracle['tasks'].items() if v['status']=='RECOVERY_QUALIFIED_FOR_E1'}
    out=root/'datasets/E1_SOURCE_CORRECTION_V1-C0';out.mkdir(mode=0o700)
    (out/'numeric').mkdir();(out/'raw').mkdir();rows=[];seen=set();duplicates=0
    try:
        for result in oracle['results']:
            ledger.guard()
            if result['task'] not in qualified or not result['student_prefixes_before_anchor'] or not result['valid_H40_windows']:continue
            if result['live_native_failures'] or result['maximum_live_shadow_difference']>1e-10 or result['status'].startswith('STOP'):continue
            folder=root/'forks'/result['id'];source=folder/'expert_labels.npz'
            if sha(source)!=result['labels_sha256']:raise ValueError('ACTUAL_RECOVERY_LABELS_CHANGED')
            acks=read(folder/'PREFIX_ACKS.json')
            for index,ack in enumerate(acks):
                p=folder/('prefix-'+str(index).zfill(3)+'.npz')
                if sha(p)!=ack['physical_receipt_sha256'] or ack['integrations']!=200 or ack['commands']!=8:raise ValueError('CORRECTION_ACTUAL_PREFIX_ACK_CHANGED')
            with np.load(source,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
            n=len(data['committed_absolute10'])
            if n!=sum(a['commands'] for a in acks) or not np.isfinite(data['committed_absolute10']).all():raise ValueError('CORRECTION_UNEXECUTED_OR_NONFINITE_COMMAND')
            legal=[]
            for anchor in np.flatnonzero(data['valid_window']):
                if anchor+40>n:raise ValueError('CORRECTION_FULL_ACTUAL_H40_REQUIRED')
                key=window_key(data['actual_state10'][anchor],data['committed_absolute10'][anchor:anchor+40],data['rgb_sha256'][anchor])
                if key in seen:duplicates+=1;continue
                seen.add(key);legal.append(int(anchor))
            if not legal:continue
            data['valid_window']=np.isin(np.arange(n),legal)
            raw=out/'raw'/result['id'];raw.mkdir()
            for paths,hashes in zip(data['rgb_paths'],data['rgb_sha256']):
                for name,digest in zip(paths,hashes):
                    if Path(str(name)).is_absolute() or '..' in Path(str(name)).parts:raise ValueError('CORRECTION_RELATIVE_RGB_PATH_REQUIRED')
                    old=folder/str(name)
                    if not old.resolve().is_relative_to(folder.resolve()) or sha(old)!=str(digest):raise ValueError('CORRECTION_ACTUAL_RGB_CHANGED')
                    target=raw/str(name);target.parent.mkdir(exist_ok=True)
                    if not target.exists():shutil.copyfile(old,target)
                    target.chmod(0o600)
            numeric=out/'numeric'/(result['id']+'.npz');atomic_npz(numeric,**data)
            row=dict(id=result['id'],task=result['task'],family='E0_STUDENT_STATE',split='TRAIN',origin='E0_STATE_AGGREGATION_ONLY',
                file=str(numeric.relative_to(out)),sha256=sha(numeric),bytes=numeric.stat().st_size,raw_root='EXPLICIT_PRIVATE_BINDING_REQUIRED',
                source_student_episode=result['source_student_episode'],anchor_tick=result['anchor_tick'],windows=len(legal),legal_anchors=legal,
                expert_commands=n,discarded_tail_windows=min(39,n),labels_sha256=result['labels_sha256'],
                actual_ACKs=len(acks),full_recovery=result['full_recovery'],full_recovery_success=result['task_success'],
                execution_actor='ORIGINAL_QUALIFIED_SOURCE_EXPERT',privileged_audit_only=True,source_result_sha256=sha(folder/'RESULT.json'))
            rows.append(row)
        if not rows:raise ValueError('EMPTY_CORRECTION_BANK_NOT_READY')
        shutil.copyfile(statistics,out/'statistics.json')
        manifest=dict(schema='E1_SOURCE_CORRECTION_V1',domain='FIXED_BASE_ARX5_TASK_SIM',frozen=True,
            statistics_sha256=STATS,original_stats_fit=False,episodes=rows,qualified_tasks=sorted(qualified),
            scope='TRAIN_ONLY_SOURCE_D0_ORIGINAL_S1',origin='E0_STATE_AGGREGATION_ONLY',new_M1_online_DAgger=False,
            runtime_source_sha=m['code']['vla_commit'],oracle_receipt_sha256=sha(root/'evidence/ORACLE_RECOVERY_RESULTS.json'),
            identical_windows_deduplicated=duplicates,student_state_vs_expert_state='ANCHOR_STUDENT_VISITED_SUBSEQUENT_ROWS_EXPERT_VISITED')
        write(out/'DATASET_MANIFEST.json',manifest)
        bindings={r['id']:str(out/'raw'/r['id']) for r in rows};write(root/'datasets/C0_RAW_BINDINGS.private.json',bindings)
        with (out/'windows.jsonl').open('x') as f:
            for row in rows:
                with np.load(out/row['file'],allow_pickle=False) as z:
                    for anchor in row['legal_anchors']:
                        f.write(json.dumps(dict(episode_id=row['id'],task=row['task'],split='TRAIN',anchor=anchor,
                            source_student_episode=row['source_student_episode'],anchor_tick=row['anchor_tick'],origin=row['origin'],
                            current_state_actor='STUDENT_AT_FORK_ANCHOR' if anchor==0 else 'EXPERT_AFTER_FORK',
                            execution_actor=row['execution_actor'],state10=z['actual_state10'][anchor].tolist(),
                            committed_absolute_H40=z['committed_absolute10'][anchor:anchor+40].tolist(),numeric_sha256=row['sha256'],
                            rgb_paths=z['rgb_paths'][anchor].tolist(),rgb_sha256=z['rgb_sha256'][anchor].tolist()),allow_nan=False)+'\n')
        audit=independent_read(out,bindings)
        tasks={t:dict(windows=sum(r['windows'] for r in rows if r['task']==t),
            distinct_student_episodes=len({r['source_student_episode'] for r in rows if r['task']==t})) for t in sorted(qualified)}
        if any(v['windows']<8 or v['distinct_student_episodes']<2 for v in tasks.values()):raise ValueError('E1_ACTUAL_CORRECTION_THRESHOLD_NOT_MET')
        card=dict(status='ACTUAL_NONEMPTY_RECOVERY_BANK_INDEPENDENT_READER_PASS',origin='E0_STATE_AGGREGATION_ONLY',
            manifest_sha256=sha(out/'DATASET_MANIFEST.json'),statistics_sha256=STATS,tasks=tasks,rows=len(rows),
            complete_H40_windows=sum(r['windows'] for r in rows),identical_windows_deduplicated=duplicates,
            original_data_changed=False,new_online_DAgger=False,target_WBC_demo=False,reader=audit,CPU_s=time.process_time()-started)
        write(out/'CORRECTION_DATASET_CARD.json',card);return card
    finally:ledger.cpu(time.process_time()-started)


def independent_read(root,bindings):
    root=Path(root);manifest=read(root/'DATASET_MANIFEST.json');count=0;rgb=0
    if sha(root/'statistics.json')!=STATS:raise ValueError('INDEPENDENT_CORRECTION_STATS_CHANGED')
    stream=(root/'windows.jsonl').open()
    try:
        for row in manifest['episodes']:
            if row['split']!='TRAIN' or row['task'] not in manifest['qualified_tasks'] or sha(root/row['file'])!=row['sha256']:raise ValueError('INDEPENDENT_CORRECTION_MANIFEST')
            with np.load(root/row['file'],allow_pickle=False) as z:
                n=len(z['committed_absolute10']);valid=np.flatnonzero(z['valid_window']).tolist()
                if valid!=row['legal_anchors'] or len(valid)!=row['windows'] or n!=row['expert_commands']:raise ValueError('INDEPENDENT_CORRECTION_WINDOW_COUNT')
                for anchor in valid:
                    record=json.loads(next(stream))
                    if record['episode_id']!=row['id'] or record['anchor']!=anchor or not np.array_equal(record['committed_absolute_H40'],z['committed_absolute10'][anchor:anchor+40]):raise ValueError('INDEPENDENT_JSONL_NUMERIC_H40_MISMATCH')
                    for name,digest in zip(z['rgb_paths'][anchor],z['rgb_sha256'][anchor]):
                        p=Path(bindings[row['id']])/str(name)
                        if sha(p)!=str(digest):raise ValueError('INDEPENDENT_CORRECTION_RGB_HASH')
                        with Image.open(p) as image:
                            if image.mode!='RGB' or image.size!=(256,256):raise ValueError('INDEPENDENT_CORRECTION_RGB_SHAPE')
                        rgb+=1
                    count+=1
        if next(stream,None) is not None:raise ValueError('UNEXPECTED_EXTRA_JSONL_WINDOWS')
    finally:stream.close()
    return dict(status='PASS_INDEPENDENT_NPZ_JSONL_RGB_READER',full_windows=count,RGB_references=rgb,padded_windows=0,rejected_student_commands_in_labels=0)


class CorrectionDataset(NativeDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.manifest['schema']!='E1_SOURCE_CORRECTION_V1':raise ValueError('EXACT_E1_CORRECTION_SCHEMA')
        self.rows=[(row,a) for row in self.manifest['episodes'] for a in row['legal_anchors']]
        if any(row['split']!='TRAIN' or row['task'] not in self.manifest['qualified_tasks'] for row,_ in self.rows):raise ValueError('QUALIFIED_TRAIN_CORRECTION_ONLY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--statistics',required=True);a=p.parse_args();print(json.dumps(export_recovery(a.root,a.statistics)),flush=True)
