"""One frozen bank update from actual one-way M1 intervention commands."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from go2arx5_vla.task_data.codec import pose10
from .artifacts import atomic_npz
from .correction import window_key,independent_read
from .scope import validate,Ledger,STATS


def check_ack_progress(arrays,ack):
    rows=arrays['rows'];before=int(arrays['tick_before']);after=int(arrays['tick_after'])
    # The final high-rate column is a safety flag, not an integration counter.
    if (before!=ack['tick_before'] or after!=ack['tick_after'] or after-before!=200
            or rows.ndim!=2 or rows.shape[0]!=200 or not np.isfinite(rows).all()
            or not np.allclose(rows[:,0],np.arange(before+1,after+1)*.002,atol=1e-8,rtol=0)):
        raise ValueError('E1_COLLECTION_ACK_NATIVE_PROGRESS')


def export(root,attempt=1):
    root=Path(root);runtime=validate(root,'4080-simulation',physical=True);ledger=Ledger(root,'core');ledger.guard();start=time.process_time()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    if not 1<=attempt<=3:raise PermissionError('E1_BOUNDED_SAVED_COLLECTION_EXPORT_ATTEMPTS')
    suffix='' if attempt==1 else '-REPAIR-'+str(attempt).zfill(2)
    original=root/'datasets/E1_SOURCE_CORRECTION_V1-C0';out=root/('datasets/E1_SOURCE_CORRECTION_V1-C1'+suffix)
    if out.exists():raise FileExistsError('PRESERVE_FIRST_ACTUAL_COLLECTION_EXPORT')
    original_card=read(original/'CORRECTION_DATASET_CARD.json');manifest=read(original/'DATASET_MANIFEST.json')
    if original_card['manifest_sha256']!=sha(original/'DATASET_MANIFEST.json'):raise ValueError('E1_C0_FROZEN_MANIFEST_CHANGED')
    registry=read(root/'plans/D-M1-1024/REGISTRY.json');qualified=set(manifest['qualified_tasks'])
    if len(registry['cases'])!=4 or {c['card']['task'] for c in registry['cases'].values()}!={'T1'} or qualified!={'T1'}:
        raise PermissionError('E1_CURRENT_T1_ONLY_FIXED_COLLECTION_PANEL')
    records=[];rows=[];seen=set();duplicates=0
    try:
        shutil.copytree(original,out);rows=list(manifest['episodes'])
        for row in rows:
            with np.load(out/row['file'],allow_pickle=False) as z:
                for a in row['legal_anchors']:seen.add(window_key(z['actual_state10'][a],z['committed_absolute10'][a:a+40],z['rgb_sha256'][a]))
        for case in sorted(registry['cases'].values(),key=lambda c:c['panel_index']):
            ledger.guard();folder=root/'episodes'/case['attempt_id'];receipt=read(folder/'INTERVENTION_RESULT.json');adjudication=read(folder/'E1_ADJUDICATION.json')
            record=dict(attempt_id=case['attempt_id'],task=case['card']['task'],student_episode=case['episode_uuid'],
                student_prefixes=receipt['student_executed_prefixes'],expert_prefixes=receipt['expert_executed_prefixes'],
                data_class=receipt['data_class'],valid_H40_windows=0,assisted_task_success=adjudication.get('assisted_task_success',False))
            records.append(record)
            if receipt['expert_executed_commands']<40:continue
            if adjudication['live_native_failures'] or adjudication['maximum_live_shadow_difference']>1e-10:raise ValueError('E1_COLLECTION_NATIVE_SAFETY_OR_PARITY_INVALID')
            acks=read(folder/'EXPERT_PREFIX_ACKS.json');begin=receipt['first_intervention_tick'];end=begin
            for i,ack in enumerate(acks):
                p=folder/('expert-prefix-'+str(i).zfill(3)+'.npz')
                if ack['tick_before']!=end or ack['tick_after']!=end+200 or sha(p)!=ack['physical_receipt_sha256']:raise ValueError('E1_COLLECTION_ACTUAL_ACK_CHAIN')
                with np.load(p,allow_pickle=False) as z:
                    check_ack_progress(z,ack)
                end+=200
            samples=read(folder/'samples.json')[begin//25:end//25]
            if len(samples)!=receipt['expert_executed_commands']:raise ValueError('E1_COLLECTION_SAMPLE_COMMAND_COUNT')
            states=[];commands=[];paths=[];hashes=[]
            for sample in samples:
                obs=sample['policy_obs'];label=sample['supervision']
                if not label or not label['actually_committed'] or not label['valid']:raise ValueError('E1_COLLECTION_POSITIVE_LABEL_MUST_BE_EXECUTED')
                states.append(pose10(np.asarray(obs['actual_tcp']),obs['actual_gripper']))
                commands.append(pose10(np.asarray(label['committed']),label['committed_gripper']))
                paths.append([obs[c]['path'] for c in ['body_rgb','wrist_rgb']]);hashes.append([obs[c]['sha256'] for c in ['body_rgb','wrist_rgb']])
            states,commands=np.asarray(states),np.asarray(commands);legal=[]
            for a in range(max(0,len(commands)-39)):
                key=window_key(states[a],commands[a:a+40],hashes[a])
                if key in seen:duplicates+=1;continue
                seen.add(key);legal.append(a)
            if not legal:continue
            raw=out/'raw'/case['attempt_id'];raw.mkdir()
            for names,digests in zip(paths,hashes):
                for name,digest in zip(names,digests):
                    if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('COLLECTION_RELATIVE_RGB_PATH_REQUIRED')
                    p=folder/name
                    if not p.resolve().is_relative_to(folder.resolve()) or sha(p)!=digest:raise ValueError('E1_COLLECTION_ACTUAL_RGB_CHANGED')
                    target=raw/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);target.chmod(0o600)
            p=out/'numeric'/(case['attempt_id']+'.npz')
            atomic_npz(p,actual_state10=states,committed_absolute10=commands,valid_window=np.isin(np.arange(len(commands)),legal),
                rgb_paths=np.asarray(paths),rgb_sha256=np.asarray(hashes),instruction=np.asarray(case['card']['instruction']),label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'))
            row=dict(id=case['attempt_id'],task=case['card']['task'],family=case['card']['family'],split='TRAIN',
                origin='E1_STUDENT_VISITED_INTERVENTION' if receipt['student_executed_prefixes'] else 'EXPERT_ASSISTED_DATA',
                file=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size,raw_root='EXPLICIT_PRIVATE_BINDING_REQUIRED',
                source_student_episode=case['episode_uuid'],anchor_tick=begin,windows=len(legal),legal_anchors=legal,
                expert_commands=len(commands),discarded_tail_windows=39,labels_sha256=sha(p),actual_ACKs=len(acks),
                full_recovery=end==16000,full_recovery_success=record['assisted_task_success'],
                execution_actor='ORIGINAL_QUALIFIED_SOURCE_EXPERT',privileged_audit_only=True,
                source_result_sha256=sha(folder/'RESULT.json'),student_executed_prefixes=receipt['student_executed_prefixes'])
            rows.append(row);record['valid_H40_windows']=len(legal)
        genuine={t:sum(r['task']==t and r['student_prefixes']>0 and r['valid_H40_windows']>0 for r in records)>=2 for t in qualified}
        new_rows=[r for r in rows if r['id'] not in {x['id'] for x in manifest['episodes']}]
        manifest.update(episodes=rows,origin='E0_STATE_AGGREGATION_PLUS_E1_INTERVENTION_DATA' if new_rows else 'E0_STATE_AGGREGATION_ONLY',
            new_M1_online_DAgger=any(genuine.values()),genuine_DAgger_per_task=genuine,C0_manifest_sha256=original_card['manifest_sha256'],
            runtime_source_sha=runtime['code']['vla_commit'],
            identical_windows_deduplicated=manifest['identical_windows_deduplicated']+duplicates)
        write(out/'DATASET_MANIFEST.json',manifest)
        with (out/'windows.jsonl').open('w') as f:
            for row in rows:
                with np.load(out/row['file'],allow_pickle=False) as z:
                    for anchor in row['legal_anchors']:
                        f.write(json.dumps(dict(episode_id=row['id'],task=row['task'],split='TRAIN',anchor=anchor,
                            source_student_episode=row['source_student_episode'],anchor_tick=row['anchor_tick'],origin=row['origin'],
                            current_state_actor='STUDENT_AT_INTERVENTION_OR_FORK_ANCHOR' if anchor==0 else 'EXPERT_AFTER_INTERVENTION_OR_FORK',
                            execution_actor=row['execution_actor'],state10=z['actual_state10'][anchor].tolist(),committed_absolute_H40=z['committed_absolute10'][anchor:anchor+40].tolist(),
                            numeric_sha256=row['sha256'],rgb_paths=z['rgb_paths'][anchor].tolist(),rgb_sha256=z['rgb_sha256'][anchor].tolist()),allow_nan=False)+'\n')
        bindings={r['id']:str(out/'raw'/r['id']) for r in rows};write(root/('datasets/C1'+suffix+'_RAW_BINDINGS.private.json'),bindings)
        audit=independent_read(out,bindings)
        tasks={t:dict(windows=sum(r['windows'] for r in rows if r['task']==t),distinct_student_episodes=len({r['source_student_episode'] for r in rows if r['task']==t})) for t in qualified}
        card=dict(status='ACTUAL_NONEMPTY_RECOVERY_BANK_INDEPENDENT_READER_PASS',origin=manifest['origin'],
            manifest_sha256=sha(out/'DATASET_MANIFEST.json'),statistics_sha256=STATS,tasks=tasks,rows=len(rows),
            complete_H40_windows=sum(r['windows'] for r in rows),new_collection_rows=len(new_rows),new_collection_windows=sum(r['windows'] for r in new_rows),
            new_online_DAgger=manifest['new_M1_online_DAgger'],genuine_DAgger_per_task=genuine,collection_attempts=records,
            original_data_changed=False,target_WBC_demo=False,reader=audit,CPU_s=time.process_time()-start,
            saved_export_attempt=attempt,prior_failed_export_preserved=attempt>1,
            late_export_changes_existing_training_binding=False)
        summary=root/('evidence/D_COLLECTION_CAPTURE_SUMMARY'+suffix+'.json')
        if summary.exists():raise FileExistsError('PRESERVE_ACTUAL_COLLECTION_EXPORT_SUMMARY')
        write(out/'CORRECTION_DATASET_CARD.json',card);write(summary,card);return card
    finally:ledger.cpu(time.process_time()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--attempt',type=int,default=1);a=p.parse_args();print(json.dumps(export(a.root,a.attempt)),flush=True)
