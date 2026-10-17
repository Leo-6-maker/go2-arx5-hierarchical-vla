"""Actual consumed sample exposure; correction gain is not a matched-task claim."""
from collections import Counter
import argparse
import json
from pathlib import Path
import time


def summarize(records,consumed,branch):
    if branch not in ['M2-BC','M2-CORR'] or type(consumed) is not int or consumed<=0 or consumed%32:
        raise ValueError('EXACT_MATCHED_E1_BRANCH_AND_OPTIMIZER_SAMPLE_BOUNDARY')
    selected={}
    for row in records:
        if type(row['cursor']) is not int or row['cursor']<0:raise ValueError('EXACT_TRACE_CURSOR')
        if row['cursor']>=consumed:continue
        if row['cursor'] in selected:raise ValueError('DUPLICATE_CONSUMED_SAMPLE_CURSOR')
        if row['split']!='TRAIN' or row['task'] not in ['T1','T2'] or row['origin'] not in ['legacy','new','E0_STATE_AGGREGATION_ONLY']:
            raise PermissionError('E1_ORIGINAL_TRAIN_OR_FIXED_C0_ONLY')
        if row['origin']=='E0_STATE_AGGREGATION_ONLY' and row['task']!='T1':
            raise PermissionError('FIXED_C0_HAS_QUALIFIED_T1_ONLY')
        if not isinstance(row['episode_id'],str) or not row['episode_id']:
            raise ValueError('EXACT_TRACE_EPISODE_ID')
        selected[row['cursor']]=row
    if set(selected)!=set(range(consumed)):raise ValueError('INCOMPLETE_ACTUAL_CONSUMED_TRACE')
    counts=Counter();ids={};correction_segments=set()
    for cursor in range(consumed):
        row=selected[cursor];origin='C0' if row['origin']=='E0_STATE_AGGREGATION_ONLY' else 'original'
        counts[row['task']]+=1;counts[origin]+=1;ids.setdefault(origin,set()).add(row['episode_id'])
        if origin=='C0':correction_segments.add(row['episode_id'])
    expected=dict(T1=consumed//2,T2=consumed//2,original=consumed,C0=0) if branch=='M2-BC' else dict(T1=consumed*3//4,T2=consumed//4,original=consumed//2,C0=consumed//2)
    for key,value in expected.items():
        if counts[key]!=value:raise ValueError('FIXED_E1_ACTUAL_EXPOSURE_DIFFERS_FROM_SEALED_SAMPLER:'+key)
    for start in range(0,consumed,32):
        batch=[selected[c] for c in range(start,start+32)]
        if sum(r['task']=='T1' for r in batch)!=(16 if branch=='M2-BC' else 24):raise ValueError('ACTUAL_EFFECTIVE_BATCH_TASK_EXPOSURE')
        if sum(r['origin']=='E0_STATE_AGGREGATION_ONLY' for r in batch)!=(0 if branch=='M2-BC' else 16):raise ValueError('ACTUAL_EFFECTIVE_BATCH_CORRECTION_EXPOSURE')
    return dict(branch=branch,actual_consumed_samples=consumed,optimizer_updates_represented=consumed//32,
        task_samples={t:counts[t] for t in ['T1','T2']},origin_samples={o:counts[o] for o in ['original','C0']},
        original_episodes_observed=len(ids.get('original',set())),correction_segments_observed=len(correction_segments),
        independent_correction_parent_episode_count='READ_FIXED_C0_MANIFEST_NOT_SEGMENT_COUNT',
        all_effective_batches_checked=True,new_M1_online_C1_samples=0,
        matched_total_updates_not_matched_task_exposure=branch=='M2-CORR',
        causal_correction_gain_claim_supported=False,recipe_or_sampler_modified=False)


def run(root):
    from scripts.pi0292.durable import read,write
    from scripts.pi0294.native_dataset import sha
    from .scope import validate,Ledger
    root=Path(root);validate(root,'hopper-neural');ledger=Ledger(root,'hopper');ledger.guard(closeout=True)
    state=read(ledger.path)
    if state['GPU_reserved_s'] or any(r['status']!='NATIVE_TERMINAL' for r in state['PBS_submissions']):
        raise PermissionError('ACTUAL_CONSUMPTION_REPORT_AFTER_NATIVE_GPU_SETTLEMENT')
    started=time.process_time();reports={};provenance={}
    try:
        for branch in ['M2-BC','M2-CORR']:
            folder=root/'training'/branch;receipts=list(folder.glob('session-*/TRAINING_RUN.json'))
            if not receipts:raise ValueError('NO_TERMINAL_NATIVE_TRAINING_RECEIPT:'+branch)
            terminal=max(receipts,key=lambda p:read(p)['global_step']);training=read(terminal)
            if training['global_step']!=512 or training['consumed_samples']!=16384:
                raise ValueError('EXACT_FIXED_512_TERMINAL_CONSUMPTION_REQUIRED')
            provenance[str(terminal.relative_to(root))]=sha(terminal);records=[]
            for path in sorted(folder.glob('session-*/worker_trace/worker-*.jsonl')):
                provenance[str(path.relative_to(root))]=sha(path)
                records.extend(json.loads(line) for line in path.read_text().splitlines())
            reports[branch]=summarize(records,training['consumed_samples'],branch)
            reports[branch]['correction_manifest_sha256']=training['correction_manifest_sha256']
        binding=read(root/'plans/CORRECTION_BINDING.private.json')
        manifest=Path(binding['dataset'])/'DATASET_MANIFEST.json';c0=read(manifest)
        if sha(manifest)!=reports['M2-CORR']['correction_manifest_sha256']:
            raise ValueError('ACTUAL_TRAINED_C0_MANIFEST_CHANGED')
        value=dict(schema='E1_ACTUAL_CONSUMED_EXPOSURE_V1',branches=reports,
            C0_manifest_sha256=sha(manifest),C0_independent_student_parent_episodes=len({r['source_student_episode'] for r in c0['episodes']}),
            C0_segments=len(c0['episodes']),evidence_sha256=provenance,
            comparison_confound='BC_T1_T2_1_TO_1_VS_CORR_3_TO_1_NOT_TASK_MATCHED',
            training_recipe_unchanged=True,C1_used=False,causal_correction_gain_claim_supported=False)
        write(root/'evidence/ACTUAL_CONSUMED_EXPOSURE.json',value);return value
    finally:ledger.cpu(time.process_time()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(run(a.root)))
