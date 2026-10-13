"""Build a new bank by referencing preserved first attempts across two authorities."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

from .durable import read,write
from .train_adapter import sha

OLD_HF_REVISION='c25eeadf9be0aa7d46bb6f7b65e260cbf3a305ea'
OLD_STATS='43469f3818119c075774d91c5b08bbc2a46a0323a572a8e58614daa48155fd2d'


def build(root,old):
    root,old=Path(root),Path(old)
    if (root/'BANK_V2_ADMISSION.json').exists():raise FileExistsError('Preserve first bank admission')
    state=read(root/'evidence/DISPATCH_supplement.json')
    if state['status']!='TERMINAL_SUCCESS':raise ValueError('ALL_24_TERMINAL_REQUIRED')
    plan=read(root/'ATTEMPT_PLAN.json')
    planned=plan['order'][2:]
    if [r['id'] for r in state['records']]!=planned:raise ValueError('SUPPLEMENT_IDENTITY_ORDER_CHANGED')
    if sha(old/'native_dataset/statistics.json')!=OLD_STATS:raise ValueError('ORIGINAL_STATS_CHANGED')
    original=read(old/'DATASET_ADMISSION.json')
    byid={r['id']:r for r in original['episodes']}
    records=[]
    for identifier in [f'T291{i:03d}' for i in range(64)]+[f'D291{i:03d}' for i in range(16)]:
        prior=byid[identifier]
        if identifier not in planned:
            if not prior['admitted'] or prior['split']!='TRAIN':raise ValueError('ORIGINAL_56_NOT_ADMITTED')
            folder=old/'episodes'/identifier
            row={**copy.deepcopy(prior),'origin_authority':'PI0291','raw_root':str(folder),
                'old_HF_revision':OLD_HF_REVISION,'recollected':False}
            if sha(folder/'samples.json')!=prior['samples_sha256']:raise ValueError('ORIGINAL_SAMPLE_CHANGED')
        else:
            if prior['admitted']:raise ValueError('SUPPLEMENT_WAS_ALREADY_ADMITTED')
            folder=root/'episodes'/identifier
            card=read(folder/'CARD.json');audit=read(root/'evidence'/('AUDIT_BOUND_'+identifier+'.json'))['episodes'][0]
            first=next(r for r in state['records'] if r['id']==identifier)
            bound=plan['attempts'][identifier]
            if not isinstance(card['seed'],int) or card['seed']!=bound['card']['seed'] or card['id']!=identifier:
                raise ValueError('EXACT_ORIGINAL_SEED_OR_ID_CHANGED')
            if card['group_id']!=bound['card']['group_id']:raise ValueError('GROUP_CHANGED')
            if audit['samples_sha256']!=sha(folder/'samples.json'):raise ValueError('NEW_SAMPLE_CHANGED_AFTER_AUDIT')
            admitted=bool(first['admitted'] and audit['status']=='PASS_FULL_RAW_AUDIT' and audit['independent_score']['persistent_success'])
            row=dict(id=identifier,seed=card['seed'],group=card['group_id'],split=card['split'],admitted=admitted,
                samples_sha256=audit['samples_sha256'],raw_root=str(folder),origin_authority='PI0292',origin_plan='PI0291',
                attempt_uid=root.name+'/'+identifier,terminal='ADMITTED' if admitted else 'REJECTED_PRESERVED',recollected=False)
        records.append(row)
    counts={split:dict(planned=64 if split=='TRAIN' else 16,admitted=sum(r['admitted'] and r['split']==split for r in records)) for split in ('TRAIN','DEV')}
    qualified=counts['TRAIN']['admitted']>=56 and counts['DEV']['admitted']>=14
    admission=dict(status='ARX5_BANK_V2_QUALIFIED' if qualified else 'BANK_V2_QUALITY_GATE_FAIL',
        episodes=records,counts=counts,original_PI0291_status='PARTIAL',old_ledger_overwritten=False,
        original_admission_sha256=sha(old/'DATASET_ADMISSION.json'),old_statistics_sha256=OLD_STATS,
        successor_attempts=planned,all_24_terminal=True,canaries_excluded=True)
    write(root/'BANK_V2_ADMISSION.json',admission)
    if not qualified:return admission
    view=root/'bank_v2_input';(view/'episodes').mkdir(parents=True)
    for row in records:
        if row['admitted']:(view/'episodes'/row['id']).symlink_to(Path(row['raw_root']))
    sys.path.insert(0,str(root/'code/scripts/pi0291_arx5_data'))
    from export_dataset import export
    manifest=export(view,root/'BANK_V2_ADMISSION.json',root/'bank_v2_native')
    write(root/'BANK_V2_MANIFEST.json',dict(status='EXPORTED_NATIVE_QUALIFICATION_PENDING',
        manifest_sha256=sha(root/'bank_v2_native/DATASET_MANIFEST.json'),statistics_sha256=manifest['statistics_sha256'],
        TRAIN_windows=manifest['TRAIN_windows'],DEV_windows=manifest['DEV_windows'],episodes=manifest['episodes'],
        original_PI0291_status='PARTIAL',source_authorities=['PI0291','PI0292']))
    prior=read(old/'native_dataset/statistics.json');current=read(root/'bank_v2_native/statistics.json')
    import numpy as np
    diff={}
    for mode,features in current['new_embodiment'].items():
        for key,metrics in features.items():
            for statistic,value in metrics.items():
                diff[mode+'/'+key+'/'+statistic]=float(np.max(np.abs(np.asarray(value)-np.asarray(prior['new_embodiment'][mode][key][statistic]))))
    write(root/'STATISTICS_PROVENANCE.json',dict(new=read(root/'bank_v2_native/STATISTICS_PROVENANCE.json'),
        old_sha256=OLD_STATS,new_sha256=sha(root/'bank_v2_native/statistics.json'),max_abs_differences=diff,
        fit_count=1,DEV_fit=False,old_file_unchanged=sha(old/'native_dataset/statistics.json')==OLD_STATS))
    return admission


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--old',required=True);a=p.parse_args()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    result=build(a.root,a.old)
    u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
    write(Path(a.root)/'evidence/BANK_BUILD_CPU.json',dict(status=result['status'],counts=result['counts'],
        CPU_s=u.ru_utime+u.ru_stime+c.ru_utime+c.ru_stime,model_calls=0,physics_steps=0))
    print(json.dumps(dict(status=result['status'],counts=result['counts'])))
