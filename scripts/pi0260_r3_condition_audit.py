"""One cached, model-free pass over sealed R2 conditions and original H40 labels."""
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pi0260_r2 import ROOT, read, sha
from pi0260_r2_inputs import batch, build, identity
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix

R3 = ROOT / 'R3'
COSMOS = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/inputs/frozen_v1/cosmos_snapshot')
AUTHORITY = 'fd8e69b59a16aa90b1761d23b3a8f5a78cee8763'


def raw_key(body, wrist, state, language):
    h = hashlib.sha256()
    for value in (body.tobytes(), wrist.tobytes(), state.astype(np.float32).tobytes(), language.encode()):
        h.update(len(value).to_bytes(8, 'big'))
        h.update(value)
    return h.hexdigest()


def rotation_distance(a, b):
    x = rot6d_rows_to_matrix(a[:, 3:9])
    y = rot6d_rows_to_matrix(b[:, 3:9])
    return Rotation.from_matrix(x @ y.transpose(0, 2, 1)).magnitude()


def summarize(items):
    groups = defaultdict(list)
    for x in items:
        groups[x['key']].append(x)
    summary = dict(windows=len(items), conditions=len(groups), group_sizes=dict(sorted(Counter(map(len, groups.values())).items())),
                   aliased_groups=sum(len(g)>1 for g in groups.values()),
                   aliased_windows=sum(len(g) for g in groups.values() if len(g)>1),
                   processed_merges=sum(len({r['raw_key'] for r in g})>1 for g in groups.values()),
                   mixed_semantics=[], max_position_rmse_m=0., max_pointwise_m=0., max_SO3_rad=0.,
                   empirical_mse_m2=0., empirical_rmse_m=0., by_task={}, aliases=[])
    mse_sum = 0.
    task_groups = defaultdict(list)
    for key, rows in groups.items():
        semantics = {(r['task'], r['command'], r['language']) for r in rows}
        if len(semantics)>1:
            summary['mixed_semantics'].append(dict(key=key, semantic_labels=sorted(map(str, semantics)),
                                                   sources=[r['source'] for r in rows]))
        positions = np.stack([r['label'][:, :3].astype(np.float64) for r in rows])
        mean = positions.mean(axis=0)  # ponytail: diagnostic only; never written as a training label.
        v = float(np.mean(np.sum((positions-mean)**2, axis=2)))
        mse_sum += len(rows)*v
        if len(rows)>1:
            summary['aliases'].append(dict(key=key,size=len(rows),sources=[r['source'] for r in rows],
                                           times_s=[r['time'] for r in rows],
                                           semantics=sorted(map(str, semantics)),empirical_mse_m2=v,
                                           raw_conditions=len({r['raw_key'] for r in rows})))
        for i in range(len(rows)):
            for j in range(i):
                distance = np.linalg.norm(positions[i]-positions[j], axis=1)
                summary['max_position_rmse_m'] = max(summary['max_position_rmse_m'], float(np.sqrt(np.mean(distance**2))))
                summary['max_pointwise_m'] = max(summary['max_pointwise_m'], float(distance.max()))
                summary['max_SO3_rad'] = max(summary['max_SO3_rad'], float(rotation_distance(rows[i]['label'], rows[j]['label']).max()))
        for task in {r['task'] for r in rows}:
            subset = [r for r in rows if r['task']==task]
            task_groups[task].extend(subset)
    summary['empirical_mse_m2'] = mse_sum/len(items)
    summary['empirical_rmse_m'] = float(np.sqrt(summary['empirical_mse_m2']))
    if len(task_groups)>1:
        summary['by_task'] = {task: summarize(rows)[0] for task, rows in task_groups.items()}
    return summary, groups


def main():
    R3.mkdir(exist_ok=True)
    codec = read(ROOT/'dataset/codec_receipt.json')
    assert codec['status']=='COMPACT_30_10' and sha(ROOT/'dataset/codec_receipt.json')=='08c9d0435a59f0f0617d648301f85ea909c2ed66d345570848402e197d7ef713'
    assert read(ROOT/'archive/R2_PUBLIC_RECEIPT.json')['status']=='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK'
    alias = read(ROOT/'evidence/actual_processor_aliases.json')
    assert len(alias['pairs'])==40
    processor = build(read(ROOT/'dataset/statistics.json'), COSMOS)
    db = sqlite3.connect(R3/'condition_cache.sqlite')
    db.execute('create table if not exists processed (raw_key text primary key, processed_key text not null)')
    cache = dict(db.execute('select raw_key, processed_key from processed'))
    all_rows=[];new=0
    for family in codec['families']:
        assert family['qualified']
        for entry in family['sequences']:
            name=entry['sequence'];record=read(ROOT/'dataset'/(name+'.json'))
            with np.load(ROOT/'dataset'/(name+'.npz')) as z:
                states=z['state_v1'].copy();labels=z['raw_action'].copy()
            for j,row in enumerate(record['rows']):
                if not(row['active'] and row['usable']):continue
                with np.load(ROOT/row['frame']) as z:
                    body=z['body_rgb'].copy();wrist=z['wrist_rgb'].copy()
                language=record['sequence']['instruction']
                raw=raw_key(body,wrist,states[j],language)
                key=cache.get(raw)
                if key is None:
                    packet=dict(body_rgb=body,wrist_rgb=wrist,state_v1=states[j],language=language)
                    key=identity(batch(processor,packet)['inputs'])
                    db.execute('insert into processed values (?,?)',(raw,key));cache[raw]=key;new+=1
                    if new%100==0:
                        db.commit();print(json.dumps(dict(processed_new=new,total_cache=len(cache))),flush=True)
                all_rows.append(dict(key=key,raw_key=raw,split=family['split'],task=record['sequence']['task'],
                    command=record['sequence']['command'],language=language,time=float(row['time']),
                    source=name+':'+str(row['index']),label=labels[j]))
    db.commit();db.close()
    assert len(all_rows)==17377
    partitions={s:[r for r in all_rows if r['split']==s] for s in ('TRAIN','KIN-VAL')}
    result=dict(status='R3_CONDITION_AUDIT_COMPLETE',authority=AUTHORITY,codec_sha256=sha(ROOT/'dataset/codec_receipt.json'),
                actual_processor_source_sha256=sha(ROOT/'runtime/pi0260_r2_inputs.py'),
                processed_new=new,unique_raw_conditions=len(cache),windows=len(all_rows),
                partitions={},cross_split_conditions=0,full_observability_proven=False,
                floor_interpretation='Uniform-window empirical deterministic position MSE floor only; not a median, p95, model error or training label.',
                model_loads=0,model_queries=0,optimizer_updates=0)
    for split, rows in partitions.items():
        result['partitions'][split],_ = summarize(rows)
    train_keys={r['key'] for r in partitions['TRAIN']}
    result['cross_split_conditions']=len(train_keys & {r['key'] for r in partitions['KIN-VAL']})
    output=R3/'condition_audit.json'
    assert not output.exists()
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status=result['status'],windows=len(all_rows),conditions={s:result['partitions'][s]['conditions'] for s in partitions},
                          aliases={s:result['partitions'][s]['aliased_groups'] for s in partitions},
                          empirical_rmse_m={s:result['partitions'][s]['empirical_rmse_m'] for s in partitions},
                          cross_split_conditions=result['cross_split_conditions'],sha256=sha(output))),flush=True)


if __name__=='__main__':
    main()
