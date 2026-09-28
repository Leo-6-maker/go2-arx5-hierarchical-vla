#!/usr/bin/env python3
"""Read only original TRAIN numeric rows and sealed bytes. No video decoding."""
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path('/scratch/e1192548/Projects/go2-arx5')
OUT=ROOT/'experiments/PI0253/evidence'
DATA=Path('/scratch/e1192548/pi0227_large_adapt_v1_inputs/train_dataset').resolve()
SEAL=ROOT/'experiments/PI0237/pi0237_failure_attribution_20260920T014727Z/evidence/pi0228_train_calib_seal.json'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    seal=json.loads(SEAL.read_text())
    assert seal['status']=='PASS' and all(seal['checks'].values())
    assert seal['train']['source_rollout_count']==612 and seal['train']['episode_rows']==9792
    spec=importlib.util.spec_from_file_location('original_tree',Path('/scratch/e1192548/pi0244_fixed16_10k_v1/scripts/pi0228_h200_large_adapt_v1.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    tree,files=helper.tree_sha256(DATA,'train')
    assert tree==seal['train']['tree_sha256']==helper.EXPECTED_TRAIN_TREE_SHA
    episodes=[json.loads(x) for x in (DATA/'meta/episodes.jsonl').read_text().splitlines() if x]
    prov=[json.loads(x) for x in (DATA/'meta/stage19_episode_provenance.jsonl').read_text().splitlines() if x]
    qualified=json.loads((DATA/'meta/stage19_qualified_manifest.json').read_text())
    assert qualified['desired_labels_only'] and not qualified['achieved_labels_used_as_training_target']
    assert len(episodes)==len(prov)==9792
    states=[];starts=[];actions0=[];actions20=[];action_ends=[];ids=[];families=[]
    for i,(ep,p) in enumerate(zip(episodes,prov)):
        assert ep['episode_index']==i and ep['source_episode_id']==p['source_episode_id'] and ep['split']=='train'
        assert p['desired_label'] and p['source_frame']=='NATIVE_WBC_RELATIVE_REFERENCE' and p['tcp_semantic_id']=='x5_link6_gripper_center'
        rel=f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet'
        d=pq.read_table(DATA/rel,columns=['timestamp','observation.state','action']).to_pydict()
        s=np.asarray(d['observation.state'],dtype=np.float32);a=np.asarray(d['action'],dtype=np.float32)
        t=np.asarray(d['timestamp'],dtype=np.float64)
        assert s.shape==a.shape==(40,10) and t.shape==(40,) and np.isfinite(s).all() and np.isfinite(a).all()
        assert np.allclose(t,p['sample_timestamps'],atol=1e-9,rtol=0) and np.allclose(np.diff(t),.05,atol=1e-9,rtol=0)
        states.append(s[0]); starts.append(t[0]); actions0.append(a[0]);actions20.append(a[20]);action_ends.append(a[-1])
        ids.append(ep['source_episode_id']);families.append(re.search(r'persistent_train_([a-d])_',ids[-1])[1])
        if i%2000==0:print(json.dumps(dict(status='TRAIN_NUMERIC_AUDIT',completed=i)),flush=True)
    np.savez_compressed(OUT/'full_train_numeric_support.npz',state=np.stack(states),start_time=np.asarray(starts),
                        action_first=np.stack(actions0),action_at_1s=np.stack(actions20),action_last=np.stack(action_ends),source_episode_id=np.asarray(ids))
    counts={}
    for family in 'abcd':
        fi=[i for i,v in enumerate(families) if v==family]
        nominal=[i for i in fi if '_j00_nominal_' in ids[i]]
        reset=[i for i in fi if ids[i].endswith(':obs-0000')]
        early=[i for i in fi if starts[i]<=.4+1e-8]
        counts[family]=dict(windows=len(fi),rollouts=len({ids[i].split(':obs-')[0] for i in fi}),
            reset_windows=len(reset),reset_rollouts=len({ids[i].split(':obs-')[0] for i in reset}),
            early_le_0p4s_windows=len(early),early_rollouts=len({ids[i].split(':obs-')[0] for i in early}),
            nominal_windows=len(nominal),nominal_reset_windows=sum(ids[i].endswith(':obs-0000') for i in nominal))
    report=dict(schema='PI0253_ORIGINAL_FULL_TRAIN_SUPPORT_V1',status='PASS_SEAL_AND_NUMERIC_SUPPORT',
        train_tree_sha256=tree,files=len(files),bytes=sum(f['bytes'] for f in files),
        original_seal_sha256=helper.sha256_file(SEAL),qualified_manifest_sha256=helper.sha256_file(DATA/'meta/stage19_qualified_manifest.json'),
        original_train_coverage=seal['train']['coverage'],counts=counts,
        start_time_histogram={str(k):v for k,v in sorted(Counter(starts).items())},
        label=dict(kind='DESIRED_EEF',frame='NATIVE_WBC_RELATIVE_REFERENCE',tcp='x5_link6_gripper_center',horizon=40,dt_s=.05,achieved_target=False),
        numeric_artifact_sha256=helper.sha256_file(OUT/'full_train_numeric_support.npz'),
        model_queries=0,optimizer_updates=0,video_decodes=0,calib_dataset_reads=0,dev_f_reads=0)
    (OUT/'full_support.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['start_time_histogram','original_train_coverage']}),flush=True)


if __name__=='__main__':main()
