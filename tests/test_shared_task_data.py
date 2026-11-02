"""Data-version copy tests; the fixture is not actual Processor qualification."""
import hashlib
import json
from pathlib import Path

import pytest
from scripts.pi0294.multitask import shared_task_data as data


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n');return data.sha(path)


def source(tmp_path):
    from test_shared_task_sampling import dataset
    root=tmp_path/'source';root.mkdir();(root/'episodes').mkdir();entries={r['id']:dict(r)for r,a in dataset().rows}
    for row in entries.values():
        body=row['id'].encode();path=root/'episodes'/(row['id']+'.npz');path.write_bytes(body)
        row.update(file=str(path.relative_to(root)),sha256=hashlib.sha256(body).hexdigest(),raw_root='/already/qualified/raw')
    stats=write(root/'statistics.json',{})
    m=dict(schema='SOURCE_COLLECTION_EXPANDED_FOUR_TASK_TRAIN_CANDIDATE_V1',training_admitted=False,
        DEV_windows=0,TRAIN_windows=34563,statistics_sha256=stats,episodes=list(entries.values()))
    mp=write(root/'DATASET_MANIFEST.json',m)
    q=dict(status='PASS_OFFICIAL_TRAIN_CODEC_ALL_WINDOWS_AND_REAL_RGB',manifest_sha256=mp,
        statistics_sha256=stats,windows=34563,torch_CUDA_initialized=False,
        episodes=[dict(id=r['id'],task=r['task'],windows=r['windows'])for r in entries.values()])
    qp=write(root/'qualification.json',q);dev=write(root/'DEV.json',dict(DEV='preserved'))
    return root,dict(source_manifest_sha256=mp,qualification_path=root/'qualification.json',
        qualification_sha256=qp,task_DEV_path=root/'DEV.json',task_DEV_sha256=dev)


def test_freeze_preserves_all_actual_input_bytes_and_candidate(tmp_path):
    root,pins=source(tmp_path);before=(root/'DATASET_MANIFEST.json').read_bytes();out=tmp_path/'new';r=data.freeze(root,out,**pins)
    assert r['entries']==59 and r['full_expert_episodes']==46 and r['expert_suffix_entries']==13
    assert r['model_and_GPU_training_admitted']is False
    m=json.loads((out/'DATASET_MANIFEST.json').read_bytes());assert m['training_admitted']is True
    assert(root/'DATASET_MANIFEST.json').read_bytes()==before
    assert(out/'statistics.json').read_bytes()==(root/'statistics.json').read_bytes()
    for row in m['episodes']:assert data.sha(out/row['file'])==row['sha256']
    with pytest.raises(PermissionError):data.freeze(root,out,**pins)


@pytest.mark.parametrize('fault',['candidate','qualification','statistics','DEV','numeric'])
def test_changed_or_unqualified_source_never_becomes_an_admitted_dataset(tmp_path,fault):
    root,pins=source(tmp_path)
    targets={'candidate':'DATASET_MANIFEST.json','qualification':'qualification.json',
        'statistics':'statistics.json','DEV':'DEV.json','numeric':'episodes/T1_000.npz'}
    (root/targets[fault]).write_text('changed')
    with pytest.raises(PermissionError):data.freeze(root,tmp_path/'new',**pins)
    assert not(tmp_path/'new/DATASET_MANIFEST.json').exists()
