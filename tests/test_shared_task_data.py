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
    dev=dict(T1_T2_existing_original_DEV_cards=[dict(id=f'{t}_DEV{i:03}',task=t,split='DEV')for t in('T1','T2')for i in range(24)],
        cabinet_DEV_candidate_cards=[dict(id=f'DOOR_DEV{i:03}',task='DOOR',study_split='DEV')for i in range(10)],
        DEV_prior_exposure='EXPOSED_DEV_NOT_TEST')
    from test_book_seed_set import seal
    bp=write(root/'BOOK_DEV.json',seal())
    qp=write(root/'qualification.json',q);dev=write(root/'DEV.json',dev)
    return root,dict(source_manifest_sha256=mp,qualification_path=root/'qualification.json',
        qualification_sha256=qp,task_DEV_path=root/'DEV.json',task_DEV_sha256=dev,
        book_DEV_path=root/'BOOK_DEV.json',book_DEV_sha256=bp)


def test_freeze_preserves_all_actual_input_bytes_and_candidate(tmp_path):
    root,pins=source(tmp_path);before=(root/'DATASET_MANIFEST.json').read_bytes();out=tmp_path/'new';r=data.freeze(root,out,**pins)
    assert r['entries']==59 and r['full_expert_episodes']==46 and r['expert_suffix_entries']==13
    assert r['model_and_GPU_training_admitted']is False
    assert r['task_DEV_counts']==dict(T1=24,T2=24,BOOK=10,DOOR=10)
    m=json.loads((out/'DATASET_MANIFEST.json').read_bytes());assert m['training_admitted']is True
    assert(root/'DATASET_MANIFEST.json').read_bytes()==before
    assert(out/'statistics.json').read_bytes()==(root/'statistics.json').read_bytes()
    for row in m['episodes']:assert data.sha(out/row['file'])==row['sha256']
    with pytest.raises(PermissionError):data.freeze(root,out,**pins)


def test_incomplete_or_changed_book_dev_binding_is_rejected_before_copy(tmp_path):
    root,pins=source(tmp_path);book=json.loads((root/'BOOK_DEV.json').read_bytes());book['DEV'].pop()
    pins['book_DEV_sha256']=write(root/'BOOK_DEV.json',book)
    with pytest.raises(ValueError):data.freeze(root,tmp_path/'new',**pins)
    assert not(tmp_path/'new').exists()


def test_original_task_dev_conditions_cannot_be_silently_dropped(tmp_path):
    root,pins=source(tmp_path);dev=json.loads((root/'DEV.json').read_bytes());dev['T1_T2_existing_original_DEV_cards'].pop()
    pins['task_DEV_sha256']=write(root/'DEV.json',dev)
    with pytest.raises(PermissionError):data.freeze(root,tmp_path/'new',**pins)
    assert not(tmp_path/'new').exists()


@pytest.mark.parametrize('fault',['candidate','qualification','statistics','DEV','numeric'])
def test_changed_or_unqualified_source_never_becomes_an_admitted_dataset(tmp_path,fault):
    root,pins=source(tmp_path)
    targets={'candidate':'DATASET_MANIFEST.json','qualification':'qualification.json',
        'statistics':'statistics.json','DEV':'DEV.json','numeric':'episodes/T1_000.npz'}
    (root/targets[fault]).write_text('changed')
    with pytest.raises(PermissionError):data.freeze(root,tmp_path/'new',**pins)
    assert not(tmp_path/'new/DATASET_MANIFEST.json').exists()
