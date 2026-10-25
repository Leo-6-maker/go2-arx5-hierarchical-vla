"""Synthetic closed1024 CPU bytes; no native endpoint, HF IO or model claim."""
import base64
from copy import deepcopy
import json

import pytest

from scripts.pi0294.test_v5_closed_training_handoff import fixture
from scripts.pi0294.pv1 import mix_evaluation_transfer as t, evaluation_transfer as old
from scripts.pi0294.pv1 import mix_accounting as a, mix_stage_handoff as handoff


@pytest.fixture
def source(monkeypatch):
    stats=a.raw(dict(synthetic_statistics='CPU_FIXTURE_ONLY'))
    monkeypatch.setattr(a,'STATS',a.sha(stats));monkeypatch.setattr(old,'STATS',a.sha(stats))
    g,_,capture=fixture();settled=json.loads(capture['settled_bytes']);audit=settled['audit']
    tensor='model-00001-of-00001.safetensors'
    files={'config.json':a.raw(dict(model_name='/scratch/fixture/cosmos')),
        'model.safetensors.index.json':a.raw(dict(weight_map=dict(weight=tensor))),
        'statistics.json':stats,'processor/processor_config.json':a.raw(dict(processor_kwargs=dict(model_name='/scratch/fixture/cosmos'))),
        'processor/statistics.json':stats,'processor/embodiment_id.json':a.raw(dict(id=17)),
        tensor:b'opaque-synthetic-weights-not-a-native-model'}
    for name in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin','SAMPLER_STATE.json','extra-a.json','extra-b.json'):
        files[name]=b'CPU_FULL_STATE_FIXTURE:'+name.encode()
    cp=audit['checkpoint_root'];row=g['PBS_submissions'][0]
    complete=dict(global_step=1024,full_resume=True,origin_job_id=row['job_id'],origin_source_sha=row['source_sha'],
        files={n:a.sha(raw)for n,raw in files.items()})
    cr=a.raw(complete);audit['checkpoint_manifest_sha256']=a.sha(cr)
    audit['streamed_witness']=dict(all_paths_rechecked=True,all_unique_inodes_independently_rehashed=True,
        paths=[dict(path=cp+'/'+n,sha256=a.sha(raw),bytes=len(raw))for n,raw in files.items()])
    row['full_checkpoint_audit_sha256']=a.sha(a.raw(audit))
    endpoint=g[a.PACKAGE]['endpoints']['C50'];endpoint.update(checkpoint_manifest_sha256=a.sha(cr),audit_sha256=row['full_checkpoint_audit_sha256'])
    capture['settled_bytes']=a.raw(settled)
    lease=json.loads(capture['lease_bytes']);lease['terminal_receipt_sha256']=a.sha(capture['settled_bytes']);capture['lease_bytes']=a.raw(lease)
    closed=json.loads(capture['closed_bytes']);closed['lease_sha256']=a.sha(capture['lease_bytes']);capture['closed_bytes']=a.raw(closed)
    gb=a.raw(g);proof=handoff.closed_endpoint(global_bytes=gb,arm='C50',**capture)
    full=a.raw(dict(schema='V5_CD_REAL_TRAINING_TERMINAL_BYTES_V1',arm='C50',global_base64=base64.b64encode(gb).decode(),
        handoff=proof,terminal_capture_base64={k:base64.b64encode(v).decode()for k,v in capture.items()}))
    raw={n:files[n]for n in old.SMALL};raw['CHECKPOINT_COMPLETE.json']=cr
    small=a.raw(dict(schema='V5_CD_CLOSED1024_SMALL_METADATA_READBACK_V1',arm='C50',training_capture_sha256=a.sha(full),
        training_handoff_sha256=a.sha(a.raw(proof)),checkpoint_manifest_sha256=a.sha(cr),tensor_payloads_read=False,
        files={n:dict(bytes=len(v),sha256=a.sha(v),base64=base64.b64encode(v).decode())for n,v in raw.items()}))
    return dict(capture=full,small=small,files=files)


def test_same_closed_state_subset_uses_original_export_and_original_restoration_dispatch(source):
    v=t.prepare(source['capture'],source['small'],arm='C50');p=v['public_plan'];b=v['private_restoration']
    assert p['checkpoint_step']==1024 and len(p['files'])==7 and not p['full_resume_payload']
    assert not v['resource_counters_changed']and not v['HF_IO_performed']and not p['native_admission']
    assert set(b['original_metadata_base64'])==set(old.REWRITTEN)
    old.reject_private(p)
    public={n:source['files'][n]for n in p['files']}
    public.update({n:base64.b64decode(raw)for n,raw in v['public_rewritten_metadata_base64'].items()})
    restored=old.restore_original(public,b,candidate='C50')
    assert restored=={n:source['files'][n]for n in p['files']}
    assert not any(n in restored for n in ('optimizer.pt','scheduler.pt','rng_state.pth','SAMPLER_STATE.json'))


@pytest.mark.parametrize('case',range(9))
def test_changed_small_capture_or_arm_never_becomes_a_delivery_plan(source,case):
    small=json.loads(source['small']);arm='C50'
    if case==0:small['training_capture_sha256']='0'*64
    if case==1:small['training_handoff_sha256']='0'*64
    if case==2:small['checkpoint_manifest_sha256']='0'*64
    if case==3:small['tensor_payloads_read']=True
    if case==4:small['files'].pop('processor/statistics.json')
    if case==5:small['files']['config.json']['base64']=base64.b64encode(b'{}').decode()
    if case==6:arm='D25'
    if case==7:small['files']['config.json']['bytes']=2*1024**2
    if case==8:small['files']['optimizer.pt']=deepcopy(small['files']['config.json'])
    with pytest.raises(PermissionError):t.prepare(source['capture'],a.raw(small),arm=arm)


@pytest.mark.parametrize('case',range(6))
def test_downloads_cannot_add_training_state_swap_arm_or_rewrite_private_identity(source,case):
    v=t.prepare(source['capture'],source['small'],arm='C50');b=v['private_restoration'];arm='C50'
    public={n:source['files'][n]for n in v['public_plan']['files']}
    public.update({n:base64.b64decode(raw)for n,raw in v['public_rewritten_metadata_base64'].items()})
    if case==0:public['optimizer.pt']=b'no'
    if case==1:public['model-00001-of-00001.safetensors']+=b'changed'
    if case==2:arm='D25'
    if case==3:b['original_metadata_base64']['config.json']=base64.b64encode(b'{}').decode()
    if case==4:b['checkpoint_manifest_sha256']='0'*64
    if case==5:b['closed_training_capture_base64']=base64.b64encode(b'{}').decode()
    with pytest.raises((PermissionError,KeyError)):old.restore_original(public,b,candidate=arm)


def test_no_old_4096_or_unclosed_capture_alias_can_supply_CD(source):
    with pytest.raises(PermissionError):old.prepare(source['capture'],source['small'],candidate='C50')
    v=json.loads(source['capture']);v['terminal_capture_base64']['OS_exit_bytes']=base64.b64encode(a.raw(dict(returncode=1))).decode()
    with pytest.raises(PermissionError):t.prepare(a.raw(v),source['small'],arm='C50')
    cr=next(iter(json.loads(source['small'])['files'].values()))
    assert cr['bytes']>0
