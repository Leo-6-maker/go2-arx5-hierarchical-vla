"""Retired aggregate caps do not erase costs or authorize native work."""
from copy import deepcopy
import base64,gzip,json,math,os,time
from pathlib import Path
import pytest

from scripts.pi0294.generalization import minimal_adoption as a,minimal_core_accounting as c
from scripts.pi0294.test_minimal_core_runtime import fixture
from scripts.pi0294.test_minimal_core_v7_successor import v7_fixture


def ledger():
    _,r=v7_fixture();*_,row,_=fixture()
    row.update(r,request=deepcopy(r),request_sha256=c.digest(r),child_run_uuid=r['run_uuid'])
    return {c.FIELD:{r['request_id']:row},'used':{'H200_s':200},'reserved':{'H200_s':100},
        'CPU_s':400,'CPU_reserved_s':500,'new_storage_current_bytes':600,'storage_reserved_bytes':700}


def test_only_valid_v7_rows_retire_aggregate_caps_without_mutating_any_counter(monkeypatch):
    monkeypatch.setattr(a,'validate',lambda g:None)
    from scripts.pi0294.pv1.evaluation_work import MATRIX
    from scripts.pi0294.pv1.future_budget import FINAL
    g=ledger();before=deepcopy(g);caps=dict(H200_s=1,Hopper_CPU_s=1,Hopper_storage_bytes=1)
    out=a.global_limits(g,caps)
    for key,total in [('H200_s',300),('Hopper_CPU_s',900),('Hopper_storage_bytes',1300)]:
        assert out[key]==math.ceil(total)+MATRIX.get(key,0)+FINAL.get(key,0)
    assert g==before and caps==dict.fromkeys(caps,1)
    g[c.FIELD]={}
    assert a.global_limits(g,caps)==caps
    g=ledger();next(iter(g[c.FIELD].values()))['request']['execution_policy_sha256']='0'*64
    assert a.global_limits(g,caps)==caps


@pytest.mark.parametrize('key,value',[('optimizer_updates',1),('checkpoint_step',1024)])
def test_v7_aggregate_permission_is_not_a_science_override(monkeypatch,key,value):
    monkeypatch.setattr(a,'validate',lambda g:None);g=ledger();row=next(iter(g[c.FIELD].values()))
    row['request'][key]=value
    with pytest.raises(PermissionError):a.global_limits(g,dict(Hopper_storage_bytes=1))


def test_actual_cold_remaining23_proposal_preserves_every_old_row_and_hold():
    directory=os.environ.get('E1_REMAINING_PREFLIGHT_CAPTURE')
    if not directory:pytest.skip('Explicit private actual cold capture required')
    b=Path(directory);load=lambda n:json.loads((b/(n+'.private.json')).read_bytes())
    p=load('RUNTIME_ASSEMBLY_PREPARED');gr=load('CURRENT_GLOBAL');lr=load('CURRENT_LOCAL')
    gb=gzip.decompress(base64.b64decode(gr['gzip_base64']));lb=gzip.decompress(base64.b64decode(lr['LOCAL']['gzip_base64']))
    out=c.register(gb,lb,request_bytes=c.raw(p['request']),manifest_bytes=c.raw(p['assembly']['rows']['hopper']['manifest']),
        physical_manifest_bytes=c.raw(p['assembly']['rows']['core']['manifest']),now_epoch=time.time())
    g=json.loads(gb);v=out['proposed_global'];w=out['proposed_LOCAL'];rid=p['request']['request_id']
    assert v['PBS_submissions']==g['PBS_submissions'] and v[c.old.CORE_ROWS]==g[c.old.CORE_ROWS]
    assert v['ARX5_minimal_first512_execution_package']==g['ARX5_minimal_first512_execution_package']
    assert all(v[c.FIELD][key]==row for key,row in g[c.FIELD].items())
    assert v['storage_reserved_bytes']==g['storage_reserved_bytes']+c.HOLD['Hopper_storage_bytes']
    assert v['new_storage_current_bytes']==g['new_storage_current_bytes']+c.CONTROL['Hopper_storage_bytes']
    assert v[c.FIELD][rid]['remaining_conditions']==23 and not out['native_admission']
    c.paired(v,w)
    broken=deepcopy(v);broken['storage_reserved_bytes']=c.HOLD['Hopper_storage_bytes']
    with pytest.raises(PermissionError):c.checked(broken)
