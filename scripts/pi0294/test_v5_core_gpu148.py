"""Captured C09 closed evidence, CPU-only authority and original onceIO checks."""
import base64,gzip,json,os
from pathlib import Path
from copy import deepcopy
import pytest
from scripts.pi0294.generalization import minimal_gpu_repair as cap,minimal_adoption as old
from scripts.pi0294.pv1 import mix_core_caps_continuation as cont,mix_core_accounting as c,mix_accounting as a
from scripts.pi0294.dec1 import source_cpu_once_io as once

@pytest.fixture
def actual():
    d=os.environ.get('ARX5_CORE_CLOSED_CAPTURE')
    if not d:pytest.skip('explicit private C09 closed capture required')
    p=json.loads(Path(d).read_bytes())['value']
    gb=gzip.decompress(base64.b64decode(p['GLOBAL_gzip_base64']));lb=gzip.decompress(base64.b64decode(p['LOCAL_gzip_base64']))
    grant=(Path(__file__).resolve().parents[2]/'reports/PI0294/design/parallel-science-20261022/V5_CORE_C50_GPU147_TO148_DIRECT_HUMAN_GRANT_20261028.json').read_bytes()
    return p,gb,lb,grant

def proposal(actual):
    p,gb,lb,h=actual
    return cap.propose(gb,lb,h,closed_readback=p,now_epoch=p['observed_epoch']+.1)

def test_only_one_hour_and_append_marker_preserves_every_cost_reservation_and_old_grant(actual):
    p,gb,lb,h=actual;v=proposal(actual);g,l=v['proposed_global'],v['proposed_LOCAL']
    before=json.loads(gb);prior=json.loads(lb)
    assert set(g)-set(before)==set(l)-set(prior)=={cap.FIELD}
    assert all(g[k]==x for k,x in before.items()if k!='revision')
    assert all(l[k]==x for k,x in prior.items())
    assert g[old.FIELD]==before[old.FIELD]and old.CAPS['H200_s']==529200
    assert old.global_limits(g,dict(H200_s=529200))==dict(H200_s=532800)
    effective=old.effective_caps(g,old.CAPS)
    assert effective==dict(old.CAPS,H200_s=532800)
    assert old.effective_caps(g,effective)==effective
    c.paired(g,l)

@pytest.mark.parametrize('kind',['cap','grant','terminal','clock','dual'])
def test_authority_cannot_change_other_caps_clock_terminal_grant_or_only_one_ledger(actual,kind):
    v=proposal(actual);g,l=v['proposed_global'],v['proposed_LOCAL']
    if kind=='cap':g[cap.FIELD]['after_H200_s']+=1
    elif kind=='grant':g[cap.FIELD]['grant_base64']=base64.b64encode(b'not approved').decode()
    elif kind=='terminal':g[cap.FIELD]['terminal_native_sha256']='0'*64
    elif kind=='clock':g[cap.FIELD]['deadline_epoch']+=1
    else:l.pop(cap.FIELD)
    with pytest.raises(PermissionError):c.paired(g,l)

@pytest.mark.parametrize('kind',['stale','lease','active','repeated'])
def test_real_cold_fresh_closed_and_once_boundaries(actual,kind):
    p,gb,lb,h=deepcopy(actual)
    if kind=='stale':p['observed_epoch']-=121
    elif kind=='lease':p['leases']['GPU']['status']='ACTIVE'
    elif kind=='active':g=json.loads(gb);g['pending_updates']=['CPU_NEGATIVE'];gb=a.raw(g)
    else:
        v=proposal(actual);gb=a.raw(v['proposed_global']);lb=a.raw(v['proposed_LOCAL'])
    with pytest.raises(PermissionError):cap.propose(gb,lb,h,closed_readback=p,now_epoch=actual[0]['observed_epoch']+.1)

def test_continuation_restores_original_panel_and_all_future_fits_with_no_extra_PBS(actual):
    p,gb,lb,h=actual;v=proposal(actual);g,l=v['proposed_global'],v['proposed_LOCAL']
    q=dict(p,GLOBAL_sha256=a.sha(a.raw(g)),LOCAL_sha256=a.sha(a.raw(l))) # CPU-only post-adoption projection
    out=cont.propose_bound_age(a.raw(g),a.raw(l),closed_readback=q,new_request_id='12345678-1234-4234-8234-123456789abc',new_nonce='12345678-1234-4234-8234-123456789abd',now_epoch=p['observed_epoch']+.2)
    assert not out['full_future_proof']['cap_changes_required']
    assert out['proposed_global']['used']==g['used']
    assert out['proposed_global']['PBS_submissions']==g['PBS_submissions']
    assert cont.validate_bound_age(out['proposed_global'])['new_PBS']==0
    assert out['full_future_proof']['dimensions']['H200_s']['projected']==529372

def test_original_onceIO_temp_CAS_and_unknown_receipt_no_replay(actual,tmp_path,monkeypatch):
    v=proposal(actual);path=tmp_path/'G';path.write_bytes(actual[1]);args=dict(expected_before_sha256=a.sha(actual[1]),proposed_bytes=a.raw(v['proposed_global']),request_id='CPU_GPU148',intent_path=tmp_path/'I',receipt_path=tmp_path/'R',lock_path=tmp_path/'L')
    orig=once._exclusive
    def loss(path,body,*a):
        if path==args['receipt_path']:raise OSError('CPU_LOST_RECEIPT')
        return orig(path,body,*a)
    monkeypatch.setattr(once,'_exclusive',loss)
    with pytest.raises(OSError):once.commit(path,**args)
    proof=once.reconcile(path,intent_path=args['intent_path'],receipt_path=args['receipt_path'])
    assert proof['status']=='CURRENT_PROPOSED_BYTES_RECEIPT_UNCONFIRMED'and not proof['automatic_retry_allowed']
    with pytest.raises(FileExistsError):once.commit(path,**args)
