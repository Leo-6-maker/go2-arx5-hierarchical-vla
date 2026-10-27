"""Actual saved failure capture with deliberate corruption; no native work."""
import base64,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_export_seal_failure as f

B=Path(__file__).resolve().parents[4]/'work/private/v5-execution-20261025'
@pytest.fixture
def capture():
    p=B/'GEN1_VANDA_OFFICIAL_EXPORT_GIT_REPAIR02_PARENT04_SEAL_FAILURE_CAPTURE.private.json'
    if not p.exists():pytest.skip('actual private native failure capture unavailable')
    return json.loads(p.read_bytes())

def test_actual_known_saved_exports_not_bank_or_training_success(capture):
    r=f.audit(capture);a=r['audit']
    assert a['completed_export_parents']==11 and a['R0_admitted_parents']==10
    assert not a['training_bank_qualified']and not a['export_finished']and a['unknown_effects']==[]
    assert a['original_failed_counter_and_UNKNOWN_unchanged']and a['support']['missing']==[]
    assert len(a['preserved_R0_rejection_ids'])==1 and len(a['saved_numeric_members'])==10
    assert r['native']['conservative_CPU_s']==150 and r['native']['Exit_status']==1
    assert len(r['prospective_bank']['episodes'])==10

@pytest.mark.parametrize('bad',['active','wrong_job','GPU','owner','bank','missing','changed','permission','foreign','CPU_unknown','source'])
def test_no_generic_failure_to_known_zero_or_bank_pass(capture,bad):
    c=deepcopy(capture)
    if bad=='active':c['native'][f.JOB]['job_state']='R'
    elif bad=='wrong_job':c['native'][f.JOB]['Variable_List']['GEN1_REQUEST_ID']='wrong'
    elif bad=='GPU':c['native'][f.JOB]['Resource_List']['ngpus']=1
    elif bad=='owner':c['actors'][0]['actual_procstart']=c['actors'][0]['expected_procstart']
    elif bad=='bank':c['bank_exists']=True
    elif bad=='missing':c['files'].pop(next(n for n in c['files']if n.endswith('/EXPORT_RECEIPT.json')))
    elif bad=='changed':next(x for x in c['inventory']if x['path'].endswith('.npz'))['sha256']='0'*64
    elif bad=='permission':next(x for x in c['inventory']if x['path'].endswith('.npz'))['mode']=0o666
    elif bad=='foreign':next(x for x in c['inventory']if x['path'].endswith('.npz'))['uid']=0
    elif bad=='CPU_unknown':c['files']['RESOURCE_LEDGER.json']['base64']=base64.b64encode(b'{}').decode()
    else:next(iter(c['modules'].values()))['sha256']='0'*64
    with pytest.raises((PermissionError,KeyError,ValueError)):f.audit(c)

def test_assembly_seals_owned_numeric_and_manifest_before_original_qualifier(tmp_path,monkeypatch):
    from scripts.pi0294.generalization import amd_export_runtime as r
    from scripts.pi0294.data_cov1 import mixed_batch
    root=tmp_path;out=root/'datasets/GEN1-PILOT-fixture-T1';out.mkdir(parents=True)
    numeric=out/'numeric/x.npz';numeric.parent.mkdir();numeric.write_bytes(b'fixture numeric');numeric.chmod(0o600)
    manifest=out/'DATASET_MANIFEST.json';manifest.write_text('{}');manifest.chmod(0o600)
    receipt=out/'EXPORT_RECEIPT.json';row=dict(id=out.name,file='numeric/x.npz',sha256=r.sha(numeric));receipt.write_text(json.dumps(row))
    (root/'plans').mkdir();(root/'plans/GEN1_V5_PREVIEW_V2.private.json').write_text('{}')
    monkeypatch.setattr(r,'read',lambda p:row if Path(p)==receipt else {})
    from scripts.pi0294.generalization import amd_bank
    monkeypatch.setattr(amd_bank,'validate',lambda *a:None)
    def original_qualifier(*a,**k):
        assert numeric.stat().st_mode&0o777==manifest.stat().st_mode&0o777==0o400
        assert r.sha(numeric)==row['sha256']
        raise RuntimeError('observed_original_qualifier_before_any_bank_creation')
    monkeypatch.setattr(mixed_batch,'qualify_all',original_qualifier)
    with pytest.raises(RuntimeError,match='observed_original_qualifier'):
        r.assemble(root,dict(episodes=[row]),None,None,None,lambda:None,{})
    assert not(root/'datasets/FROZEN_GEN1_BANK').exists()

def test_real_failure_settlement_releases_only_own_hold_preserves_all_old_costs(capture):
    p=B/'GEN1_CPU1497612_SAVED_FAILURE_FULL_AUDIT_CAPTURE.private.json'
    if not p.exists():pytest.skip('independent aggregate output readback unavailable')
    c=json.loads(p.read_bytes());r=f.audit(c);g=json.loads(base64.b64decode(c['GLOBAL_base64']));e=f.e
    old=deepcopy(e.original.rows(g));package=deepcopy(g[e.original.original.PACKAGE]);before=deepcopy(g)
    out=e.settle(g,request_id=f.REQUEST,native=r['native'],audit=r['audit'])
    assert out['CPU_s']==before['CPU_s']+150 and out['CPU_reserved_s']==before['CPU_reserved_s']-36000
    assert out['new_storage_current_bytes']==before['new_storage_current_bytes']+e.METADATA
    assert out['storage_reserved_bytes']==before['storage_reserved_bytes']-e.METADATA
    assert out['used']==before['used']and out['reserved']==before['reserved']
    assert e.original.rows(out)==old and out[e.original.original.PACKAGE]==package
    assert not e.rows(out)[0]['training_bank_qualified']and not e.rows(out)[0]['export_finished']
    assert capture==c or capture['files']==c['files']
    with pytest.raises(PermissionError):e.settle(out,request_id=f.REQUEST,native=r['native'],audit=r['audit'])
