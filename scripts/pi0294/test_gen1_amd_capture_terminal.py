"""Actual saved R1 failure bytes and real temporary onceIO, no raw audit PASS."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_capture_terminal as t,amd_adoption as a
from scripts.pi0294.dec1.source_cpu_once_io import commit,reconcile

W=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
B=W/'work/private/v5-execution-20261025'

@pytest.fixture
def inputs():
    file=B/'GEN1_AMD_PILOT_R1_RAW_AUDIT_FAILURE_SAVED_EFFECTS_INDEPENDENT_READONLY.private.json'
    if not file.exists():pytest.skip('actual private R1 saved-effects capture unavailable')
    audit=json.loads(file.read_bytes());g=json.loads(base64.b64decode(json.loads((B/'GEN1_AMD_PILOT_R1_GLOBAL_RESERVE.private.json').read_bytes())['value']['GLOBAL_base64']))
    return g,audit


def test_actual_complete_native_effects_preserved_without_relabeling_failed_audits(inputs):
    g,audit=inputs;v=t.failure_proposal(g,audit,now_epoch=audit['epoch']+1)
    assert v['actual']['live_ticks']==v['actual']['shadow_ticks']==32000
    assert v['actual']['engine_starts']==173 and v['actual']['Core_CPU_s']==1257.589495225
    assert v['actual']['independently_admitted_demonstrations']==0
    n=v['proposed_global'];r=n[a.FIELD]['teacher_pilot']['engineering_recovery']
    assert len(r['saved_full_cases'])==2 and len(r['unattempted_condition_ids'])==10
    assert r['original_unknown_lists_and_failed_audits_preserved']and not r['native_training_admitted']
    assert all(n[k]==v for k,v in g.items()if k not in(a.FIELD,'revision'))
    with pytest.raises(PermissionError):t.failure_proposal(n,audit,now_epoch=audit['epoch']+2)
    # Financial-only settlement remains legal after the hard deadline.
    assert t.failure_proposal(g,audit,now_epoch=1793537550.)['new_PBS']==0


@pytest.mark.parametrize('bad',['source','manifest','incomplete','physics','unknown','seed_error','native_safety','fake_admission','CPU','raw_digest'])
def test_exact_raw_audit_failure_boundary_rejects_unrelated_or_unknown_effects(inputs,bad):
    g,audit=deepcopy(inputs);case=audit['cases']['0']
    if bad=='source':audit['source']='0'*40
    elif bad=='manifest':case['manifest_sha256']='0'*64
    elif bad=='incomplete':case['raw_status']='FAILED_PRESERVED'
    elif bad=='physics':audit['live_ticks']=0
    elif bad=='unknown':audit['unknown_native_effects']=['unknown']
    elif bad=='seed_error':case['original_raw_audit_error']='other'
    elif bad=='native_safety':case['native_safety_failures']=1
    elif bad=='fake_admission':case['independent_raw_admitted']=True
    elif bad=='CPU':audit['Core_CPU_s']=float('nan')
    elif bad=='raw_digest':case['raw_manifest_sha256']='bad'
    with pytest.raises(PermissionError):t.failure_proposal(g,audit,now_epoch=audit['epoch']+1)


def test_original_once_commit_then_lost_reply_can_only_reconcile(inputs,tmp_path):
    g,audit=inputs;v=t.failure_proposal(g,audit,now_epoch=audit['epoch']+1)
    p=tmp_path/'GLOBAL.json';p.write_bytes(a.raw(g));intent=tmp_path/'INTENT.json';receipt=tmp_path/'RECEIPT.json'
    args=dict(expected_before_sha256=a.sha(a.raw(g)),proposed_bytes=a.raw(v['proposed_global']),request_id='GEN1-R1-RAW-AUDIT-FAILURE-01',intent_path=intent,receipt_path=receipt,lock_path=tmp_path/'owner.lock')
    commit(p,**args);assert reconcile(p,intent_path=intent,receipt_path=receipt)['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
    with pytest.raises(FileExistsError):commit(p,**args)


@pytest.mark.parametrize('bad',[None,'active','wrong_audit','duplicate','late'])
def test_only_closed_source_and_new_saved_only_reaudit_role_can_be_reserved(inputs,bad):
    g,audit=inputs;g=t.failure_proposal(g,audit,now_epoch=audit['epoch']+1)['proposed_global'];row=g[a.FIELD]['teacher_pilot']['engineering_recovery']
    leases={key:dict(status='CLOSED',terminal_kind='COMPLETE_NATIVE_TRAJECTORIES_WITH_RAW_AUDIT_ENGINEERING_FAILURE',independent_saved_effects_audit_sha256=row['independent_saved_effects_audit_sha256'],original_failed_audits_and_unknown_lists_preserved=True,manifest_sha256=row['workers'].get(key,{}).get('manifest_sha256','a'*64))for key in ('owner','0','1')};now=audit['epoch']+2
    if bad=='active':leases['owner']['status']='ACTIVE'
    elif bad=='wrong_audit':leases['0']['independent_saved_effects_audit_sha256']='b'*64
    elif bad=='duplicate':row['saved_data_reaudit']={}
    elif bad=='late':now=1793537550.
    call=lambda:t.reaudit_allocation(g,run_uuid='gen1-amd-saved-r1-reaudit-01',manifest_sha256='c'*64,closed_leases=leases,now_epoch=now)
    if bad is None:
        v=call();r=v['proposed_global'][a.FIELD]['teacher_pilot']['engineering_recovery']['saved_data_reaudit'];assert r['engine_starts']==4 and r['physics_ticks']==r['episode_starts']==0
        assert not r['original_raw_files_may_be_modified']and not r['Teacher_replay']and not v['new_PBS']
    else:
        with pytest.raises(PermissionError):call()
