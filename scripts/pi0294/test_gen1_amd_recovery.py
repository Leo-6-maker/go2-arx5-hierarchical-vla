"""Actual captured failure bytes and original temporary onceIO, no new physics."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_recovery as r,amd_adoption as a
from scripts.pi0294.dec1.source_cpu_once_io import commit,reconcile

W=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
B=W/'work/private/v5-execution-20261025'


@pytest.fixture
def actual():
    audits=sorted(B.glob('GEN1_AMD_PILOT01_COUNTER_FAILURE_INDEPENDENT_READONLY_*.private.json'))
    if not audits:pytest.skip('actual private saved failure capture unavailable')
    audit=json.loads(audits[-1].read_bytes())
    g=json.loads(base64.b64decode(json.loads((B/'GEN1_AMD_PILOT01_GLOBAL_RESERVE_PILOT12.private.json').read_bytes())['value']['GLOBAL_base64']))
    return g,audit


def test_actual_saved_raw_counters_replace_stale_lower_bound_without_refund(actual):
    g,audit=actual;v=r.failure_proposal(g,audit,now_epoch=audit['epoch']+1)
    actual=v['actual'];assert actual['live_ticks']==30300 and actual['shadow_ticks']==30400
    assert actual['engine_starts']==162 and actual['episode_starts']==2
    assert actual['Core_CPU_s']==1187.7440292649999 and not actual['unknown_effects']
    n=v['proposed_global'];assert len(n[a.FIELD]['teacher_pilot']['unattempted_condition_ids'])==10
    assert all(n[k]==v for k,v in g.items()if k not in(a.FIELD,'revision'))
    assert not v['native_training_or_physics_admission']
    with pytest.raises(PermissionError):r.failure_proposal(n,audit,now_epoch=audit['epoch']+2)


@pytest.mark.parametrize('bad',['source','manifest','replay','unknown','clock','CPU','raw_ticks','safety','S1','success','classification'])
def test_exact_failed_boundary_rejects_changed_identity_or_scientific_failure(actual,bad):
    g,audit=map(deepcopy,actual);now=audit['epoch']+1
    if bad=='source':audit['source']='0'*40
    elif bad=='manifest':audit['workers']['0']['manifest_sha256']='0'*64
    elif bad=='replay':g[a.FIELD]['teacher_pilot']['status']='NATIVE_ENGINEERING_TERMINAL'
    elif bad=='unknown':audit['unknown_effects']=['unknown']
    elif bad=='clock':now=1793537549.8133287
    elif bad=='CPU':audit['Core_CPU_s']=float('nan')
    elif bad=='raw_ticks':audit['live_ticks']=30000
    elif bad=='safety':audit['workers']['0']['native_safety_failures']=1
    elif bad=='S1':audit['workers']['0']['S1_rejections']=1
    elif bad=='success':audit['workers']['0']['data_admitted']=True
    elif bad=='classification':audit['workers']['0']['scientific_negative']=True
    with pytest.raises(PermissionError):r.failure_proposal(g,audit,now_epoch=now)


def test_original_onceIO_applied_reply_lost_readonly_and_no_replay(actual,tmp_path):
    g,audit=actual;v=r.failure_proposal(g,audit,now_epoch=audit['epoch']+1)
    file=tmp_path/'GLOBAL.json';file.write_bytes(a.raw(g));intent=tmp_path/'INTENT.json';receipt=tmp_path/'RECEIPT.json'
    args=dict(expected_before_sha256=a.sha(a.raw(g)),proposed_bytes=a.raw(v['proposed_global']),
        request_id='GEN1-AMD-PILOT12-FAILED-01',intent_path=intent,receipt_path=receipt,lock_path=tmp_path/'owner.lock')
    commit(file,**args)
    assert reconcile(file,intent_path=intent,receipt_path=receipt)['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
    with pytest.raises(FileExistsError):commit(file,**args)


def test_actual_failed_cost_plus_complete_cold_future_fits_same_caps(actual):
    from scripts.pi0294.generalization import amd_pilot_scope as scope
    g,audit=actual;n=r.failure_proposal(g,audit,now_epoch=audit['epoch']+1)['proposed_global']
    local=base64.b64decode(json.loads((B/'GEN1_AMD_PILOT01_PREPARED.private.json').read_bytes())['historical_LOCAL_base64'])
    proof=scope.full_forecast(a.raw(n),local)
    assert proof['cap_changes_required']=={} and proof['matrix3_and_final20GPUh40GiB_retained']
    assert proof['dimensions']['PBS_submissions']['projected']==36
    assert proof['dimensions']['Core_storage_bytes']['projected']<=477*1024**3
    assert proof['dimensions']['episode_starts']['additional']>=14
    assert proof['T_CD']==1792932749.8133287 and proof['deadline']==1793537549.8133287


@pytest.mark.parametrize('bad',[None,'active_lease','different_audit','old_root','resample','duplicate','late','scientific_failure'])
def test_cold_recovery_requires_closed_exact_engineering_boundary(actual,bad):
    from scripts.pi0294.generalization.amd_capture import cards,COUNTER_POLICY
    g,audit=actual;g=r.failure_proposal(g,audit,now_epoch=audit['epoch']+1)['proposed_global']
    prepared=json.loads((B/'GEN1_AMD_PILOT01_PREPARED.private.json').read_bytes())
    packet=base64.b64decode(prepared['common']['plans/GEN1_V5_PREVIEW_V2.private.json'])
    leases={key:dict(status='CLOSED',terminal_kind='KNOWN_ENGINEERING_FAILURE_WITH_REAL_PHYSICS',
        independent_failure_audit_sha256=g[a.FIELD]['teacher_pilot']['actual']['independent_audit_sha256'],
        old_costs_and_raw_preserved=True,admitted_demonstrations=0,
        manifest_sha256=g[a.FIELD]['teacher_pilot']['workers'].get(key,{}).get('manifest_sha256','a'*64))
        for key in ('owner','0','1')}
    workers={str(i):dict(root='/new/children/gen1-amd-teacher-r1-'+str(i).zfill(2),
        run_uuid='gen1-amd-teacher-r1-'+str(i).zfill(2),manifest_sha256=str(i)*64,
        counter_policy=COUNTER_POLICY,attempt_ids=[x['id']for x in cards(packet,i)])for i in (0,1)}
    now=audit['epoch']+2
    if bad=='active_lease':leases['owner']['status']='ACTIVE'
    elif bad=='different_audit':leases['1']['independent_failure_audit_sha256']='f'*64
    elif bad=='old_root':workers['0']['root']=g[a.FIELD]['teacher_pilot']['workers']['0']['root']
    elif bad=='resample':workers['0']['attempt_ids'][0]='another-condition'
    elif bad=='duplicate':g[a.FIELD]['teacher_pilot']['engineering_recovery']={}
    elif bad=='late':now=1793451149.8133287
    elif bad=='scientific_failure':g[a.FIELD]['teacher_pilot']['actual']['admitted_demonstrations']=1
    call=lambda:r.recovery_allocation(g,packet_bytes=packet,workers=workers,closed_leases=leases,now_epoch=now)
    if bad is None:
        v=call();assert v['row']['prior_failed_physical_attempts']==2
        assert v['row']['possible_total_physical_attempts']==14 and v['row']['engine_starts']==1056
        assert not v['new_PBS']and not v['row']['valid_scientific_failure_replay']
        assert v['proposed_global'][a.FIELD]['teacher_pilot']['actual']==g[a.FIELD]['teacher_pilot']['actual']
    else:
        with pytest.raises(PermissionError):call()
