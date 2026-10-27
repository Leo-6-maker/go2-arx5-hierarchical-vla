"""Real closed inputs, explicitly fictitious future count grant/request/OS.

Production pins the separately published direct human grant. Future test
grants are memory-only and do not produce a registry entry, PBS or evidence.
"""
import base64,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_training_accounting as a,amd_training_retry as retry

B=Path(__file__).resolve().parents[4]/'work/private/v5-execution-20261025'
@pytest.fixture
def actual():
    path=B/'GEN1_INDEPENDENT_TRAIN_01_FAILED1497713_TERMINAL_INDEPENDENT_READBACK.private.json'
    if not path.exists():pytest.skip('private exact1497713 closed capture unavailable')
    readback=json.loads(path.read_bytes());capture=json.loads((B/'GEN1_INDEPENDENT_TRAIN_01_FAILED1497713_FULL_INPUTS_READONLY.private.json').read_bytes())
    prepared=json.loads((B/'GEN1_INDEPENDENT_TRAIN_01_PREPARED.private.json').read_bytes())
    local=base64.b64decode(json.loads(base64.b64decode(prepared['small']['plans/GEN1_OWNER_RESOURCES.private.json']))['historical_LOCAL_base64'])
    g=base64.b64decode(readback['GLOBAL_base64']);scope=a.authority.validate(json.loads(g))
    grant=dict(schema=retry.GRANT_SCHEMA,status='DIRECT_HUMAN_AUTHORIZED',
        human_answer='EXPLICIT_CPU_TEST_FIXTURE_NOT_ACTUAL_HUMAN_APPROVAL',
        caps={'PBS_submissions':dict(before=36,after=37)},old_request_id=retry.context.failure.REQUEST,
        old_job_id=retry.context.failure.JOB,existing_four_cap_grant_sha256=a.authority.GRANT_SHA,
        other_caps_clocks_science_and_final_protection_unchanged=True,
        **{k:scope[k]for k in ('T_CD','stop_new_training_epoch','deadline_epoch')})
    return dict(global_bytes=g,historical_LOCAL_bytes=local,grant_bytes=a.raw(grant),
        capture=capture,closed_readback=readback,now_epoch=readback['epoch']+1,
        original_recipe=json.loads(base64.b64decode(prepared['small']['plans/FROZEN_SOURCE_PAIR_RECIPE.json'])))

def adopt(p):return retry.propose(**{k:p[k]for k in ('global_bytes','historical_LOCAL_bytes','grant_bytes','capture','closed_readback','now_epoch')})

@pytest.fixture
def adopted(actual,monkeypatch):
    monkeypatch.setattr(retry,'GRANT_SHA',a.sha(actual['grant_bytes']))
    return adopt(actual)['proposed_global']

def request(g):
    old=a.rows(g)[0];r=deepcopy(old)
    # Copy the actual original native REQUEST, not its terminal central row.
    cap=json.loads((B/'GEN1_INDEPENDENT_TRAIN_01_FAILED1497713_FULL_INPUTS_READONLY.private.json').read_bytes())
    r=json.loads(base64.b64decode(cap['files']['PBS/REQUEST.json']['base64']))
    r.update(source_sha='a'*40,request_id='11111111-1111-4111-8111-111111111111',
        submission_nonce='22222222-2222-4222-8222-222222222222',
        run_uuid='gen1-cold-retry-cpu-fixture',run_manifest_sha256='c'*64,
        cold_input_audit_sha256='d'*64,job_id=None,status=a.STATES[0],
        retry_generation=1,retry_scope_grant_sha256=retry.GRANT_SHA,
        retry_predecessor_context_sha256=g[retry.FIELD]['context_sha256'],submit_count=34)
    r.pop('request_sha256',None);r['request_sha256']=a.sha(a.raw(r));return r

def register(g,p,r=None):
    return a.register(g,request(g)if r is None else r,
        closed=g[retry.FIELD]['predecessor_context']['closed_export'],
        original_recipe=p['original_recipe'],historical_LOCAL_bytes=p['historical_LOCAL_bytes'],now_epoch=p['now_epoch']+1)

def test_production_accepts_only_exact_published_direct_grant_not_future_CPU_fixture(actual):
    # Resolve from this repository, independently of private captured paths.
    path=Path(__file__).resolve().parents[2]/'reports/PI0294/design/parallel-science-20261022/GEN1_GPU_AUTONOMOUS_SUBMISSIONS_AND_FIRST_PBS37_AUTHORIZATION_20261027.json'
    assert a.sha(path.read_bytes())==retry.GRANT_SHA
    assert json.loads(path.read_bytes())['human_answer']=='允许授权多次GPU任务，自主推进'
    with pytest.raises(PermissionError,match='NOT_YET_PINNED'):adopt(actual)
    g=json.loads(actual['global_bytes']);r=a.rows(g)[0]
    fake=dict(r,retry_generation=1,retry_scope_grant_sha256='0'*64,retry_predecessor_context_sha256='1'*64)
    with pytest.raises(PermissionError):a.validate_request(fake,central=True)

def test_actual_new_direct_grant_full_captured_proposal_only_no_native_IO(actual):
    path=Path(__file__).resolve().parents[2]/'reports/PI0294/design/parallel-science-20261022/GEN1_GPU_AUTONOMOUS_SUBMISSIONS_AND_FIRST_PBS37_AUTHORIZATION_20261027.json'
    v=adopt(dict(actual,grant_bytes=path.read_bytes()))
    assert not v['full_remaining_proof']['cap_changes_required']and not v['native_admission']
    assert not v['actual_LOCAL_written']and v['new_PBS_submissions']==v['new_requests']==0

def test_fixture_one_count_adoption_changes_no_old_caps_costs_reservations_or_history(actual,adopted):
    before=json.loads(actual['global_bytes']);g=adopted
    assert all(g[k]==v for k,v in before.items()if k!='revision')
    assert g[a.authority.FIELD]['caps']['PBS_submissions']==36
    assert retry.effective_caps(g,a.authority.CAPS)['PBS_submissions']==37
    assert a.scope.full_forecast(a.raw(g),actual['historical_LOCAL_bytes'])['cap_changes_required']=={}
    from scripts.pi0294.pv1.resources import pbs_limit
    from scripts.pi0294.pv1.neural_ledger import check
    assert pbs_limit(g)==37;check(g)
    with pytest.raises(PermissionError):adopt(dict(actual,global_bytes=a.raw(g)))

@pytest.mark.parametrize('bad',['wrong_bytes','additional_cap','wrong_predecessor','reset_clock','GLOBAL_changed','LOCAL','late','open_lease'])
def test_fixture_pin_still_rejects_scope_expansion_changed_bytes_or_incomplete_predecessor(actual,monkeypatch,bad):
    p=deepcopy(actual);grant=json.loads(p['grant_bytes'])
    if bad=='wrong_bytes':monkeypatch.setattr(retry,'GRANT_SHA','0'*64)
    elif bad in ('additional_cap','wrong_predecessor','reset_clock'):
        if bad=='additional_cap':grant['caps']['H200_s']=dict(before=435600,after=435601)
        elif bad=='wrong_predecessor':grant['old_request_id']='other'
        else:grant['deadline_epoch']+=1
        p['grant_bytes']=a.raw(grant);monkeypatch.setattr(retry,'GRANT_SHA',a.sha(p['grant_bytes']))
    else:
        monkeypatch.setattr(retry,'GRANT_SHA',a.sha(p['grant_bytes']))
        if bad=='GLOBAL_changed':p['global_bytes']=a.raw(dict(json.loads(p['global_bytes']),revision=99999))
        elif bad=='LOCAL':p['historical_LOCAL_bytes']=b'{}'
        elif bad=='late':p['now_epoch']=grant['deadline_epoch']
        else:
            lease=json.loads(base64.b64decode(p['closed_readback']['lease_base64']));lease['status']='ACTIVE';p['closed_readback']['lease_base64']=base64.b64encode(a.raw(lease)).decode();p['closed_readback']['lease_sha256']=a.sha(a.raw(lease))
    with pytest.raises((PermissionError,KeyError)):adopt(p)

def test_second_request_full_reserve_bind_settle_keeps_first_failure_and_original_package(actual,adopted):
    before=deepcopy(adopted);r=request(before);v=register(before,actual,r);g=v['proposed_global']
    assert v['full_remaining_proof']['cap_changes_required']=={}
    assert v['full_remaining_proof']['dimensions']['PBS_submissions']['projected']==37
    assert g['used']['PBS_submissions']==34 and len(a.rows(g))==2
    assert g['PBS_submissions'][:-1]==before['PBS_submissions']
    assert g[a.authority.FIELD][a.FIELD]==before[a.authority.FIELD][a.FIELD]
    assert g[a.original.PACKAGE]==before[a.original.PACKAGE]
    assert g['CPU_s']==before['CPU_s']+1800 and g['new_storage_current_bytes']==before['new_storage_current_bytes']+64*1024**2
    assert g['reserved']['H200_s']==before['reserved']['H200_s']+22560
    assert g['CPU_reserved_s']==before['CPU_reserved_s']+54000
    assert g['storage_reserved_bytes']==before['storage_reserved_bytes']+70000000000
    assert a.profile(r)==retry.PROFILE and a.profile(a.rows(g)[0])==a.PROFILE
    with pytest.raises(PermissionError):register(g,actual,r)
    g=a.bind(g,r['request_id'],'123.stdct-mgmt-02',now_epoch=actual['now_epoch']+2)
    from scripts.pi0294.pv1.neural_ledger import check
    check(g)
    with pytest.raises(PermissionError):a.bind(g,r['request_id'],'124.stdct-mgmt-02',now_epoch=actual['now_epoch']+3)
    native=dict(job_state='F',job_id='123.stdct-mgmt-02',request_id=r['request_id'],submission_nonce=r['submission_nonce'],Exit_status=0,conservative_GPU_s=8500,conservative_CPU_s=8200)
    effects=dict(schema=a.EFFECTS,job_id=native['job_id'],request_id=r['request_id'],request_sha256=r['request_sha256'],arm=a.bank.IDENTITY,optimizer_updates_committed=1024,query_attempts=29,model_load_attempts=3,pending_updates=[],unknown_query_intents=[])
    audit=dict(status=a.AUDIT,arm=a.bank.IDENTITY,actual_consumed_cursor=32768,parent_checkpoint_manifest_sha256=a.original.PARENT,statistics_sha256=a.original.STATS)
    out=a.settle_training(g,r['request_id'],native=native,effects=effects,audit=audit,now_epoch=actual['now_epoch']+9000)
    assert out['optimizer_updates']==14786 and out['generation_queries']==4511
    assert out['used']['H200_s']==158929 and out['used']['PBS_submissions']==34
    assert out['PBS_submissions'][:-1]==before['PBS_submissions']
    assert out[a.authority.FIELD][a.FIELD]==before[a.authority.FIELD][a.FIELD]
    assert out[retry.FIELD]['endpoint']==audit and not a.rows(out)[0]['scientific_endpoint_complete']
    check(out)
    with pytest.raises(PermissionError):a.settle_training(out,r['request_id'],native=native,effects=effects,audit=audit,now_epoch=actual['now_epoch']+9001)
    with pytest.raises(PermissionError):register(out,actual)

@pytest.mark.parametrize('bad',['old_request','old_nonce','old_source','bank','context','grant','original_alias','pending'])
def test_only_new_same_science_request_and_known_inactive_state_can_reserve(actual,adopted,bad):
    g=deepcopy(adopted);r=request(g);old=a.rows(g)[0]
    if bad=='old_request':r['request_id']=old['request_id']
    elif bad=='old_nonce':r['submission_nonce']=old['submission_nonce']
    elif bad=='old_source':r['source_sha']=old['source_sha']
    elif bad=='bank':r['bank_manifest_sha256']='0'*64
    elif bad=='context':r['retry_predecessor_context_sha256']='0'*64
    elif bad=='grant':r['retry_scope_grant_sha256']='0'*64
    elif bad=='original_alias':
        for k in retry.REQUEST_KEYS:r.pop(k)
    else:g['unknown_effects']=['fixture unresolved write']
    with pytest.raises(PermissionError):register(g,actual,r)

def test_original_once_io_applies_only_temporary_fixture_proposals_and_rejects_replay(actual,adopted,tmp_path):
    from scripts.pi0294.dec1.source_cpu_once_io import commit
    path=tmp_path/'GLOBAL.json';path.write_bytes(actual['global_bytes'])
    def cas(key,out):
        return commit(path,expected_before_sha256=a.sha(path.read_bytes()),proposed_bytes=a.raw(out),request_id=key,intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_suffix('.owner.lock'))
    cas('CPU-FIXTURE-COUNT-ADOPT',adopted);g=register(adopted,actual)['proposed_global'];cas('CPU-FIXTURE-REGISTER',g)
    before=path.read_bytes()
    with pytest.raises(FileExistsError):cas('CPU-FIXTURE-REGISTER',dict(g,revision=g['revision']+1))
    assert path.read_bytes()==before and len(a.rows(json.loads(before)))==2

@pytest.fixture
def root_fixture(actual,adopted,tmp_path):
    """Temporary paths stand for registered hosts; no native qualification."""
    g=deepcopy(adopted);m=g[retry.FIELD];prior=m['predecessor_context']
    old=tmp_path/'campaign/children'/Path(prior['predecessor_root']).name;old.mkdir(parents=True)
    root=old.with_name('gen1-retry-fixture');(root/'plans').mkdir(parents=True)
    prior['predecessor_root']=str(old);m['context_sha256']=a.sha(a.raw(prior))
    (old/'RUN_MANIFEST.json').write_bytes(base64.b64decode(actual['capture']['files']['RUN_MANIFEST.json']['base64']))
    lease=tmp_path/'old-lease.json';lease.write_bytes(base64.b64decode(actual['closed_readback']['lease_base64']))
    binding=json.loads(base64.b64decode(actual['capture']['files']['OUTPUT_BINDING.private.json']['base64']));binding['lease']=str(lease);(old/'OUTPUT_BINDING.private.json').write_bytes(a.raw(binding))
    p=json.loads((B/'GEN1_INDEPENDENT_TRAIN_01_PREPARED.private.json').read_bytes());manifest=deepcopy(p['manifest']);seal=deepcopy(p['seal']);r=request(g)
    manifest.update(run_uuid=root.name,output_root=str(root));manifest['code']['vla_commit']=seal['source_sha']=r['source_sha'];manifest['code']['source_seal_sha256']=a.sha(a.raw(seal));manifest['execution']['profile']=retry.PROFILE
    r.update(run_uuid=root.name,run_manifest_sha256=a.sha(a.raw(manifest)));r.pop('request_sha256')
    for path,value in [(root/'RUN_MANIFEST.json',manifest),(root/'SOURCE_SEAL.json',seal),(root/'plans/GEN1_CLOSED_TRAIN_PREDECESSOR_CONTEXT.private.json',prior)]:path.write_bytes(a.raw(value));path.chmod(0o400)
    (root.parent.parent/'HOPPER_RESOURCE_LEDGER.json').write_bytes(a.raw(g))
    return root,r,lease

def test_runtime_and_owner_share_actual_path_join_to_same_closed_predecessor(root_fixture):
    root,r,_=root_fixture;v=retry.verify_registered_root(root,r)
    assert v['predecessor_job_id']=='1497713.stdct-mgmt-02'
    assert not v['native_training_permission']and not(root/'A40_NATIVE_EFFECT_LEDGER.json').exists()
    assert not(root/'control/resident-owner').exists()

@pytest.mark.parametrize('bad',['old_lease_active','context_mode','context_bytes','different_parent'])
def test_own_root_join_refuses_stale_lease_changed_context_or_inherited_weights(root_fixture,bad):
    root,r,lease=root_fixture
    if bad=='old_lease_active':x=json.loads(lease.read_bytes());x['status']='ACTIVE';lease.write_bytes(a.raw(x))
    elif bad=='context_mode':(root/'plans/GEN1_CLOSED_TRAIN_PREDECESSOR_CONTEXT.private.json').chmod(0o600)
    elif bad=='context_bytes':
        path=root/'plans/GEN1_CLOSED_TRAIN_PREDECESSOR_CONTEXT.private.json';path.chmod(0o600);path.write_bytes(b'{}');path.chmod(0o400)
    else:
        path=root/'RUN_MANIFEST.json';m=json.loads(path.read_bytes());m['science']['parent_checkpoint_manifest_sha256']='0'*64;path.chmod(0o600);path.write_bytes(a.raw(m));path.chmod(0o400);r['run_manifest_sha256']=a.sha(a.raw(m))
    with pytest.raises(PermissionError):retry.verify_registered_root(root,r)

def test_applied_fixture_adoption_reply_loss_remains_unknown_without_resend(actual,adopted,tmp_path):
    from scripts.pi0294.dec1.source_cpu_once_io import commit,reconcile
    from scripts.pi0294.dec1.owner_transport import Journal,UnknownMutation
    path=tmp_path/'GLOBAL.json';path.write_bytes(actual['global_bytes']);folder=tmp_path/'rpc';folder.mkdir()
    j=Journal(folder,dict(run_uuid='fixture',job_id='NO_NATIVE_JOB',source_sha='a'*40,request_sha256='b'*64));key='CPU-FIXTURE-ADOPT';calls=[]
    def lost(intent,limit):
        calls.append(1);commit(path,expected_before_sha256=a.sha(actual['global_bytes']),proposed_bytes=a.raw(adopted),request_id=key,intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_suffix('.owner.lock'));raise TimeoutError('fixture lost applied CAS reply')
    with pytest.raises(UnknownMutation):j.call('ADOPT','MUTATION',lost)
    view=reconcile(path,intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'))
    assert view['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'and not view['automatic_retry_allowed']
    with pytest.raises(UnknownMutation):j.call('ADOPT','MUTATION',lost)
    assert calls==[1]and json.loads(j.state_path.read_bytes())['status']=='UNKNOWN'
