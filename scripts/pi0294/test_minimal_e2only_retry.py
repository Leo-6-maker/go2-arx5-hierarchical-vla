"""Actual closed-zero predecessor; every future request and queue is a CPU fixture."""
import base64,gzip,json,os,time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.pi0294.generalization import minimal_execution as e,minimal_e2only_execution as x,minimal_e2only_contract as c,minimal_e2only_physical as physical
from scripts.pi0294.dec1 import vanda_campaign as queue

@pytest.fixture(scope='module')
def captured():
    root=os.environ.get('ARX5_CURRENT_REMAINING_CAPTURE_DIR')
    if not root:pytest.skip('actual private closed1507480 capture required')
    m=Path(root);gproof=json.loads((m/'QUALIFIED_E1_NATIVE01_SOURCE_COLD_INDEPENDENT.private.json').read_bytes());gb=gzip.decompress(base64.b64decode(gproof['GLOBAL_gzip_base64']));assert e.sha(gb)==gproof['GLOBAL_sha256'];g=json.loads(gb)
    proof=json.loads((m/'E2ONLY_RETRY02_ACTUAL_CLOSED_ZERO_ROOT_SNAPSHOT.private.json').read_bytes())
    files={a:base64.b64decode(proof['files'][b]['base64'])for a,b in
        [('lease_bytes','lease'),('settled_bytes','settled'),('finite_OS_bytes','finite_OS0')]}
    lb=gzip.decompress(base64.b64decode(json.loads((m/'D25_SAME64_WIRE01_PREPARED.private.json').read_bytes())['LOCAL_gzip_base64']));return dict(g=g,gb=gb,lb=lb,files=files)

@pytest.fixture(scope='module')
def proposed(captured):
    a=captured;g=a['g'];now=time.time();zero=x.closed_zero_binding(g,**a['files']);old,row=x.basic(g);custody=deepcopy(old['physical_custody']);custody.update(request_id='00000000-0000-4000-8000-000000000081',submission_nonce='00000000-0000-4000-8000-000000000082',run_uuid=c.run_name(2),closed_zero_predecessor=zero,retained_initial_bytes=old['physical_custody']['retained_initial_bytes']+123456,closed_zero_roots=[dict(root='/CPU_FIXTURE/'+row['run_uuid'],run_uuid=row['run_uuid'],snapshot_file='plans/E2_ONLY_CLOSED_ZERO_'+row['run_uuid']+'.private.json',snapshot_sha256='a'*64,snapshot_identity_sha256='b'*64,manifest_sha256=row['run_manifest_sha256'],lease_sha256=zero['lease_sha256'],finite_OS_path='/CPU_FIXTURE/postF/OS_EXIT.json',allocated_bytes=123456)])
    r=dict(x.fixed_request_fields(2),zero_effect_predecessor=zero,request_id=custody['request_id'],submission_nonce=custody['submission_nonce'],source_sha='a'*40,run_manifest_sha256='b'*64,cold_input_audit_sha256='c'*64,raw_parent_audit_plan_sha256='d'*64,frozen_input_packet_sha256='e'*64,Core_measurement_capture_sha256=g[e.FIELD]['Core_measurement_proof']['capture_sha256'],physical_custody_sha256=e.sha(e.raw(custody)),effective_queue_proof_sha256='f'*64,created_epoch=now,submit_count=len(g['PBS_submissions'])+1,**{k:g[e.authority.FIELD][k]for k in('T_CD','stop_new_training_epoch','deadline_epoch')})
    physical_proof=dict(status='ACTUAL_E2_ONLY_ALL_OLD_GROSS_AND_NEW70GB_NO_PERMISSION',observed_epoch=now,custody_sha256=r['physical_custody_sha256'],previous_snapshot_identity_sha256=custody['previous_snapshot_identity_sha256'],retained_allocated_bytes=custody['retained_initial_bytes'],new_allocated_bytes=1234,next_output_upper_bytes=999,ledger_changed=False,native_permission=False)
    full=queue.collect_queue(runner=lambda *a,**k:SimpleNamespace(returncode=0,stdout='',stderr=''),clock=lambda:now)
    fresh=dict(complete_queue=full,complete_queue_verified=True,observed_epoch=now,GLOBAL_sha256=e.sha(a['gb']),old_workers_terminal=True,old_closed_science_lease_sha256=x.authority.LEASE,independent_confirmation_base64='',closed_zero_predecessor=zero)
    args=dict(request_bytes=e.raw(r),physical_custody=custody,physical_proof=physical_proof,physical_observed_epoch=now,fresh_readback=fresh,now_epoch=now,closed_zero_files=a['files'])
    result=x.register(a['gb'],a['lb'],**args);return dict(result=result,args=args,request=r,zero=zero,old=old,now=now)

def test_real_closed_zero_join_and_distinct_request_scope(captured,proposed):
    a=proposed;g=a['result']['proposed_global'];before=captured['g'];assert g['PBS_submissions'][:-1]==before['PBS_submissions'];assert g[x.FIELD]['prior_attempts']==[before[x.FIELD]];assert g[x.authority.FIELD]==before[x.authority.FIELD];assert g[e.package.FIELD]==before[e.package.FIELD];assert g['CPU_s']==before['CPU_s']+3600;assert g['optimizer_updates']==before['optimizer_updates'];assert a['result']['full_future_proof']['cap_changes_required']=={};assert a['result']['full_future_proof']['dimensions']['PBS_submissions']['cap']is None
    assert x.basic(g,a['zero']['request_id'])[0]==before[x.FIELD]
    assert x.basic(g)[1]['run_uuid']==c.run_name(2);assert len(x.rows(g))==2
    bound=x.bind(g,a['request']['request_id'],'999998.stdct-mgmt-02',now_epoch=a['now']+1);assert x.basic(bound)[1]['job_id']=='999998.stdct-mgmt-02'
    from scripts.pi0294.generalization import minimal_execution_adapter as adapters
    assert adapters.registered(bound,a['zero']['request_id'])[2]==before[x.FIELD]

@pytest.mark.parametrize('key',['lease_bytes','settled_bytes','finite_OS_bytes'])
def test_zero_replacement_requires_all_actual_closed_file_identities(captured,key):
    files=deepcopy(captured['files']);v=json.loads(files[key])
    if key=='lease_bytes':v['status']='ACTIVE'
    if key=='settled_bytes':v['native']['Exit_status']=99
    if key=='finite_OS_bytes':v['returncode']=1
    files[key]=e.raw(v)
    with pytest.raises(PermissionError):x.closed_zero_binding(captured['g'],**files)

def test_cannot_reuse_old_request_or_forge_a_partial_training_zero(captured,proposed):
    g=deepcopy(captured['g']);g['PBS_submissions'][-1]['committed_optimizer_updates']=1
    with pytest.raises(PermissionError):x.closed_zero_binding(g,**captured['files'])
    args=deepcopy(proposed['args']);r=json.loads(args['request_bytes']);r['request_id']=proposed['zero']['request_id'];args['request_bytes']=e.raw(r)
    with pytest.raises(PermissionError):x.register(captured['gb'],captured['lb'],**args)

def test_prior_zero_gross_storage_is_mandatory(captured,proposed):
    args=deepcopy(proposed['args']);del args['physical_custody']['closed_zero_roots']
    with pytest.raises((PermissionError,KeyError)):physical.validate_custody(args['physical_custody'],proposed['request'],predecessor_binding=c.binding())


def test_retry_reuses_same_three_original_phase_contract(proposed):
    from scripts.pi0294.generalization import minimal_session_gate as gates,minimal_execution_adapter as adapters
    r=proposed['request'];assert adapters.request(r)is x
    assert r['session_plan']==c.sessions()and r['recovery_checkpoint_policy']==c.recovery_policy()
    assert [v['phase']for v in c.tickets()]==['MODEL_RAW','FIRST_UPDATE','OFFICIAL_EXPORT']
    c.check_phase(c.sessions()[0]['id'],'MODEL_RAW',dict(optimizer_updates=0,generation_queries=0,model_loads=0))
    c.check_phase(c.sessions()[0]['id'],'FIRST_UPDATE',dict(optimizer_updates=0,generation_queries=0,model_loads=1))
    c.check_phase(c.sessions()[0]['id'],'OFFICIAL_EXPORT',dict(optimizer_updates=512,generation_queries=36,model_loads=1))
    assert gates.e2only(r)


def test_repeated_pure_LOCAL_projection_does_not_double_charge_retry(proposed,captured):
    g=proposed['result']['proposed_global'];l=json.loads(captured['lb']);once=x.budget_LOCAL(g,l);twice=x.budget_LOCAL(g,once)
    assert once==twice
    assert once['CPU_s']['local_PC']==l['CPU_s']['local_PC']+2*x.CONTROL['local_CPU_s']


@pytest.mark.parametrize('success',[False,True])
def test_original_terminal_settles_only_new_attempt_and_preserves_old_zero(captured,proposed,success):
    from scripts.pi0294.test_minimal_e2only import terminal_packet
    a=proposed;g=x.bind(a['result']['proposed_global'],a['request']['request_id'],'999998.stdct-mgmt-02',now_epoch=a['now']+1)
    packet=dict(bound=g,now=a['now']);row,native,effects,audit=terminal_packet(packet,success)
    args=a['args'];out=x.settle(g,row['request_id'],native=native,effects=effects,audit=audit,physical_proof=args['physical_proof'],physical_observed_epoch=args['physical_observed_epoch'],now_epoch=a['now']+12)
    assert out['PBS_submissions'][:-1]==captured['g']['PBS_submissions']
    assert out[x.FIELD]['prior_attempts']==[captured['g'][x.FIELD]]
    assert out['optimizer_updates']==captured['g']['optimizer_updates']+(512 if success else 122)
    assert x.basic(out)[1]['scientific_endpoint_complete']is success
    with pytest.raises(PermissionError):x.settle(out,row['request_id'],native=native,effects=effects,audit=audit,physical_proof=args['physical_proof'],physical_observed_epoch=args['physical_observed_epoch'],now_epoch=a['now']+13)


def test_paid_control_precedes_real_owner_and_is_not_charged_at_registration_again(captured,proposed):
    a=proposed;r=a['request'];args=a['args'];now=a['now'];proof=dict(GLOBAL_sha256=e.sha(captured['gb']),closed_zero_predecessor=a['zero'],no_existing_new_owner_or_dispatch=True,observed_epoch=now)
    with pytest.raises(PermissionError):x.require_paid_retry(captured['g'],r)
    paid=x.propose_retry_control(captured['gb'],captured['lb'],request_bytes=e.raw(r),closed_zero_files=captured['files'],closed_readback=proof,now_epoch=now)
    g=paid['proposed_global'];x.require_paid_retry(g,r);assert g['CPU_s']==captured['g']['CPU_s']+3600;assert g['PBS_submissions']==captured['g']['PBS_submissions'];assert paid['full_future_proof']['cap_changes_required']=={}
    args=deepcopy(args);args['fresh_readback']['GLOBAL_sha256']=e.sha(e.raw(g));registered=x.register(e.raw(g),captured['lb'],**args)['proposed_global'];assert registered['CPU_s']==g['CPU_s'];assert x.planned_retry(registered)==e._vector({});assert len(registered[x.FIELD]['prior_attempts'])==1
    with pytest.raises(PermissionError):x.propose_retry_control(e.raw(g),captured['lb'],request_bytes=e.raw(r),closed_zero_files=captured['files'],closed_readback=proof,now_epoch=now)


def test_actual_guard_inspection_retains_the_prior_proof_with_a_closed_zero_root(tmp_path,monkeypatch):
    # Exercise the actual inspect path, including the closed-root iteration
    # that follows prior.inspect; no model or fabricated native admission.
    from types import SimpleNamespace
    old=tmp_path/'old';zero=tmp_path/'zero';new=tmp_path/'new';prefix=tmp_path/'prefix'
    for p in(old,zero,new,prefix):p.mkdir()
    (zero/'metadata.json').write_text('{}\n');snapshot=physical.inventory.prefix_inventory(zero)
    prior_proof=dict(prefix_allocated_bytes=10000000000,old_metadata_ceiling=1000000,old_science_allocated_bytes=10000000000,new_allocated_bytes=46671091712,retained_cold_allocated_bytes=0)
    guard=object.__new__(physical.Guard);guard.root=new;guard.old_root=old;guard.original=lambda:None;guard.clock=lambda:1.;guard.request=dict(physical_custody_sha256='placeholder');guard.custody=dict(schema=physical.execution.PHYSICAL_SCHEMA,previous_snapshot_identity_sha256=physical.inventory.identity(physical.inventory.prefix_inventory(old)),retained_initial_bytes=66671091712,paid_output_ceiling=physical.execution.PAID_OUTPUT,new_output_bound_bytes=70000000000)
    guard.zero_roots=[(zero,dict(snapshot_identity_sha256=physical.inventory.identity(snapshot)))];guard.prior=SimpleNamespace(inspect=lambda **_:prior_proof,old_root=prefix,prior=SimpleNamespace(custody=dict(prefix_root=str(prefix)),metadata=[]),custody=dict(retained_cold_roots=[]));guard.request['physical_custody_sha256']=e.sha(e.raw(guard.custody))
    out=guard.inspect(next_output_upper_bytes=1000)
    assert out['retained_allocated_bytes']==sum(prior_proof.values())+snapshot['allocated_bytes']
    assert out['next_output_upper_bytes']==1000 and out['ledger_changed']is False
