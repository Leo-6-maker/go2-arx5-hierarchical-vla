"""Synthetic CPU resource/intent proposals, never native authorization or IO."""
from copy import deepcopy
import json
import os
from unittest.mock import patch

import pytest

from scripts.pi0294.pv1 import evaluation_admission as a
from scripts.pi0294.pv1 import evaluation_accounting as accounting
from scripts.pi0294.pv1 import evaluation_platform as e
from scripts.pi0294.pv1.evaluation_work import remaining
from scripts.pi0294.test_pv1_a40_evaluation_identity import case,proof
from scripts.pi0294.test_pv1_neural_ledger import value
from scripts.pi0294.test_pv1_v4_evaluation_work import POP,evidence,synthetic_population_only
from scripts.pi0294.pv1 import evaluation_ledger
from scripts.pi0294.dec1 import source_cpu_once_io as once
from scripts.pi0294.dec1.vanda_campaign import QUEUE


def raw(value):return a.canonical(value)


def rebuild_snapshot(i,*,now=None):
    g=json.loads(i['global_bytes']);local=json.loads(i['local_bytes'])
    s=json.loads(i['snapshot_bytes']);u,r=a.resource_floor(g,local)
    u.update(Core_storage_bytes=s['Core_storage_upper_bytes'],local_storage_bytes=s['LOCAL_storage_upper_bytes'])
    s.update(global_ledger_sha256=a.hashlib.sha256(i['global_bytes']).hexdigest(),
        LOCAL_ledger_sha256=a.hashlib.sha256(i['local_bytes']).hexdigest(),used=u,reserved=r)
    if now is not None:s['observed_epoch']=now
    i['snapshot_bytes']=raw(s)


def inputs(c):
    histories=dict(B0_CORR=evidence(count=14),A_CONTROL=evidence('A_CONTROL',24),B_COVERAGE=evidence('B_COVERAGE',48))
    work=remaining(POP,evidence_by_candidate=histories)
    r=deepcopy(c['r']);r.update(submit_count=1,noise_bank=294102,remaining_conditions=24,
        created_epoch=c['a']['allocation_start_epoch']-10,
        new_bank_manifest_sha256=work['new_bank_manifest_sha256'],scientific_admission_version=e.VERSION,
        CPU_upper_s=80000,CPU_envelope_profile=accounting.BOUNDED_CPU_PROFILE,declared_process_CPU_limit_s=79760,
        cold_input_audit_sha256='a'*64,checkpoint_reuse_proof_sha256='b'*64)
    c['now']=c['a']['allocation_start_epoch']+20;c['r']=r
    job=next(j for j in work['jobs'] if j['candidate']=='A_CONTROL')
    r.update(planned_panel_sha256=a.digest(job),prior_outcomes_sha256=a.digest(work['histories']['A_CONTROL']['prior_rows']),
        additional_output_upper_bytes=job['GPU_site_output_upper_bytes'])
    g=value();g.update(campaign_run_uuid=a.CAMPAIGN,CPU_reserved_s=0,new_storage_current_bytes=0,
        new_storage_peak_bytes=0,storage_reserved_bytes=0)
    local=dict(status='ACTIVE_CONTROL_PLANE',run_uuid=a.CAMPAIGN,T0_epoch=e.T0,hard_closeout_epoch=e.HARD_CLOSE,
        CPU_s=dict(CoreLab=0,Hopper=0,local_PC=0),CPU_reserved_s=dict(CoreLab=0,Hopper=0,local_PC=0),
        GPU_allocated_s=0,GPU_reserved_s=0,PBS_submissions=[],optimizer_updates=0,neural_queries=0,
        episode_starts=0,engine_starts=0,physics_ticks=dict(live=0,shadow=0,closed_source_component_financial_upper=0),
        network_transfer_accounted_bytes=0,unknown_effects=[],Hopper_unknown_effects=[],pending_updates=[],pending_queries=[])
    s=dict(schema=a.SNAPSHOT,campaign_run_uuid=a.CAMPAIGN,original_T0_epoch=e.T0,
        hard_close_epoch=e.HARD_CLOSE,observed_epoch=c['now'],Core_storage_upper_bytes=1024**3,
        LOCAL_storage_upper_bytes=1024**3,
        primary_attempt_inventory={candidate:list(h['all_actual_attempts']) for candidate,h in work['histories'].items()})
    i=dict(manifest_bytes=(c['root']/'RUN_MANIFEST.json').read_bytes(),request_bytes=raw(r),population_bytes=POP,
        evidence_by_candidate=histories,global_bytes=raw(g),local_bytes=raw(local),snapshot_bytes=raw(s),now_epoch=c['now'])
    rebuild_snapshot(i);return i


def prepared(i):return a.recompute(**i,purpose='PREPARE')


def sealed_for_reserve(i):
    proof_=prepared(i);r=json.loads(i['request_bytes']);r['prequeue_future_admission_sha256']=proof_['prequeue_context_sha256']
    i['request_bytes']=raw(r);return i


def native(c,i):
    i=sealed_for_reserve(i)
    proposal=a.reserve_proposal(i['global_bytes'],recompute_inputs=i)['proposed_global']
    request=json.loads(i['request_bytes'])
    proposal=accounting.bind_job(proposal,request_id=request['request_id'],job_id=c['a']['job_id'],now_epoch=c['now'])
    i['global_bytes']=raw(proposal);rebuild_snapshot(i)
    return i


def site(i):
    r=json.loads(i['request_bytes']);now=i['now_epoch']
    return dict(schema='PI0294_V4_NATIVE_PREQUEUE_SITE_V1',
        source='VANDA_LOGIN_NATIVE_AMGR_QUEUE_AND_QUOTA_READBACK',request_id=r['request_id'],
        submission_nonce=r['submission_nonce'],request_sha256=a.hashlib.sha256(i['request_bytes']).hexdigest(),
        observed_epoch=now,queue_proof=dict(schema=QUEUE,user='e1192548',observed_epoch=now,
            qselect=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=0,stdout='',stderr=''),qstat=None),
        project=dict(name=r['project'],active=True,users=['e1192548'],cluster=['stdct-mgmt-02'],
            start_date='2026-10-01',end_date='2026-12-31'),balance_GPU_hours=1000,balance_CPU_hours=10000,
        effective_queue_limit=dict(source='VANDA_NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS',requested_queue='auto',
            destination='batch_gpu',maximum_wall_s=28800,observed_epoch=now),
        quota=dict(user='e1192548',scope='OWN_PROJECT_AND_FILESYSTEM_MINIMUM',observed_epoch=now,available_bytes=1024**4))


def materialize(c,i):
    from scripts.pi0294.test_pv1_a40_evaluation_identity import put
    old=c['root'];root=old.parent/'campaign/children'/old.name;root.parent.mkdir(parents=True);old.rename(root)
    directory=root/'pbs/alloc-01';(root/'pbs/alloc-24').rename(directory)
    c['root']=root;c['directory']=directory
    m=json.loads(i['manifest_bytes']);index=dict(schema='PI0294_V4_PRIMARY_HISTORY_INPUT_INDEX_V1',
        campaign_run_uuid=a.CAMPAIGN,authority_release=e.AUTHORITY,population='plans/POPULATION.json',evidence={})
    originals={index['population']:i['population_bytes']}
    for candidate,items in i['evidence_by_candidate'].items():
        index['evidence'][candidate]=[]
        for number,item in enumerate(items):
            paths={k:f'plans/history/{candidate}/{number}/{k}.json' for k in item}
            originals.update({paths[k]:v for k,v in item.items()});index['evidence'][candidate].append(paths)
    originals[a.INDEX]=raw(index)
    for name,data in originals.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(0o400)
    sealed={n:a.hashlib.sha256(v).hexdigest() for n,v in originals.items()}
    m['execution']['sealed_inputs']=sealed;put(root/'RUN_MANIFEST.json',m)
    i['manifest_bytes']=(root/'RUN_MANIFEST.json').read_bytes()
    r=json.loads(i['request_bytes']);r.update(sealed_inputs=sealed,run_manifest_sha256=a.hashlib.sha256(i['manifest_bytes']).hexdigest())
    i['request_bytes']=raw(r);i=sealed_for_reserve(i)
    request=directory/'REQUEST.json';put(request,json.loads(i['request_bytes']))
    # Preserve the exact final request bytes produced by the fixture's encoder.
    request.chmod(0o600);request.write_bytes(i['request_bytes']);request.chmod(0o400)
    for name,key in ((a.FRESH_LOCAL,'local_bytes'),(a.FRESH_SNAPSHOT,'snapshot_bytes')):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(i[key]);p.chmod(0o400)
    p=root/a.FRESH_SITE;p.write_bytes(raw(site(i)));p.chmod(0o400)
    ledger=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';ledger.write_bytes(i['global_bytes'])
    return root,request,ledger,i


def test_full_independent_dev_works_with_partial_b0_and_preserves_original_protection(case):
    i=inputs(case);before=deepcopy(i);p=prepared(i)
    assert i==before and p['all_remaining_costs_fit'] is True
    assert p['native_or_scientific_admission'] is False and p['current_credit']['H200_s']==0
    assert p['dimensions']['PBS_submissions']['remaining_S2']==3
    assert p['dimensions']['H200_s']['matrix']==86580 and p['dimensions']['H200_s']['final']==72000
    assert len(i['evidence_by_candidate']['B0_CORR'])==14


def test_exact_prequeue_digest_has_no_request_hash_cycle_and_never_writes(case):
    i=inputs(case);p=prepared(i);i=sealed_for_reserve(i);before=deepcopy(i)
    q=a.reserve_proposal(i['global_bytes'],recompute_inputs=i)
    assert i==before and q['revision_increment_or_IO_performed'] is False and q['qsub_performed'] is False
    row=q['proposed_global']['PBS_submissions'][0]
    assert row['job_id'] is None and row['status']==accounting.STATES[0]
    assert row['prequeue_future_admission_sha256']==p['prequeue_context_sha256']
    assert row['request_sha256']==a.hashlib.sha256(i['request_bytes']).hexdigest()
    assert row['charges_DEC1_subcaps'] is False and q['proposed_global']['reserved']['optimizer_updates']==0
    assert q['proposed_global']['reserved']['H200_s']==28860
    assert q['proposed_global']['CPU_reserved_s']==80000


def test_exact_current_escrow_and_predebited_queries_cpu_are_credited_once(case):
    c=case;i=inputs(c);before=prepared(i);i=native(c,i)
    p=a.recompute(**i,purpose='NATIVE_PREMODEL',queue_proof=proof(c))
    for key in ('H200_s','PBS_submissions','neural_queries','Hopper_CPU_s','Hopper_storage_bytes'):
        assert p['dimensions'][key]['projected']==before['dimensions'][key]['projected']
    g=json.loads(i['global_bytes']);r=json.loads(i['request_bytes'])
    for kind,count,effect in [('model_loads',1,'load'),('generation_queries',1,'query'),('CPU_s',7,'cpu')]:
        g=accounting.charge(g,request_id=r['request_id'],job_id=c['a']['job_id'],kind=kind,count=count,effect_id=effect)
    i['global_bytes']=raw(g);rebuild_snapshot(i)
    q=a.recompute(**i,purpose='NATIVE_PREMODEL',queue_proof=proof(c))
    assert q['dimensions']==p['dimensions'] or all(q['dimensions'][k]['projected']==p['dimensions'][k]['projected'] for k in p['dimensions'])
    assert q['control_prepayment_credit_applied'] is False


@pytest.mark.parametrize('fault',['unknown','pending','LOCAL_unknown','LOCAL_pending','stale','future',
    'global_hash','local_hash','T0','clock','missing_attempt','extra_attempt','duplicate_attempt',
    'wrong_panel','wrong_prior','wrong_bank','training_version','short_CPU','dropped_old_tick',
    'dropped_Core_hold','missing_storage','caller_caps','unsealed_context','reused_request'])
def test_no_cost_drop_inventory_omission_stale_proof_or_caller_permission(case,fault):
    i=inputs(case);s=json.loads(i['snapshot_bytes']);r=json.loads(i['request_bytes']);g=json.loads(i['global_bytes']);local=json.loads(i['local_bytes'])
    expected_error=True
    if fault=='unknown':g['unknown_effects']=['PRESERVED_UNKNOWN']
    if fault=='pending':g['pending_updates']=['PRESERVED_PENDING']
    if fault=='LOCAL_unknown':local['unknown_effects']=['UNKNOWN']
    if fault=='LOCAL_pending':local['pending_queries']=['QUERY_UNKNOWN']
    if fault=='stale':s['observed_epoch']-=121
    if fault=='future':s['observed_epoch']+=1
    if fault=='global_hash':s['global_ledger_sha256']='f'*64
    if fault=='local_hash':s['LOCAL_ledger_sha256']='f'*64
    if fault=='T0':s['original_T0_epoch']+=1
    if fault=='clock':i['now_epoch']=e.STOP_NEW
    if fault=='missing_attempt':s['primary_attempt_inventory']['B0_CORR'].pop()
    if fault=='extra_attempt':s['primary_attempt_inventory']['A_CONTROL'].append('omitted-actual')
    if fault=='duplicate_attempt':s['primary_attempt_inventory']['A_CONTROL']*=2
    if fault=='wrong_panel':r['planned_panel_sha256']='f'*64
    if fault=='wrong_prior':r['prior_outcomes_sha256']='f'*64
    if fault=='wrong_bank':r['new_bank_manifest_sha256']='f'*64
    if fault=='training_version':r['scientific_admission_version']='DEC1_SOURCE_TRAIN_STAGE_V1'
    if fault=='short_CPU':r['CPU_upper_s']=54000
    if fault=='dropped_old_tick':local['physics_ticks']['closed_source_component_financial_upper']=16490
    if fault=='dropped_Core_hold':local['source_cpu_reserved']=dict(CoreLab_CPU_s=1200,episode_starts=4,physics_ticks=16000)
    if fault=='missing_storage':s['Core_storage_upper_bytes']=0
    if fault=='caller_caps':
        # A caller's purported high cap cannot alter adopted original limits.
        s['used']['Hopper_CPU_s']=576000;s['caps']=dict(Hopper_CPU_s=999999999);expected_error=False
    if fault=='unsealed_context':r['prequeue_future_admission_sha256']='f'*64
    if fault=='reused_request':
        i=sealed_for_reserve(i);proposal=a.reserve_proposal(i['global_bytes'],recompute_inputs=i)['proposed_global']
        g=proposal;r=json.loads(i['request_bytes'])
    i.update(request_bytes=raw(r),global_bytes=raw(g),local_bytes=raw(local),snapshot_bytes=raw(s))
    if fault in ('unknown','pending','LOCAL_unknown','LOCAL_pending','dropped_old_tick','dropped_Core_hold','reused_request'):
        s['global_ledger_sha256']=a.hashlib.sha256(i['global_bytes']).hexdigest();s['LOCAL_ledger_sha256']=a.hashlib.sha256(i['local_bytes']).hexdigest();i['snapshot_bytes']=raw(s)
    before=deepcopy(i)
    if fault=='unsealed_context':
        with pytest.raises(PermissionError):a.reserve_proposal(i['global_bytes'],recompute_inputs=i)
    elif expected_error:
        with pytest.raises(PermissionError):prepared(i)
    else:
        p=prepared(i);assert p['shortfalls']['Hopper_CPU_s']>0 and p['dimensions']['Hopper_CPU_s']['cap']==576000
    assert i==before


@pytest.mark.parametrize('fault',['unbound','wrong_job','wrong_nonce','stale_queue','incomplete_queue','terminal','another_request'])
def test_current_credit_requires_exact_bound_live_job_never_unknown_retry(case,fault):
    c=case;i=native(c,inputs(c));g=json.loads(i['global_bytes']);r=g['PBS_submissions'][0];p=proof(c)
    if fault=='unbound':r.update(status=accounting.STATES[0],job_id=None)
    if fault=='wrong_job':r['job_id']='987.stdct-mgmt-02'
    if fault=='wrong_nonce':c['j']['Variable_List']['PV1_SUBMISSION_NONCE']='other';p=proof(c)
    if fault=='stale_queue':p['observed_epoch']-=121
    if fault=='incomplete_queue':p['qstat']['stdout']='{"Jobs":{}}'
    if fault=='terminal':r['status']='NATIVE_TERMINAL'
    if fault=='another_request':r['request_id']='22345678-1234-4234-8234-123456789abc'
    i['global_bytes']=raw(g);rebuild_snapshot(i);before=deepcopy(i)
    with pytest.raises((PermissionError,KeyError)):a.recompute(**i,purpose='NATIVE_PREMODEL',queue_proof=p)
    assert i==before


def test_registered_files_and_original_once_io_commit_one_global_intent_only(case):
    c=case;root,request,ledger,i=materialize(c,inputs(c));m=json.loads(i['manifest_bytes'])
    loaded=a.file_inputs(root,request,now_epoch=i['now_epoch'])
    assert loaded==i
    # Only cold runtime registration/time are fixture-owned. Actual file,
    # proposal, resource, site, locking, CAS and readonly reconciliation run.
    with patch('scripts.pi0294.pv1.neural_scope.validate',return_value=(m,{},{},{},{})),\
         patch('time.time',return_value=i['now_epoch']),patch.dict(os.environ,dict(PBS_JOBID='')):
        done=evaluation_ledger.reserve_once(root,request)
        assert done['qsub_performed'] is False
        row=json.loads(ledger.read_bytes())['PBS_submissions'][0]
        assert row['status']==accounting.STATES[0] and row['job_id'] is None
        before=ledger.read_bytes()
        with pytest.raises((PermissionError,FileExistsError)):evaluation_ledger.reserve_once(root,request)
        assert ledger.read_bytes()==before
        with pytest.raises(PermissionError,match='NATIVE_CLOCK_ONLY'):
            evaluation_ledger.reserve_once(root,request,now_epoch=i['now_epoch'])
    rid=json.loads(i['request_bytes'])['request_id'];base=ledger.parent
    reconciled=once.reconcile(ledger,intent_path=base/('V4-DEV-'+rid+'.INTENT.private.json'),
        receipt_path=base/('V4-DEV-'+rid+'.RECEIPT.private.json'))
    assert reconciled['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
    assert reconciled['automatic_retry_allowed'] is False


@pytest.mark.parametrize('fault',['changed_history','missing_seal','aliased_history','writable_fresh'])
def test_registered_root_cannot_swap_or_omit_original_evidence(case,fault):
    c=case;root,request,ledger,i=materialize(c,inputs(c));before=ledger.read_bytes()
    index=json.loads((root/a.INDEX).read_bytes());name=index['evidence']['B0_CORR'][0]['result']
    if fault=='changed_history':
        p=root/name;p.chmod(0o600);p.write_bytes(b'{}');p.chmod(0o400)
    if fault=='missing_seal':
        from scripts.pi0294.test_pv1_a40_evaluation_identity import put
        r=json.loads(request.read_bytes());r['sealed_inputs'].pop(name);put(request,r)
    if fault=='aliased_history':
        from scripts.pi0294.test_pv1_a40_evaluation_identity import put
        index['evidence']['B0_CORR'][0]['result']=index['evidence']['B0_CORR'][0]['audit'];put(root/a.INDEX,index)
    if fault=='writable_fresh':(root/a.FRESH_SNAPSHOT).chmod(0o600)
    with pytest.raises(PermissionError):a.file_inputs(root,request,now_epoch=i['now_epoch'])
    assert ledger.read_bytes()==before


@pytest.mark.parametrize('fault',['nonce','stale','project','dates','GPU_balance','CPU_balance','quota',
    'queue_failure','foreign_GPU','queue_wall','old_queue_limit'])
def test_site_input_validation_is_required_before_resource_metadata_effect(case,fault):
    c=case;i=sealed_for_reserve(inputs(c));p=a.recompute(**i,purpose='RESERVE');s=site(i)
    if fault=='nonce':s['submission_nonce']='other'
    if fault=='stale':s['observed_epoch']-=121
    if fault=='project':s['project']['name']='other'
    if fault=='dates':s['project']['end_date']='2026-01-01'
    if fault=='GPU_balance':s['balance_GPU_hours']=0
    if fault=='CPU_balance':s['balance_CPU_hours']=10
    if fault=='quota':s['quota']['available_bytes']=1
    if fault=='queue_failure':s['queue_proof']['qselect']['returncode']=162
    if fault=='foreign_GPU':
        q=proof(c);c['j']['Job_Name']='unrelated';q=proof(c);s['queue_proof']=q
    if fault=='queue_wall':s['effective_queue_limit']['maximum_wall_s']=100
    if fault=='old_queue_limit':s['effective_queue_limit']['observed_epoch']-=121
    with pytest.raises(PermissionError):a.prequeue_site(raw(s),request_bytes=i['request_bytes'],
        global_bytes=i['global_bytes'],resource_proof=p,now_epoch=i['now_epoch'])


def test_once_transport_unknown_retains_intent_and_never_reissues_write(case):
    c=case;root,request,ledger,i=materialize(c,inputs(c));m=json.loads(i['manifest_bytes']);before=ledger.read_bytes()
    with patch('scripts.pi0294.pv1.neural_scope.validate',return_value=(m,{},{},{},{})),\
         patch('time.time',return_value=i['now_epoch']),patch.dict(os.environ,dict(PBS_JOBID='')):
        with patch.object(once.os,'replace',side_effect=OSError('EXACT_TEMPORARY_TEST_WRITE_UNKNOWN')):
            with pytest.raises(OSError):evaluation_ledger.reserve_once(root,request)
        assert ledger.read_bytes()==before
        rid=json.loads(i['request_bytes'])['request_id'];base=ledger.parent
        intent=base/('V4-DEV-'+rid+'.INTENT.private.json');intent_before=intent.read_bytes()
        receipt=base/('V4-DEV-'+rid+'.RECEIPT.private.json')
        assert intent.exists() and not receipt.exists()
        with pytest.raises(FileExistsError,match='NO_AUTOMATIC_RETRY'):
            evaluation_ledger.reserve_once(root,request)
        assert intent.read_bytes()==intent_before and ledger.read_bytes()==before
        reconciled=once.reconcile(ledger,intent_path=intent,receipt_path=receipt)
        assert reconciled['status']=='CURRENT_OLD_BYTES_NO_RETRY_AUTHORIZATION'
        assert reconciled['automatic_retry_allowed'] is False
