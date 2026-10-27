"""Real closed failure bytes and explicit future request fixtures, CPU only."""
import base64,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_export_repair_accounting as r,amd_adoption as a

B=Path(__file__).resolve().parents[4]/'work/private/v5-execution-20261025'
@pytest.fixture
def inputs():
 p=B/'GEN1_VANDA_OFFICIAL_EXPORT_CPU1497552_INDEPENDENT_CLOSED_SPACE_READBACK.private.json'
 if not p.exists():pytest.skip('private actual closed failed-role proof unavailable')
 closed=json.loads(p.read_bytes());g=json.loads(base64.b64decode(closed['GLOBAL_base64']));failed=json.loads((B/'GEN1_VANDA_OFFICIAL_EXPORT_CPU1497552_KNOWN_FAILED_EFFECT_AUDIT_CPU_PREPARED.private.json').read_bytes())['audit'];original=json.loads((B/'GEN1_VANDA_OFFICIAL_EXPORT_PREPARED.private.json').read_bytes());old=r.original.rows(g)[0]
 item=dict(platform_data_version=r.VERSION,native_platform='VANDA_A40',project='CFP04-CF-007',job_name=r.JOB_NAME,physical_GPU_count=0,scope_grant_sha256=a.GRANT_SHA,T_CD=g[a.FIELD]['T_CD'],deadline_epoch=g[a.FIELD]['deadline_epoch'],limits=deepcopy(r.LIMITS),raw_plan_sha256=old['raw_plan_sha256'],closed_capture_proof_sha256=old['closed_capture_proof_sha256'],predecessor_request_id=r.failure.REQUEST,predecessor_job_id=r.failure.JOB,predecessor_manifest_sha256=old['run_manifest_sha256'],source_sha='a'*40,request_sha256='b'*64,run_manifest_sha256='c'*64,predecessor_closure_readback_sha256=a.sha(a.raw(closed)),request_id='11111111-1111-4111-8111-111111111111',submission_nonce='22222222-2222-4222-8222-222222222222',job_id=None,status=r.STATES[0])
 return dict(g=g,item=item,closed_failure=closed,failed_audit=failed,historical_LOCAL_bytes=base64.b64decode(original['historical_LOCAL_base64']),now_epoch=closed['epoch']+1)

def register(p):
 p=deepcopy(p);g=p.pop('g');item=p.pop('item');return r.register(g,item,**p)

def test_actual_full_future_replacement_no_refund_no_second_output_or_control_payment(inputs):
 g=inputs['g'];old=deepcopy(r.original.rows(g)[0]);v=register(inputs);out=v['proposed_global'];proof=v['full_remaining_proof']
 assert not proof['cap_changes_required']and proof['matrix3_and_final20GPUh40GiB_retained']
 assert r.original.rows(out)[0]==old and out['new_storage_current_bytes']==g['new_storage_current_bytes']
 assert out['CPU_s']==g['CPU_s']and out['CPU_reserved_s']==g['CPU_reserved_s']+36000
 assert out['storage_reserved_bytes']==g['storage_reserved_bytes']+64*1024**2
 assert out['reserved']==g['reserved']and out['used']==dict(g['used'],PBS_submissions=g['used']['PBS_submissions']+1)
 assert out[r.original.original.PACKAGE]==g[r.original.original.PACKAGE]
 for k,x in proof['dimensions'].items():assert x['projected']<=x['cap']
 again=dict(inputs,g=out)
 with pytest.raises(PermissionError):register(again)

@pytest.mark.parametrize('bad',['active_old','old_refund','old_lease','old_counter','aggregate','GPU','old_request','new_plan','late','different_GLOBAL'])
def test_no_credit_without_exact_closed_known_zero_and_same_panel(inputs,bad):
 p=deepcopy(inputs)
 if bad=='active_old':p['g']['PBS_submissions'][-1]['status']=r.STATES[1]
 elif bad=='old_refund':p['g']['new_storage_current_bytes']-=10*1024**3
 elif bad=='old_lease':
  lease=json.loads(base64.b64decode(p['closed_failure']['lease_base64']));lease['status']='ACTIVE';b=a.raw(lease);p['closed_failure']['lease_base64']=base64.b64encode(b).decode();p['closed_failure']['lease_sha256']=a.sha(b)
 elif bad=='old_counter':p['closed_failure']['original_UNKNOWN_unchanged']=False
 elif bad=='aggregate':p['closed_failure']['old_output_bytes']=r.original.LIMITS['output_bytes']+1
 elif bad=='GPU':p['item']['physical_GPU_count']=1
 elif bad=='old_request':p['item']['request_id']=r.failure.REQUEST
 elif bad=='new_plan':p['item']['raw_plan_sha256']='f'*64
 elif bad=='late':p['now_epoch']=1793537549.
 else:p['g']['CPU_s']+=1
 p['item']['predecessor_closure_readback_sha256']=a.sha(a.raw(p['closed_failure']))
 with pytest.raises((PermissionError,KeyError)):register(p)

def test_bound_complete_repair_charges_actual_cpu_plus_only_new_metadata(inputs):
 v=register(inputs)['proposed_global'];item=r.rows(v)[0];v=r.bind(v,request_id=item['request_id'],job_id='123.stdct-mgmt-02');before=deepcopy(v)
 native=dict(job_state='F',job_id='123.stdct-mgmt-02',request_id=item['request_id'],submission_nonce=item['submission_nonce'],Exit_status=0,conservative_CPU_s=900.)
 audit=dict(status='PASS_INDEPENDENT_GEN1_CPU_EXPORT_EFFECTS',request_sha256=item['request_sha256'],all_processes_terminal=True,unknown_effects=[],export_finished=True,training_bank_qualified=True,model_loads=0,queries=0,updates=0,constructors=0,physics_ticks=0,episode_starts=0,shared_source_and_repair_output_bytes=2*1024**3)
 out=r.settle(v,request_id=item['request_id'],native=native,audit=audit)
 assert out['CPU_s']==before['CPU_s']+900 and out['new_storage_current_bytes']==before['new_storage_current_bytes']+r.METADATA
 assert r.original.rows(out)==r.original.rows(before)and r.rows(out)[0]['training_bank_qualified']
 with pytest.raises(PermissionError):r.settle(out,request_id=item['request_id'],native=native,audit=audit)

def test_new_CPU_queue_join_retains_old_failed_row_and_real_zero_GPU_identity(inputs):
    from scripts.pi0294.generalization import amd_export_queue as q
    g=register(inputs)['proposed_global'];request=r.rows(g)[0]['request_id'];before=deepcopy(g)
    result=q.view(g,[],request_id=request);assert result['own_CPU_unbound']and not result['unknown']and g==before
    g=r.bind(g,request_id=request,job_id='123.stdct-mgmt-02')
    rows=[dict(platform='VANDA_A40',job_id='123.stdct-mgmt-02',state='R',project='CFP04-CF-007',name=r.JOB_NAME,ngpus=0)]
    result=q.view(g,rows,request_id=request);assert len(result['own'])==1 and not result['unknown']
    rows[0]['ngpus']=1
    with pytest.raises(PermissionError):q.view(g,rows,request_id=request)
