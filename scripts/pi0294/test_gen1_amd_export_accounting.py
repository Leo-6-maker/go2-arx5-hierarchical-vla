"""Whole captured budget plus explicit future closure fixtures, no real PBS."""
import base64,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_export_accounting as e,amd_remaining as r,amd_adoption as a,amd_pilot_scope as scope
from scripts.pi0294.test_gen1_amd_remaining_terminal import inputs as terminal_fixture,B

@pytest.fixture
def inputs():
 g,audit=terminal_fixture.__wrapped__();g=r.settlement(g,audit,now_epoch=audit['epoch']+1)['proposed_global'];old=g[a.FIELD]['teacher_pilot']['engineering_recovery'];ten=old['remaining_conditions'];saved=old['saved_data_reaudit']
 p=json.loads((B/'GEN1_AMD_REMAINING10_PREPARED.private.json').read_bytes());packet=base64.b64decode(p['common']['plans/GEN1_V5_PREVIEW_V2.private.json']);local=base64.b64decode(p['historical_LOCAL_base64'])
 outcomes=deepcopy(ten['outcomes'])
 for row in saved['raw_admitted_cases'].values():outcomes.append(dict(attempt_id=row['attempt_id'],terminal=True,unknown_effects=[],native_safety_failures=0,full_task_success=True,independent_raw_audit='PASS_INDEPENDENT_RAW_AUDIT',raw_and_custody_verified=True))
 leases={}
 for key in ['remaining_owner','remaining_0','remaining_1','R1_owner','R1_0','R1_1','saved_reaudit']:
  manifest='f'*64
  if key in ('remaining_0','remaining_1'):manifest=ten['workers'][key[-1]]['manifest_sha256']
  elif key in('R1_0','R1_1'):manifest=old['workers'][key[-1]]['manifest_sha256']
  elif key=='saved_reaudit':manifest=saved['manifest_sha256']
  leases[key]=dict(status='CLOSED',manifest_sha256=manifest,independent_terminal_audit_sha256=saved['independent_terminal_audit_sha256']if key=='saved_reaudit'else ten['independent_terminal_audit_sha256'],independent_saved_effects_audit_sha256=old['independent_saved_effects_audit_sha256'])
 closed=dict(status='INDEPENDENT_GEN1_ALL_SOURCE_ROLES_CLOSED',remaining_terminal_audit_sha256=ten['independent_terminal_audit_sha256'],saved_reaudit_sha256=saved['independent_terminal_audit_sha256'],all_processes_terminal=True,original_files_and_valid_failures_preserved=True,leases=leases,outcomes=outcomes,fixture=True)
 row=dict(platform_data_version=e.VERSION,native_platform='VANDA_A40',project='CFP04-CF-007',job_name=e.JOB_NAME,physical_GPU_count=0,scope_grant_sha256=a.GRANT_SHA,T_CD=g[a.FIELD]['T_CD'],deadline_epoch=g[a.FIELD]['deadline_epoch'],limits=deepcopy(e.LIMITS),source_sha='a'*40,request_sha256='b'*64,run_manifest_sha256='c'*64,closed_capture_proof_sha256=a.sha(a.raw(closed)),raw_plan_sha256='d'*64,request_id='11111111-1111-4111-8111-111111111111',submission_nonce='22222222-2222-4222-8222-222222222222',status=e.STATES[0],job_id=None)
 return g,closed,row,packet,local,audit['epoch']+2

def call(values):
 g,closed,row,packet,local,now=values
 return e.register(g,row,closed_capture_proof=closed,packet_bytes=packet,historical_LOCAL_bytes=local,now_epoch=now)

def test_full_future_reservation_credits_only_actual_registered_export_once(inputs):
 g,closed,row,packet,local,now=inputs;before=scope.full_forecast(a.raw(g),local);v=call(inputs);after=v['full_remaining_proof']
 assert not after['cap_changes_required']and v['original_C_D_package_unchanged']and not v['native_admission']
 for key in('PBS_submissions','Hopper_CPU_s','Hopper_storage_bytes'):
  assert after['dimensions'][key]['projected']==pytest.approx(before['dimensions'][key]['projected'])
 assert v['proposed_global']['used']['PBS_submissions']==g['used']['PBS_submissions']+1
 assert e.rows(v['proposed_global'])[0]['job_id']is None
 with pytest.raises(PermissionError):e.register(v['proposed_global'],row,closed_capture_proof=closed,packet_bytes=packet,historical_LOCAL_bytes=local,now_epoch=now)

@pytest.mark.parametrize('bad',['active_capture','missing_raw','unsafe','unknown','active_lease','wrong_lease','GPU','LOCAL','late','request'])
def test_no_new_export_from_preparation_unknown_or_wrong_native_identity(inputs,bad):
 values=deepcopy(inputs);g,closed,row,packet,local,now=values
 if bad=='active_capture':g[a.FIELD]['teacher_pilot']['engineering_recovery']['remaining_conditions']['status']=r.STATUS
 elif bad=='missing_raw':closed['outcomes'].pop()
 elif bad=='unsafe':closed['outcomes'][0]['native_safety_failures']=1
 elif bad=='unknown':closed['outcomes'][0]['unknown_effects']=['unknown']
 elif bad=='active_lease':closed['leases']['remaining_owner']['status']='ACTIVE'
 elif bad=='wrong_lease':closed['leases']['remaining_1']['manifest_sha256']='0'*64
 elif bad=='GPU':row['physical_GPU_count']=1
 elif bad=='LOCAL':values=values[:4]+(b'{}',)+values[5:]
 elif bad=='late':values=values[:5]+(1793537550.,)
 else:row['job_id']='123.stdct-mgmt-02'
 row['closed_capture_proof_sha256']=a.sha(a.raw(closed))
 with pytest.raises((PermissionError,KeyError)):call(values)

def test_real_reply_binding_and_zero_gpu_terminal_do_not_grant_training(inputs):
 g,closed,row,packet,local,now=inputs;g=call(inputs)['proposed_global'];request=row['request_id'];g=e.bind(g,request_id=request,job_id='123.stdct-mgmt-02')
 with pytest.raises(PermissionError):e.bind(g,request_id=request,job_id='123.stdct-mgmt-02')
 audit=dict(status='PASS_INDEPENDENT_GEN1_CPU_EXPORT_EFFECTS',request_sha256=row['request_sha256'],all_processes_terminal=True,unknown_effects=[],export_finished=True,model_loads=0,queries=0,updates=0,constructors=0,physics_ticks=0,episode_starts=0,training_bank_qualified=False)
 native=dict(job_state='F',job_id='123.stdct-mgmt-02',request_id=request,submission_nonce=row['submission_nonce'],Exit_status=0,conservative_CPU_s=123.5)
 v=e.settle(g,request_id=request,native=native,audit=audit);saved=e.rows(v)[0];assert saved['export_finished']and not saved['training_bank_qualified']
 assert v['CPU_s']==g['CPU_s']+123.5 and v['reserved']==g['reserved']and v[scope.a.PACKAGE]==g[scope.a.PACKAGE]
 assert v['new_storage_current_bytes']==g['new_storage_current_bytes']+10*1024**3
 with pytest.raises(PermissionError):e.settle(v,request_id=request,native=native,audit=audit)
