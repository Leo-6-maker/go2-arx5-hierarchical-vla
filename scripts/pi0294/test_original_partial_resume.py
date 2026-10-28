"""Captured actual eleven outcomes; new C13 process/source remain CPU fixtures."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_core_partial_resume as p,mix_core_accounting as c,mix_accounting as a,mix_core_admission as admission,mix_core_settlement as terminal

@pytest.fixture
def packet():
 path=os.environ.get('PV1_C13_CPU_PACKET')
 if not path:pytest.skip('Captured actual complete C13 cold packet required')
 return json.loads(Path(path).read_bytes())

@pytest.fixture
def resources():
 path=os.environ.get('PV1_C13_RESOURCE_CAPTURE')
 if not path:pytest.skip('Captured original post600 raw resource inputs required')
 value=json.loads(Path(path).read_bytes())
 return {k:base64.b64decode(v)if k.endswith('_bytes')else v for k,v in value['kwargs'].items()}

def test_original_stage_only_moves_thirteen_and_keeps_all_old_costs(packet):
 obj=packet['assembly'];r=obj['record'];g=packet['registered']['proposed_global'];l=packet['registered']['proposed_LOCAL']
 assert r['remaining_conditions']==13 and r['reserved_GPU_s']==18060 and r['CPU_upper_s']==40000
 assert r['additional_output_upper_bytes']==4*1024**3
 assert g['used']==packet['before_global']['used'] and g['PBS_submissions']==packet['before_global']['PBS_submissions']
 assert g[c.FIELD][p.prior.FAILED]==packet['before_global'][c.FIELD][p.prior.FAILED]
 assert g[a.PACKAGE]['completed_stages']==['TRAIN:C50:1024','TRAIN:D25:1024']
 assert l['V5_CD_Core_CPU_substages'][c.key('C50')]['policy_episodes']==13
 with pytest.raises(PermissionError):c.register(g,l,r,base64.b64decode(obj['rows']['core']['manifest_bytes_base64']),terminal_capture={k:base64.b64decode(v)for k,v in packet['terminal'].items()},now_epoch=r['created_epoch']+1)

@pytest.mark.parametrize('key,value',[('remaining_conditions',24),('CPU_upper_s',80000),('query_limit',2097),('walltime_s',28800),('evaluation_candidate','D25'),('partial_panel_history_sha256','0'*64)])
def test_caller_cannot_expand_or_change_partial_identity(packet,key,value):
 r=deepcopy(packet['assembly']['record']);r[key]=value
 with pytest.raises(PermissionError):c.validate_row(r)

def test_original_post600_full_recompute_joins_remainder_and_future(resources):
 result=admission.recompute(**resources)
 assert result['shortfalls']=={} and result['allocation']['allocation_end_epoch']-result['allocation']['allocation_start_epoch']==18000
 assert result['model_permission']is False and result['physics_authorized']is False
 assert result['dimensions']['PBS_submissions']['projected']==39

@pytest.mark.parametrize('bad',['old-eight-hour-unit','stale-device','empty-prior','foreign-GPU','unknown-global'])
def test_original_native_and_unknown_fences_still_reject(resources,bad):
 v=deepcopy(resources)
 key='native_facts_bytes';x=json.loads(v[key])
 if bad=='old-eight-hour-unit':x['native_unit_properties']['RuntimeMaxUSec']='8h'
 elif bad=='stale-device':x['device']['exact_desktop_coexistence']['kernel_read']['observed_epoch']-=900
 elif bad=='foreign-GPU':x['native_GPU_raw']['processes']['stdout']=json.loads(v['request_bytes'])['GPU_uuid']+', 999, independent-work, 9000\n'
 elif bad=='empty-prior':v['prior_bytes']=a.raw([])
 else:
  key='global_bytes';x=json.loads(v[key]);x['unknown_effects']=['EXACT_TEST_UNKNOWN_NOT_CLEARED']
 if bad!='empty-prior':v[key]=a.raw(x)
 with pytest.raises((PermissionError,ValueError,KeyError)):admission.recompute(**v)

def test_same_exact_bytes_cache_never_caches_unknown_or_clock(tmp_path,monkeypatch):
 from scripts.pi0294.pv1 import successor_core_context as context,successor_cd as s
 from scripts.pi0294.test_v5_core_static_binding import bound
 m,l,_=bound();root=tmp_path/'campaign/children/own';root.mkdir(parents=True);path=root.parent.parent/'control/owner/LOCAL_RESOURCE_LEDGER.private.json';path.parent.mkdir(parents=True)
 marker=dict(s.scope(1000.),grant_sha256=s.GRANT_SHA,before_sha256={'GLOBAL':'a'*64,'LOCAL':'b'*64},authorized_storage_caps=s.NEW_STORAGE,old_clock_and_history_preserved=True,scope_only_not_GPU_or_physics_admission=True)
 l.update({s.FIELD:marker,'unknown_effects':[],'pending_updates':[],'pending_queries':[],'V5_CD_execution_package':{'scope_grant_sha256':s.GRANT_SHA}})
 binding='plans/V5_CD_CORE_LEDGER_BINDING.private.json';b=a.raw({'canonical_LOCAL_ledger':str(path)});m['execution']={'sealed_inputs':{binding:a.sha(b)}};m['code']['vla_commit']='a'*40
 mb=a.raw(m);l['V5_CD_Core_CPU_substages'][context.GEN_ALLOCATION]['run_manifest_sha256']=a.sha(mb)
 (root/binding).parent.mkdir();(root/binding).write_bytes(b);(root/'RUN_MANIFEST.json').write_bytes(mb);(root/'V5_CD_SCOPE_BINDING.json').write_bytes(a.raw(marker));path.write_bytes(a.raw(l));context._BINDING_CACHE.clear();monkeypatch.setattr(context.time,'time',lambda:1500.)
 assert context.ledger_binding(root)[0]==marker;assert context.ledger_binding(root)[0]==marker
 l['unknown_effects']=['REAL_DIFFERENT_BYTES'];path.write_bytes(a.raw(l))
 with pytest.raises(PermissionError):context.ledger_binding(root)
 l['unknown_effects']=[];path.write_bytes(a.raw(l));assert context.ledger_binding(root)[0]==marker
 monkeypatch.setattr(context.time,'time',lambda:2001.)
 with pytest.raises(PermissionError):context.ledger_binding(root)


def test_original_terminal_releases_only_the_reduced_current_holds(resources):
 from scripts.pi0294.pv1.successor_core_context import LEDGER_SCHEMA
 g,l=json.loads(resources['global_bytes']),json.loads(resources['local_bytes']);r=json.loads(resources['request_bytes']);row=g[c.FIELD][r['request_id']];process=json.loads(resources['native_facts_bytes']);start=process['started_epoch']
 args=dict(request_id=r['request_id'],native_handle=row['native_handle'])
 for kind,count,effect in [('model_loads',1,'model/0'),('generation_queries',1,'query/0'),('CPU_s',12.5,'CPU/0')]:g=c.charge(g,**args,kind=kind,count=count,effect_id=effect)
 phys=dict(schema=LEDGER_SCHEMA,run_uuid=r['run_uuid'],authority_release=r['authority_release'],Core_CPU_s=350.,main_starts=1,engine_starts=3,live_and_recovery_integrations=200,shadow_integrations=341,reservations={},unknown_effects=[])
 proof=dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_TERMINAL_V1',request_id=r['request_id'],native_handle=row['native_handle'],hostname=row['hostname'],uid=2009,GPU_uuid=r['GPU_uuid'],boot_id=row['boot_id'],pid=row['pid'],proc_start=row['proc_start'],process_state='TERMINAL',request_sha256=row['request_sha256'],submission_nonce=r['submission_nonce'],recorded_process_terminal=True,terminal_inventory_stable=True,exit_evidence_kind='SYSTEMD_EXEC_MAIN_EXIT',native_exit_status=1,finished_epoch=start+700.,observed_epoch=start+701.,conservative_GPU_s=760,conservative_CPU_s=40,new_resident_allocated_upper_bytes=row['additional_output_upper_bytes'])
 effects=dict(schema='PV1_CORE_GPU_NATIVE_DEV_EFFECT_SUMMARY_V1',request_id=r['request_id'],native_handle=row['native_handle'],request_sha256=row['request_sha256'],source_sha=r['source_sha'],run_manifest_sha256=r['run_manifest_sha256'],evaluation_candidate='C50',checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],statistics_sha256=r['statistics_sha256'],optimizer_updates_committed=0,new_physics=0,new_episodes=0,effect_journal_complete=True,query_attempts=1,model_load_attempts=1,unknown_query_intents=[])
 delivery=dict(status='ACTUAL_CLOSED_C50_1024_INFERENCE_VIEW_DELIVERED_NO_MODEL_LOADED',checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],full_resume_files_present=False,new_GPU_model_queries_physics=0,HF_revision='e'*40,files={f'model-{i}.safetensors':{}for i in range(3)})
 kw=dict(request_id=r['request_id'],native_proof=proof,effects=effects,physical_bytes=a.raw(phys),delivery_receipt_bytes=a.raw(delivery),now_epoch=start+702.)
 out=terminal.propose(a.raw(g),a.raw(l),**kw);v,w=out['proposed_global'],out['proposed_LOCAL']
 assert v['reserved']['H200_s']==g['reserved']['H200_s']-18060
 assert v['CPU_reserved_s']==g['CPU_reserved_s']-(40000-12.5)
 assert v['new_storage_current_bytes']==g['new_storage_current_bytes']
 assert v[c.FIELD][p.prior.FAILED]==g[c.FIELD][p.prior.FAILED]
 assert v[a.PACKAGE]['completed_stages']==g[a.PACKAGE]['completed_stages']
 assert w['Core_native_GPU_current_storage_bytes']==l['Core_native_GPU_current_storage_bytes']
 with pytest.raises(PermissionError):terminal.propose(a.raw(v),a.raw(w),**kw)
