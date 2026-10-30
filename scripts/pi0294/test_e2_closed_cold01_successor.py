"""Actual closed cold01 history; proposed source/role bindings are CPU fixtures."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_accounting as a
from scripts.pi0294.generalization import minimal_e2only_cold2 as n,minimal_e2only_execution as x
from scripts.pi0294.generalization import minimal_e2only_physical as physical,minimal_remaining_concurrency as concurrency

@pytest.fixture(scope='module')
def captured():
 d=os.environ.get('D25_NODE_LOCAL05_PRIVATE_CAPTURE_DIR')
 if not d:pytest.skip('actual closed E2 cold01 capture required')
 D=Path(d)/'MAINTENANCE30_CONTROL_IO';read=lambda f:json.loads((D/f).read_bytes())
 p=read('NODE_LOCAL07FIX_RUNTIME_ASSEMBLY_FINAL_PREPARED.private.json');gb=gzip.decompress(base64.b64decode(p['proposed_GLOBAL_gzip_base64']));lb=gzip.decompress(base64.b64decode(p['proposed_LOCAL_gzip_base64']));g=json.loads(gb)
 capture=read('E2_COLD02_CLOSED01_CAPTURE.private.json');packet_body=gzip.decompress(base64.b64decode(capture['packet_gzip_base64']));assert a.sha(packet_body)==capture['packet_sha256'];packet=json.loads(packet_body)
 source=read('D25_SOURCE07FIX_COPY_PREPARED.private.json');seal=deepcopy(source['seal']);seal['source_sha']='f'*40
 for name in n.TRAINING_FILES:assert seal['files'][name]==n.TRAINING_FILES[name]
 now=capture['observed_epoch']+1
 s=n.prepare_scope(gb,lb,authority_bytes=(D/'DIRECT_AUTHORITY.json').read_bytes(),closed_capture=capture,
   old_request=packet['old_request'],old_manifest=packet['manifest'],old_physical_custody=json.loads(base64.b64decode(packet['sealed_files_base64'][x.PHYSICAL])),source_seal_bytes=a.raw(seal),frozen=source['frozen'],
   source_readback=dict(status='ACTUAL_COMPLETE_COLD_SOURCE_SHA_READBACK',source_sha=seal['source_sha'],source_seal_sha256=a.sha(a.raw(seal)),file_sha256=seal['files'],uid=22182,hostname='stdct-login-01',model_calls=0,physics_ticks=0,observed_epoch=now,CPU_fixture_only=True),now_epoch=now)
 r=deepcopy(packet['old_request']);m=deepcopy(packet['manifest']);p=json.loads(base64.b64decode(packet['sealed_files_base64'][x.PHYSICAL]));oldphys=deepcopy(p)
 p.update(run_uuid=n.RUN);p['retained_initial_bytes']+=capture['record']['allocated_bytes'];p['closed_pre_native_roots'].append(dict(root=packet['snapshot']['root'],closure=capture['record'],snapshot_file=n.SNAPSHOT))
 root=str(Path(m['output_root']).with_name(n.RUN));m.update(run_uuid=n.RUN,output_root=root);m['code'].update(vla_commit=s['source_sha'],source_seal_sha256=s['source_seal_sha256'])
 m['execution']['sealed_inputs'].update({n.BINDING:n.digest(s),n.SNAPSHOT:capture['record']['snapshot_sha256'],x.PHYSICAL:a.sha(a.raw(p)),'SOURCE_SEAL.json':s['source_seal_sha256'],'MIX_OWNER_CONFIG.private.json':'1'*64,'plans/MINIMAL_FROZEN_INPUT_PACKET.private.json':'2'*64,'plans/MINIMAL_CORE_LOCAL_CAPTURE.private.json':'3'*64})
 r.update(source_sha=s['source_sha'],run_uuid=n.RUN,pre_native_cold_generation=2,pre_native_owner_CPU_upper_s=2600,closed_pre_native_role=capture['record'],**{n.TAG:n.digest(s)},physical_custody_sha256=a.sha(a.raw(p)),run_manifest_sha256=a.sha(a.raw(m)),created_epoch=now,frozen_input_packet_sha256='2'*64)
 proposed=n.propose(gb,lb,scope=s,manifest_bytes=a.raw(m),physical_bytes=a.raw(p),request=r)
 return dict(D=D,g=g,gb=gb,lb=lb,scope=s,request=r,manifest=m,physical=p,oldphysical=oldphys,proposed=proposed)


def test_exact_unused_bank_and_native_forecast_are_preserved(captured):
 v=captured;g=v['g'];new=v['proposed']['proposed_global'];m=n.validate(new);p=x.paid_retry_controls(new)[n.RID]
 assert new[x.PREPAID]==g[x.PREPAID]and new['PBS_submissions']==g['PBS_submissions']
 for k in('used','reserved','CPU_s','CPU_reserved_s','new_storage_current_bytes','optimizer_updates','generation_queries','model_loads'):assert new[k]==g[k]
 assert x.planned_retry(new)==x.planned_retry(g)
 assert p['run_uuid']==n.RUN and p['source_sha']=='f'*40 and p['before_CPU_s']==g[x.PREPAID][n.RID]['before_CPU_s']
 assert m['scope']['old_owner_CPU_s_retained']+m['scope']['metadata_CPU_s_retained']+m['scope']['new_owner_CPU_upper_s']==3600
 assert v['proposed']['full_future_proof']['cap_changes_required']=={}and len(v['proposed']['full_future_proof']['dimensions'])==14
 assert not v['proposed']['new_native_permission']
 from scripts.pi0294.generalization.minimal_scope import first512_forecast
 assert first512_forecast(v['gb'],v['lb'])['dimensions']==v['proposed']['full_future_proof']['dimensions']


def test_actual_d25_cold_hold_is_allowed_but_bound_or_changed_roles_are_rejected(captured):
 from scripts.pi0294.pv1 import mix_core_accounting as c
 g=captured['g'];rows=concurrency.cold_Core_rows(g);assert len(rows)==1 and rows[0]['request_id']=='9c771bec-ba9c-4a44-aefa-f424c58f5909'
 for field,value in [('status',c.STATES[1]),('source_sha','a'*40),('query_attempts',1),('request_sha256','0'*64)]:
  bad=deepcopy(g);bad[c.FIELD][rows[0]['request_id']][field]=value
  with pytest.raises(PermissionError):concurrency.cold_Core_rows(bad)


@pytest.mark.parametrize('field,value',[('session_plan',[]),('query_limit',38),('optimizer_update_limit',513),('scheduler_horizon',512),('request_id','00000000-0000-4000-8000-000000000000')])
def test_scientific_or_identity_changes_never_use_this_continuation(captured,field,value):
 r=deepcopy(captured['request']);r[field]=value
 with pytest.raises(PermissionError):n.validate_bound_request(captured['proposed']['proposed_global'],r)


def test_original_physical_parser_requires_both_closed_roles(captured):
 p=deepcopy(captured['physical']);r=captured['request'];physical.validate_custody(p,r,predecessor_binding=x.c.binding())
 p['closed_pre_native_roots'].pop()
 with pytest.raises(PermissionError):physical.validate_custody(p,r,predecessor_binding=x.c.binding())


def test_original_register_and_complete_future_keep_single_native_hold(captured,monkeypatch):
 from scripts.pi0294.dec1 import vanda_campaign as queue
 from scripts.pi0294.pv1 import evaluation_account_jobs
 from scripts.pi0294.pv1.neural_ledger import check
 v=captured;g=v['proposed']['proposed_global'];r=v['request'];p=v['physical'];now=r['created_epoch']
 actual_zero=json.loads((v['D'].parent/'E2ONLY_RETRY02_ACTUAL_CLOSED_ZERO_ROOT_SNAPSHOT.private.json').read_bytes())
 keys={'lease_bytes':'lease','settled_bytes':'settled','finite_OS_bytes':'finite_OS0'}
 zero={k:base64.b64decode(actual_zero['files'][name]['base64'])for k,name in keys.items()}
 monkeypatch.setattr(queue,'queue_rows',lambda *args,**kwargs:[])
 monkeypatch.setattr(queue,'classify',lambda *args:([],[],[]))
 monkeypatch.setattr(evaluation_account_jobs,'checked_outside',lambda *args:[])
 fresh=dict(observed_epoch=now,complete_queue=dict(CPU_fixture_only=True),independent_confirmation_base64=base64.b64encode(b'{}').decode(),GLOBAL_sha256=a.sha(a.raw(g)),complete_queue_verified=True,old_workers_terminal=True,old_closed_science_lease_sha256=x.authority.LEASE,closed_zero_predecessor=r['zero_effect_predecessor'])
 proof=dict(status='ACTUAL_E2_ONLY_ALL_OLD_GROSS_AND_NEW70GB_NO_PERMISSION',observed_epoch=now,custody_sha256=r['physical_custody_sha256'],previous_snapshot_identity_sha256=p['previous_snapshot_identity_sha256'],ledger_changed=False,native_permission=False,retained_allocated_bytes=p['retained_initial_bytes'],new_allocated_bytes=0,next_output_upper_bytes=physical.CONTROL_GROWTH_RESERVE,CPU_fixture_only=True)
 result=x.register(a.raw(g),v['lb'],request_bytes=a.raw(r),physical_custody=p,physical_proof=proof,physical_observed_epoch=now,fresh_readback=fresh,now_epoch=now,closed_zero_files=zero)
 new=result['proposed_global'];check(new);row=next(row for row in new['PBS_submissions']if row['request_id']==n.RID)
 assert row['status']==x.e.STATES[0]and row['job_id']is None and row['pre_native_cold_generation']==2
 assert new['PBS_submissions'][:-1]==g['PBS_submissions']and new['used']['PBS_submissions']==g['used']['PBS_submissions']+1
 assert new['CPU_s']==g['CPU_s']and new[x.PREPAID]==g[x.PREPAID]
 assert new['reserved']['H200_s']==g['reserved']['H200_s']+x.native_resources()['H200_s']
 assert x.planned_retry(new)==x.e._vector({})
 assert result['full_future_proof']['cap_changes_required']=={}


def test_new_metadata_keeps_unrelated_profiles_and_removes_old_stop_fields(captured):
 from scripts.pi0294.generalization import minimal_e2only_cold2_prepare as p
 old=captured['scope']['old_manifest']['output_root']
 binding=dict(run_root=old,role='PI0294_ARX5_E2_ONLY_RAW512_02_COLD01',lease='/project/leases/COLD01.json')
 original=dict(name=x.PROFILE,role=binding['role'],manifest=old+'/RUN_MANIFEST.json',python='/original/python',PBS_required=True)
 other=dict(name='OTHER_HEALTHY_PROFILE',manifest='/healthy/RUN_MANIFEST.json')
 reg=dict(entries=[dict(role=binding['role'],path=old)])
 env=dict(profiles=[original,other]);before=deepcopy((reg,env))
 newreg,newenv,b=p.metadata(reg,env,binding,'a'*64)
 assert(reg,env)==before and newreg['entries'][:-1]==reg['entries']
 assert newenv['profiles'][1]==other and newenv['closed_profile_history']==[original]
 assert newenv['profiles'][0]['python']==original['python']
 assert b['run_root'].endswith('/'+n.RUN)and b['lease'].endswith('COLD02.json')
 oldlease=dict(status='CLOSED',run_uuid=n.OLD_RUN,authority_release='old-release',deadline_epoch=123,
  owner='EXISTING_SOLE_MUTATION_OWNER',original_paid_owner_bank_s=3000,closed_epoch=122,terminal_reason='STOP',STOP_sha256='b'*64)
 newlease=p.lease(oldlease,b)
 assert newlease['status']=='ACTIVE'and newlease['CPU_upper_s']==2600 and newlease['deadline_epoch']==123
 assert not({'closed_epoch','terminal_reason','STOP_sha256'}&set(newlease))and oldlease['status']=='CLOSED'
 for target in('manifest','role'):
  bad=deepcopy(env);bad['profiles'][0][target]='CHANGED_ACTIVE_ROLE'
  with pytest.raises(PermissionError):p.metadata(reg,bad,binding,'a'*64)
 with pytest.raises(PermissionError):p.metadata(newreg,newenv,binding,'a'*64)


def test_changed_environment_requires_exact_original_bytes(captured):
 v=captured;m=deepcopy(v['manifest']);m['execution']['sealed_inputs']['plans/A40_ENVIRONMENT_BINDING.private.json']='e'*64
 for evidence in(None,(b'{}',b'{}')):
  with pytest.raises(PermissionError,match='E2_COLD02_.*ENVIRONMENT'):
   n.propose(v['gb'],v['lb'],scope=v['scope'],manifest_bytes=a.raw(m),physical_bytes=a.raw(v['physical']),
    request=dict(v['request'],run_manifest_sha256=a.sha(a.raw(m))),environment_binding_bytes=evidence)
