"""Actual closed bytes/population, explicit non-native source/profile fixtures."""
import base64,json,os,time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.pi0294.pv1 import mix_vanda_eval_assembly as build,mix_vanda_eval_identity as identity
from scripts.pi0294.pv1 import mix_vanda_cohost as cohost,mix_accounting as a,mix_vanda_eval_accounting as accounting

workspace=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE',str(Path(__file__).resolve().parents[4])))
base=workspace/'work/private/v5-execution-20261025'
if not(base/'VANDA_AUDIT_CODEC_COLD_01_C50_UNCHANGED_CLOSED_STREAM_CONFIRMED.private.json').is_file():
    pytest.skip('Actual private closed-stream/codec captures unavailable; not native fixtures',allow_module_level=True)
c=SimpleNamespace(PREFIX="VANDA_AUDIT_CODEC_COLD_01_",W=workspace,B=base,R=Path(__file__).resolve().parents[2],a=a,SOURCE='30d6d891c4b8fe927e66d439d2862901458d2b0e')
c.original=SimpleNamespace(prepared=lambda:json.loads((base/'VANDA_COHOST_PREPARED.private.json').read_bytes()))

def args():
 actual=json.loads(sorted(c.B.glob('*VANDA_AUDIT_CODEC_COLD_01_STATE_BEFORE_READONLY_*'))[-1].read_bytes())['value'];gb=base64.b64decode(next(v['base64']for n,v in actual['files'].items()if n.endswith('HOPPER_RESOURCE_LEDGER.json')));g=json.loads(gb)
 template=json.loads((c.W/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01/CORE_PAIRED_PEER_TEMPLATES_7cc75180.private.json').read_bytes());closed=json.loads(sorted(c.B.glob('C50_REAL_CLOSED_HANDOFF_READONLY_*.private.json'))[-1].read_bytes())['value']['closed_capture'];cap=deepcopy(closed);cap['global_base64']=base64.b64encode(gb).decode();cap['handoff']=cohost.endpoint(gb,cap)
 ref=json.loads((c.B/(c.PREFIX+'C50_UNCHANGED_CLOSED_STREAM_CONFIRMED.private.json')).read_bytes())['plan'];dp=json.loads((c.B/'D25_VANDA_FIRST_PREPARED.private.json').read_bytes());old=c.original.prepared();parent=str(Path(old['root']).parent.parent)
 codec_proof=json.loads((c.B/(c.PREFIX+'INDEPENDENT_COMPLETE_CONFIRMED.private.json')).read_bytes());toolchain=json.loads(base64.b64decode(next(v['base64']for n,v in codec_proof['files'].items()if n.endswith('VANDA_AUDIT_CODEC_TOOLCHAIN.private.json'))));lease=base64.b64decode(json.loads((c.B/(c.PREFIX+'CLOSE_METADATA_LEASE.private.json')).read_bytes())['value']['closed_lease_base64']);source='b'*40;frozen=template['rows']['A_CONTROL']['core']['frozen'];seal=dict(source_sha=source,files=frozen,frozen_numerical_overlay=template['rows']['A_CONTROL']['core']['manifest_template']['code']['frozen_numerical_overlay'])
 config=json.loads((c.B/'C50_CORE_EVAL_OWNER_COLD_PREPARED.private.json').read_bytes())['config'];repair=cohost.repair.validated(g)
 profiles=dict(core=dict(name=identity.PHYSICAL_PROFILE,source_release=source,PBS_required=True,container=old['profile']['container']),hopper=dict(name=identity.NEURAL_PROFILE,source_release=source,PBS_required=True))
 return dict(template=template,global_bytes=gb,historical_LOCAL_bytes=base64.b64decode(old['historical_LOCAL_base64']),training_capture_bytes=a.raw(cap),reference_plan=ref,
  source_sha=source,source_seal_bytes=a.raw(seal),parent=parent,registry=old['registry'],environment_registry=old['environment_registry'],profiles=profiles,
  native_neural_bindings=json.loads(base64.b64decode(dp['small']['NEURAL_BINDINGS.private.json'])),native_training_inputs=json.loads(base64.b64decode(dp['small']['plans/PV1_TRAIN_INPUTS.private.json'])),
  asset_plan=json.loads(base64.b64decode(old['small']['plans/ASSETS.private.json'])),codec_toolchain=toolchain,codec_preparation_root=str(Path(old['root']).parent.parent/'control/v5-cd-c50-vanda-codec-cold-01'),codec_closed_lease_bytes=lease,
  queue_limit_bytes=a.raw(dict(source='VANDA_NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS',requested_queue='auto',destination='batch_gpu',maximum_wall_s=604800,observed_epoch=time.time())),independent_job_bytes=base64.b64decode(config['outside_confirmation_base64']),
  owner_code={'v5_vanda_colocated_compute_owner.py':b'EXPLICIT_CPU_OWNER_PROGRAM_HASH_FIXTURE_NOT_EXECUTED'},programs=dict(AUTH='EXPLICIT_CPU_FIXTURE_NOT_EXECUTED',QUOTA='EXPLICIT_CPU_FIXTURE_NOT_EXECUTED'),request_id=repair['replacement_request_id'],nonce=repair['replacement_submission_nonce'],run_name=repair['replacement_run_uuid'],now_epoch=time.time())

def test_actual_closed_C50_whole_budget_packet_no_hash_cycle_or_double_payment():
 x=args();old=json.loads(x['global_bytes']);packet=build.assemble(**x);r=packet['request'];om=packet['owner']['manifest'];cfg=packet['owner']['config'];new=packet['proposed_global']
 assert packet['native_registered']is False and packet['new_GPU']==packet['new_PBS']==packet['new_model']==packet['new_physics']==0
 assert r['owner_manifest_sha256']==a.sha(a.raw(om))and om['execution']['configuration_sha256']==a.sha(a.raw(cfg))
 assert cfg['sites']['hopper']['manifest_sha256']==r['run_manifest_sha256']==packet['rows']['hopper']['manifest_sha256']
 assert cfg['sites']['core']['manifest_sha256']==packet['record']['physical_manifest_sha256']
 assert old['used']['PBS_submissions']==29 and new['used']['PBS_submissions']==30
 assert old['CPU_s']==new['CPU_s']and old['new_storage_current_bytes']==new['new_storage_current_bytes']
 for role,row in packet['rows'].items():
  m=row['manifest'];assert a.sha(a.raw(row['binding']))==m['execution']['output_binding_sha256']
  assert all(a.sha(b)==m['execution']['sealed_inputs'][n]for n,b in row['files'].items())
  assert json.loads(row['files']['plans/PRIOR_EVALUATION_OUTCOMES.json'])==[]
  assert 'evidence/COUNTER_NATIVE_QUALIFICATION.json'not in row['files']
  assert 'plans/CORE_GLOBAL_LEDGER_BINDINGS.private.json'not in row['files']
  assert m['owner_uid']==22182 and m['limits']['training_updates']==0
  assert r['sealed_inputs']['plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json']==m['execution']['sealed_inputs']['plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json']
 assert accounting.rows(new)[0]['request_id']==r['request_id']
 assert c.SOURCE not in r['source_sha']  # native metadata source is not claimed as fixture Eval source

def test_scope_and_original_frozen_source_substitution_refused():
 for change in ('frozen','codec_active','old_global','changed_asset','old_core_profile'):
  x=args()
  if change=='frozen':
   v=json.loads(x['source_seal_bytes']);v['files']['scripts/pi0294/score.py']='0'*64;x['source_seal_bytes']=a.raw(v)
  elif change=='codec_active':
   v=json.loads(x['codec_closed_lease_bytes']);v['status']='ACTIVE';x['codec_closed_lease_bytes']=a.raw(v)
  elif change=='old_global':
   v=json.loads(x['global_bytes']);v['revision']+=1;x['global_bytes']=a.raw(v)
  elif change=='changed_asset':x['asset_plan']['files'][next(iter(x['asset_plan']['files']))]='0'*64
  else:x['profiles']['core']['source_release']='0'*40
  try:build.assemble(**x)
  except PermissionError:pass
  else:raise AssertionError('accepted substitution '+change)

def test_pure_packet_cannot_replace_actual_cold_readback():
 x=args();packet=build.assemble(**x)
 assert packet['native_registration_ready_after_owner_fresh_full_queue_and_project']is False
 audit=dict(schema='V5_CD_VANDA_ACTUAL_COLD_PAIRED_INPUT_READBACK_V1',status='PASS_ACTUAL_REGISTERED_IMMUTABLE_COLD_INPUTS_NO_MODEL_OR_PHYSICS',
  source_sha=packet['request']['source_sha'],run_uuid=packet['request']['run_uuid'],GLOBAL_sha256=a.sha(x['global_bytes']),
  peer_manifest_sha256={k:r['manifest_sha256']for k,r in packet['rows'].items()},all_source_files_full_SHA=True,all_sealed_inputs_full_SHA=True,
  all_fifteen_original_numerical_files=True,original_26_assets_full_SHA=True,nine_original_model_inodes_unchanged=True,
  actual_registry_and_own_ACTIVE_leases=True,new_GPU=0,new_PBS=0,new_model=0,new_constructor=0,new_physics=0,observed_epoch=time.time())
 # The receipt is an explicit CPU fixture, never actual native evidence.
 result=build.bind_actual_cold_readback(packet,audit_bytes=a.raw(audit),global_bytes=x['global_bytes'],now_epoch=time.time())
 assert result['native_registered']is False and result['request']['cold_input_audit_sha256']==a.sha(a.raw(audit))
 for key in ('all_source_files_full_SHA','all_fifteen_original_numerical_files','nine_original_model_inodes_unchanged','actual_registry_and_own_ACTIVE_leases'):
  bad=deepcopy(audit);bad[key]=False
  with pytest.raises(PermissionError):build.bind_actual_cold_readback(packet,audit_bytes=a.raw(bad),global_bytes=x['global_bytes'],now_epoch=time.time())


def test_new_cpu_slot_policy_is_explicit_in_all_sealed_manifests_and_config():
 from scripts.pi0294.pv1.mix_vanda_affinity import POLICY
 x=args();old=build.assemble(**x)
 assert all('CPU_slot_policy'not in r['manifest']['execution']for r in old['rows'].values())
 x['cpu_slot_policy']=POLICY;packet=build.assemble(**x);owner=packet['owner'];cfg=owner['config']
 assert cfg['CPU_slot_policy']==owner['manifest']['execution']['CPU_slot_policy']==POLICY
 assert a.sha(a.raw(cfg))==owner['manifest']['execution']['configuration_sha256']
 assert packet['request']['owner_manifest_sha256']==a.sha(a.raw(owner['manifest']))
 for row in packet['rows'].values():
  assert row['manifest']['execution']['CPU_slot_policy']==POLICY
  assert a.sha(a.raw(row['manifest']))==row['manifest_sha256']
 assert packet['rows']['core']['manifest']['limits']['cpus']==[4,5]
 x['cpu_slot_policy']='UNKNOWN_POLICY'
 with pytest.raises(PermissionError):build.assemble(**x)
