"""One exact cold role successor of the closed, unsubmitted E2cold01.

The original unused native envelope and paid control stay single. Raw payment,
closed roles and native history are immutable; the effective view changes only
source/role identity. This module performs no transport, IO, qsub or effects.
"""
from copy import deepcopy
import json,re
from . import minimal_execution as e

FIELD='ARX5_minimal_E2_only_raw512_closed_cold01_repair'
BINDING='plans/E2_CLOSED_COLD01_SUCCESSOR_SCOPE.private.json'
SNAPSHOT='plans/E2_ONLY_CLOSED_COLD01.private.json'
TAG='closed_cold01_repair_sha256'
RID='5a2c08f1-4d4d-41b0-bad0-e8f99257a229'
NONCE='bd1b2352-a838-400a-ab3c-81fdd84a47e5'
OLD_RUN='arx5-minimal-e2-only-raw-fixed512-02-cold01'
RUN='arx5-minimal-e2-only-raw-fixed512-02-cold02'
OLD_SOURCE='123508f76068e96ea1f8adcc1b59a0f9a8c1067e'
OLD_MANIFEST='3f6b6b955443e4ceacaa3141c5c6f258064511ffac5563ffed4a568cf598d027'
LEASE='1abed61c0c14f9d9151a5f5ff7a2a444c82729ca8393169e4913e9d5e3529ddd'
OS='53d90d76e6ca3a78e2ab8402c2782bc60a441b2eede8033f11cfcb07e11a9afa'
STOP='54bb49f87d7fa5de9bfb67a251eab44b144ddd77b69ef7846bf8ca721eb45d43'
AUTH='49506e3a135ad6ad931cd672cb9f0b9f6214e47fb072fd160b3f29b5932f007b'
GLOBAL='e602d251c9c1abaa48ae36e9018fa7a15b02e0b55c9932705a0cea67e2e8ae2e'
LOCAL='d98efaf9ab92234719e5db2c23635f122c87f39e9d862777980c11696484a89e'
TRAINING_FILES={
 'scripts/pi0294/generalization/minimal_dataset.py':'430ba270da63d77744d0c44c08b1071e4e214d8b41cc3aa4a549e968e4ea3f90',
 'scripts/pi0294/generalization/minimal_model.py':'3be2b2edc2c3e4039ba2896df9e07497bb3b916c0f005bbf90ebdd4ebfb7f4a6',
 'scripts/pi0294/generalization/minimal_training.py':'487b7ba80cd3d4a618794056229027af162cdb1fe5f066c3759f52834543d8c2',
 'scripts/pi0294/generalization/target_eef10_contract.py':'bbfab680a19dc45e249c586a16a6ee8ac105ea4c3a5790a8800e620575f18cdc'}
need=e.need

def digest(v):return e.sha(e.raw(v))
def pin(v):return type(v)is str and re.fullmatch('[0-9a-f]{64}',v)is not None

def closed_role(v):
    need(type(v)is dict and v.get('schema')=='ARX5_E2_CLOSED_UNREGISTERED_PREPARATION_V1'
         and (v.get('request_id'),v.get('run_uuid'),v.get('source_sha'),v.get('manifest_sha256'))==(RID,OLD_RUN,OLD_SOURCE,OLD_MANIFEST)
         and v.get('owner_OS_sha256')==OS and v.get('owner_stop_sha256')==STOP
         and v.get('closed_lease_sha256')==LEASE and v.get('lease_status')=='CLOSED'
         and v.get('native_requests')==0 and type(v['native_requests'])is int
         and v.get('all_recorded_actors_terminal')is True
         and type(v.get('allocated_bytes'))is int and 0<v['allocated_bytes']<=256*1024**2
         and pin(v.get('snapshot_sha256'))and pin(v.get('snapshot_identity_sha256')),
         'E2_COLD02_ONLY_REAL_CLOSED_UNREGISTERED_COLD01')
    return v


def request_shape(r):
    closed_role(r.get('closed_pre_native_role'))
    need(r['request_id']==RID and r['submission_nonce']==NONCE and r['run_uuid']==RUN
         and r.get('attempt_index')==2 and r.get('pre_native_cold_generation')==2
         and r.get('pre_native_owner_CPU_upper_s')==2600 and pin(r.get(TAG))
         and r['source_sha']!=OLD_SOURCE,'E2_COLD02_EXACT_SINGLE_UNUSED_IDENTITY_AND_RETAINED_OWNER_CPU')


def validate(g):
    from . import minimal_e2only_execution as x
    m=g.get(FIELD)
    if m is None:return None
    need(type(m)is dict and set(m)=={'scope','binding'},'E2_COLD02_EXACT_SCOPE_AND_ACTUAL_BINDING')
    s,b=m['scope'],m['binding'];old=g[x.PREPAID][RID];prior=s['old_request'];x.request(prior)
    need(s['schema']=='ARX5_E2_EXACT_CLOSED_COLD01_SUCCESSOR_SCOPE_V1'and s['authority_sha256']==AUTH
         and s['before_GLOBAL_sha256']==GLOBAL and s['before_LOCAL_sha256']==LOCAL
         and s['old_paid_marker_sha256']==digest(old)and s['old_request_sha256']==digest(prior)
         and digest(s['old_manifest'])==OLD_MANIFEST
         and digest(s['old_physical_custody'])==old['physical_custody_sha256']==prior['physical_custody_sha256']
         and prior['request_id']==RID and prior['source_sha']==old['source_sha']==OLD_SOURCE
         and prior['run_uuid']==old['run_uuid']==OLD_RUN and prior['run_manifest_sha256']==OLD_MANIFEST
         and old['submission_nonce']==s['submission_nonce']==NONCE and s['request_id']==RID and s['run_uuid']==RUN
         and re.fullmatch('[0-9a-f]{40}',s['source_sha'])and s['source_sha']!=OLD_SOURCE
         and pin(s['source_seal_sha256'])and pin(s['source_readback_sha256'])
         and s['new_native_requests']==s['new_control_payment']==s['new_tensor_transfers']==0
         and s['old_owner_CPU_s_retained']==400 and s['metadata_CPU_s_retained']==600 and s['new_owner_CPU_upper_s']==2600
         and s['T_CD']==prior['T_CD']and s['deadline_epoch']==prior['deadline_epoch']
         and prior['created_epoch']<=s['adopted_epoch']<min(prior['stop_new_training_epoch'],1793404800)
         and closed_role(s['closed_role'])==s['closed_role'],
         'E2_COLD02_EXACT_OLD_FINANCIAL_SCIENTIFIC_AND_CLOCK_HISTORY')
    need(set(b)=={'manifest_sha256','physical_custody_sha256'}and all(pin(v)for v in b.values()),
         'E2_COLD02_COMPLETE_NEW_IMMUTABLE_ROLE_BINDING')
    rows=[r for r in g['PBS_submissions']if r.get('request_id')==RID]
    need(len(rows)<=1,'E2_COLD02_ONE_NATIVE_IDENTITY')
    if rows:validate_bound_request(g,rows[0],_marker=m)
    return m


def effective_controls(g,raw_controls):
    m=validate(g)
    if m is None:return raw_controls
    s,b=m['scope'],m['binding'];items=dict(raw_controls);v=deepcopy(items[RID])
    v.update(run_uuid=RUN,source_sha=s['source_sha'],run_manifest_sha256=b['manifest_sha256'],
        physical_custody_sha256=b['physical_custody_sha256'])
    items[RID]=v
    return items


def validate_bound_request(g,r,*,_marker=None):
    from .minimal_e2only_execution import TERMINAL_KEYS
    m=_marker or validate(g);need(m is not None,'E2_COLD02_ADOPTED_SCOPE_REQUIRED')
    request_shape(r);s,b=m['scope'],m['binding'];old=s['old_request']
    mutable={'source_sha','run_uuid','run_manifest_sha256','physical_custody_sha256','cold_input_audit_sha256',
        'effective_queue_proof_sha256','created_epoch','submit_count','pre_native_cold_generation',
        'pre_native_owner_CPU_upper_s','closed_pre_native_role','frozen_input_packet_sha256',TAG}
    need(r.get(TAG)==digest(s)and r['source_sha']==s['source_sha']
         and r['run_manifest_sha256']==b['manifest_sha256']and r['physical_custody_sha256']==b['physical_custody_sha256']
         and r['closed_pre_native_role']==s['closed_role']
         and all(r.get(k)==v for k,v in old.items()if k not in mutable)
         and set(r)-set(old)<=mutable|TERMINAL_KEYS|{'request_sha256','status','job_id','submitted_epoch'},
         'E2_COLD02_ONLY_ROLE_BINDING_FIELDS_CHANGE')
    return m


def sealed_context(root,manifest,request,g):
    from pathlib import Path
    m=validate(g);need(m is not None,'E2_COLD02_ACTUAL_ADOPTED_SCOPE')
    from scripts.pi0292.durable import read
    from scripts.pi0294.native_dataset import sha
    s=m['scope'];root=Path(root)
    need(read(root/BINDING)==s and sha(root/BINDING)==manifest['execution']['sealed_inputs'].get(BINDING)==digest(s)
         and root.name==manifest['run_uuid']==request['run_uuid']==RUN
         and manifest['code']['vla_commit']==request['source_sha']==s['source_sha']
         and manifest['code']['source_seal_sha256']==s['source_seal_sha256']
         and sha(root/'RUN_MANIFEST.json')==m['binding']['manifest_sha256']
         and request.get(TAG)==digest(s),'E2_COLD02_OWN_IMMUTABLE_SOURCE_SCOPE_AND_MANIFEST')


def prepare_scope(global_bytes,local_bytes,*,authority_bytes,closed_capture,old_request,old_manifest,old_physical_custody,source_seal_bytes,frozen,source_readback,now_epoch):
    from scripts.pi0294.pv1.neural_ledger import check
    from scripts.pi0294.pv1.evaluation_core_scope import ORIGINAL_NUMERICAL_TABLE_SHA
    from . import minimal_e2only_execution as x
    g=json.loads(global_bytes);check(g);x.request(old_request);record=closed_role(closed_capture['record']);seal=json.loads(source_seal_bytes)
    need(FIELD not in g and e.sha(global_bytes)==GLOBAL and e.sha(local_bytes)==LOCAL and e.sha(authority_bytes)==AUTH
         and all(r['status']==e.STATES[2]for r in g['PBS_submissions'])
         and not any(r.get('request_id')==RID for r in g['PBS_submissions'])
         and closed_capture['GLOBAL_sha256']==GLOBAL and closed_capture['new_cold02_absent']is True
         and closed_capture['model_calls']==closed_capture['new_native_requests']==closed_capture['new_mutations']==0
         and 0<=now_epoch-closed_capture['observed_epoch']<=120,
         'E2_COLD02_FRESH_EXACT_UNSUBMITTED_BOUNDARY_AND_DIRECT_AUTHORITY')
    need(len(frozen)==15 and e.sha(json.dumps(frozen,sort_keys=True).encode())==ORIGINAL_NUMERICAL_TABLE_SHA
         and all(seal['files'].get(n)==pin for n,pin in frozen.items())
         and all(seal['files'].get(n)==pin for n,pin in TRAINING_FILES.items())
         and source_readback['status']=='ACTUAL_COMPLETE_COLD_SOURCE_SHA_READBACK'
         and source_readback['source_sha']==seal['source_sha']
         and source_readback['source_seal_sha256']==e.sha(source_seal_bytes)
         and source_readback['file_sha256']==seal['files']and source_readback['uid']==22182
         and source_readback['hostname'].startswith('stdct-login-')
         and source_readback['model_calls']==source_readback['physics_ticks']==0
         and 0<=now_epoch-source_readback['observed_epoch']<=120,'E2_COLD02_ACTUAL_COMPLETE_SOURCE_AND_ORIGINAL_FROZEN_BYTES')
    return dict(schema='ARX5_E2_EXACT_CLOSED_COLD01_SUCCESSOR_SCOPE_V1',authority_sha256=AUTH,
        before_GLOBAL_sha256=GLOBAL,before_LOCAL_sha256=LOCAL,old_paid_marker_sha256=digest(g[x.PREPAID][RID]),
        old_request=deepcopy(old_request),old_request_sha256=digest(old_request),old_manifest=deepcopy(old_manifest),
        old_physical_custody=deepcopy(old_physical_custody),closed_role=deepcopy(record),
        request_id=RID,submission_nonce=NONCE,run_uuid=RUN,source_sha=seal['source_sha'],source_seal_sha256=e.sha(source_seal_bytes),
        source_readback_sha256=digest(source_readback),new_native_requests=0,new_control_payment=0,new_tensor_transfers=0,
        old_owner_CPU_s_retained=400,metadata_CPU_s_retained=600,new_owner_CPU_upper_s=2600,
        T_CD=old_request['T_CD'],deadline_epoch=old_request['deadline_epoch'],adopted_epoch=now_epoch)


def propose(global_bytes,local_bytes,*,scope,manifest_bytes,physical_bytes,request,environment_binding_bytes=None):
    from scripts.pi0294.pv1.neural_ledger import check
    from . import minimal_e2only_execution as x,minimal_e2only_physical as physical
    g=json.loads(global_bytes);need(e.sha(global_bytes)==GLOBAL and e.sha(local_bytes)==LOCAL and FIELD not in g,
        'E2_COLD02_ONE_EXACT_CANONICAL_ADOPTION')
    m=json.loads(manifest_bytes);p=json.loads(physical_bytes);physical.validate_custody(p,request,predecessor_binding=x.c.binding())
    v=deepcopy(g);v[FIELD]=dict(scope=deepcopy(scope),binding=dict(manifest_sha256=e.sha(manifest_bytes),physical_custody_sha256=e.sha(physical_bytes)))
    v['revision']+=1;validate(v);validate_bound_request(v,request);x.require_paid_retry(v,request);check(v)
    need(m['run_uuid']==m['output_root'].split('/')[-1]==RUN and m['code']['vla_commit']==scope['source_sha']
         and m['code']['source_seal_sha256']==scope['source_seal_sha256']
         and m['execution']['sealed_inputs'].get(BINDING)==digest(scope)
         and m['execution']['sealed_inputs'].get(x.PHYSICAL)==e.sha(physical_bytes),
         'E2_COLD02_REAL_NEW_MANIFEST_AND_PHYSICAL_CUSTODY')
    mutable_inputs={'SOURCE_SEAL.json','MIX_OWNER_CONFIG.private.json',x.PHYSICAL,
        'plans/MINIMAL_FROZEN_INPUT_PACKET.private.json','plans/MINIMAL_CORE_LOCAL_CAPTURE.private.json'}
    previous=scope['old_manifest']
    env='plans/A40_ENVIRONMENT_BINDING.private.json'
    if m['execution']['sealed_inputs'].get(env)!=previous['execution']['sealed_inputs'].get(env):
        need(type(environment_binding_bytes)is tuple and len(environment_binding_bytes)==2,
             'E2_COLD02_EXACT_OLD_AND_NEW_ENVIRONMENT_BINDING_BYTES')
        before,after=environment_binding_bytes;oldenv,newenv=map(json.loads,(before,after))
        need(e.sha(before)==previous['execution']['sealed_inputs'][env]
             and e.sha(after)==m['execution']['sealed_inputs'][env]
             and set(oldenv)==set(newenv)and pin(newenv.get('environment_registry_sha256'))
             and all(newenv[k]==v for k,v in oldenv.items()if k!='environment_registry_sha256'),
             'E2_COLD02_ONLY_ENVIRONMENT_REGISTRY_METADATA_HASH_CHANGES')
        mutable_inputs.add(env)
    need(m['code']['frozen_numerical_files']==previous['code']['frozen_numerical_files']
         and m['code']['official_gr00t_commit']==previous['code']['official_gr00t_commit']
         and m['authority']==previous['authority']and m['limits']==previous['limits']
         and m['scope']==previous['scope']and m['execution']['profile']==previous['execution']['profile']
         and all(m['execution']['sealed_inputs'].get(n)==pin for n,pin in previous['execution']['sealed_inputs'].items()if n not in mutable_inputs),
         'E2_COLD02_EVERY_OLD_SCIENTIFIC_INPUT_UNCHANGED')
    before_physical=scope['old_physical_custody']
    need(all(p.get(k)==val for k,val in before_physical.items()if k not in('run_uuid','retained_initial_bytes','closed_pre_native_roots'))
         and p['closed_pre_native_roots']==before_physical['closed_pre_native_roots']+[
             dict(root=previous['output_root'],closure=scope['closed_role'],snapshot_file=SNAPSHOT)]
         and p['retained_initial_bytes']>=before_physical['retained_initial_bytes']+scope['closed_role']['allocated_bytes'],
         'E2_COLD02_NO_OLD_PHYSICAL_LINEAGE_OR_GROSS_REFUND')
    future=e.scope.first512_forecast(e.raw(v),local_bytes)
    need(future['cap_changes_required']=={}and v[x.PREPAID]==g[x.PREPAID]
         and v['PBS_submissions']==g['PBS_submissions']and v['used']==g['used']and v['reserved']==g['reserved']
         and v['CPU_s']==g['CPU_s'],'E2_COLD02_NO_PAYMENT_HOLD_HISTORY_RESET')
    return dict(proposed_global=v,full_future_proof=future,new_native_permission=False,original_onceIO_required=True)
