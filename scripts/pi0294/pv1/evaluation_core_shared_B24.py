"""Exact direct-human shared GPU0 B24 cold continuation, never a generic bypass."""
from copy import deepcopy
import hashlib,json,re
from . import evaluation_core_B24_1000 as late
from . import evaluation_core_prebind_failure as failed
FIELD='Core_B24_shared_GPU0_human_boundary'
GRANT_SHA='2790bb5a88a9e456a10eb2cb6698ef2f064977ba2508d581d0fe92fe3eca858d'
PLAN='plans/B24_EXACT_SHARED_GPU_HUMAN_GRANT.json'
RID='4cc2048c-3e0f-43ea-93d0-9450a3569adc'
CONTROL='26a13cf0-6f9b-4c22-a97a-0e82ffde59f7'
NONCE='2980dc19-aea3-47bc-b172-9a9888cfc23e'
RUN='pv1-v4a-b4096-n294101-shared-gpu0-03'
GPU='GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'
def need(ok,why):
 if not ok:raise PermissionError('PV1_B24_SHARED_'+why)
def grant(raw):
 need(type(raw) is bytes and hashlib.sha256(raw).hexdigest()==GRANT_SHA,'EXACT_DIRECT_HUMAN_SHARED_GPU_GRANT')
 g=json.loads(raw);need(g['direct_human_answer']=='允许直接共享GPU0做eval，显存充足，如果失败可以尝试GPU1' and g['new_request_id']==RID and g['new_control_id']==CONTROL and g['latest_native_start_epoch']==late.LATEST and g['budget_changes']=={},'EXACT_ONLY_B24_AND_UNCHANGED_CAPS_CLOCKS')
 return g
def allowed(r):
 if 'shared_GPU_grant_sha256' not in r:return False
 need(r.get('shared_GPU_grant_sha256')==GRANT_SHA and r.get('late_start_grant_sha256')==late.GRANT_SHA
      and r.get('request_id')==RID and r.get('run_uuid',r.get('child_run_uuid'))==RUN
      and r.get('control_id')==CONTROL and r.get('submission_nonce')==NONCE
      and r.get('evaluation_candidate')=='B_COVERAGE' and r.get('stage')=='P4_DEV' and r.get('noise_bank')==294101
      and r.get('GPU_uuid')==GPU and not any(k.startswith('queue_') for k in r),'EXACT_ONE_GPU0_ORIGINAL_B24_REQUEST_NOT_GENERAL_SHARING')
 return True
def validate(g):
 from . import evaluation_core_accounting as c
 m=g.get(FIELD)
 if m is None:
  need(not any('shared_GPU_grant_sha256' in r for r in g.get(c.FIELD,{}).values()),'NO_UNADOPTED_SHARED_ROW');return None
 need(m['schema']=='PV1_B24_EXACT_SHARED_GPU_ONCE_BOUNDARY_V1' and m['grant_sha256']==GRANT_SHA
      and m['replacement_request_id']==RID and m['replacement_submission_nonce']==NONCE
      and m['control_id']==CONTROL and m['replacement_run_uuid']==RUN
      and late.T0<=m['created_epoch']<late.LATEST and m['automatic_retry'] is False
      and re.fullmatch('[0-9a-f]{40}',m['replacement_source_sha']) is not None,'ONE_EXACT_FRESH_COLD_BOUNDARY')
 old=g[c.FIELD][failed.RID];failed.validate_row(old);cl=m['predecessor_dual_closure']
 need(old['status']==c.STATES[3] and cl['request_id']==failed.RID and cl['terminal_native_sha256']==old['terminal_native_sha256']
      and cl['GPU_lease_status']==cl['physical_lease_status']=='CLOSED' and cl['native_exit_status']==1
      and all(cl[k]==0 for k in ('remaining_holds','active_native_actors','actual_valid_conditions','actual_model_loads','actual_queries','actual_physics','actual_training_updates')),
      'REAL_PAID_KNOWN_ZERO_PREDECESSOR_WITH_BOTH_LEASES_CLOSED')
 if RID in g[c.FIELD]:
  r=g[c.FIELD][RID];allowed(r);need(r['source_sha']==m['replacement_source_sha'] and r['startup_recovery_sha256']==c.digest(m),'SAME_NEW_SEALED_SOURCE_AND_BOUNDARY')
 return m
def validate_pair(g,l):
 m=validate(g);need(l.get(FIELD)==m,'SAME_ACTUALLY_ADOPTED_DUAL_SHARED_BOUNDARY');return m
def check_request(manifest,r,g=None):
 need(allowed(r),'SHARED_REQUEST_REQUIRED')
 need(manifest['execution']['sealed_inputs'].get(PLAN)==r['sealed_inputs'].get(PLAN)==GRANT_SHA
      and manifest['execution']['sealed_inputs'].get(late.PLAN)==r['sealed_inputs'].get(late.PLAN)==late.GRANT_SHA,
      'BOTH_EXACT_HUMAN_GRANTS_SEALED_IN_REAL_MANIFEST_AND_REQUEST')
 if g is not None:
  m=validate(g);need(m is not None and m['replacement_source_sha']==r['source_sha'],'CURRENT_ACTUAL_ADOPTED_SOURCE')
 return late.LATEST
def install(g,l,raw,*,source_sha,closure,now_epoch):
 from . import evaluation_core_accounting as c
 grant(raw);c.checked(g);c.scope.validate_pair(g,l)
 need(FIELD not in g and FIELD not in l and RID not in g[c.FIELD] and late.T0<=now_epoch<late.LATEST
      and all(r['status']==c.STATES[3] for r in g[c.FIELD].values())
      and all(x['status']=='CLOSED' for x in g['V4_DEV_control_reservations'].values())
      and g['unknown_effects']==l['unknown_effects']==[] and g['pending_updates']==l['pending_updates']==[],'FRESH_INACTIVE_CLOSED_ONCE_BOUNDARY_NO_UNKNOWN_REPLAY')
 g,l=deepcopy(g),deepcopy(l);m=dict(schema='PV1_B24_EXACT_SHARED_GPU_ONCE_BOUNDARY_V1',grant_sha256=GRANT_SHA,replacement_request_id=RID,replacement_submission_nonce=NONCE,replacement_run_uuid=RUN,replacement_source_sha=source_sha,control_id=CONTROL,created_epoch=now_epoch,predecessor_dual_closure=deepcopy(closure),automatic_retry=False)
 for v in (g,l):v[FIELD]=deepcopy(m)
 validate_pair(g,l);c.checked(g);c.scope.validate_pair(g,l);return g,l
