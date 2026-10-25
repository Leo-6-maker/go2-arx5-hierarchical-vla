"""One same-panel cold owner forwarding repair under explicit shared GPU GO."""
from copy import deepcopy
import hashlib,json,re
from . import evaluation_core_shared_B24 as parent,evaluation_core_B24_1000 as late
FIELD='Core_B24_shared_GPU0_owner_facts_cold_repair'
RID='631b7d63-7f7b-419a-9eb9-fb1a7defa9f8';CONTROL='90a17516-531a-4248-b62b-59571a4929db';NONCE='48515f1e-2666-4abe-a2cb-6ebe12fc8032';RUN='pv1-v4a-b4096-n294101-shared-gpu0-repair-04'
SCOPE_SHA='24d23684217b28c95dfe32b56b1056e77385e4a1a535da81a3651360060d95ce'
PLAN='plans/B24_SHARED_GPU0_ENGINEERING_REPAIR_SCOPE.json'
def need(ok,why):
 if not ok:raise PermissionError('PV1_B24_SHARED_REPAIR_'+why)
def scope(raw):
 need(type(raw) is bytes and hashlib.sha256(raw).hexdigest()==SCOPE_SHA,'EXACT_PUBLISHED_OWNER_FACTS_REPAIR');return json.loads(raw)
def allowed(r):
 need(r.get('shared_owner_repair_scope_sha256')==SCOPE_SHA and r.get('shared_GPU_grant_sha256')==parent.GRANT_SHA
      and r.get('late_start_grant_sha256')==late.GRANT_SHA and r.get('request_id')==RID
      and r.get('run_uuid',r.get('child_run_uuid'))==RUN and r.get('control_id')==CONTROL and r.get('submission_nonce')==NONCE
      and r.get('GPU_uuid')==parent.GPU and r.get('evaluation_candidate')=='B_COVERAGE' and r.get('stage')=='P4_DEV'
      and r.get('noise_bank')==294101 and not any(k.startswith('queue_') for k in r),'EXACT_ONE_ORIGINAL_B24_GPU0_COLD_FORWARDING_REPAIR')
 return True
def validate(g):
 from . import evaluation_core_accounting as c
 m=g.get(FIELD)
 if m is None:
  need(not any('shared_owner_repair_scope_sha256' in r for r in g.get(c.FIELD,{}).values()),'NO_ORPHAN_REPAIR_ROW');return None
 parent.validate(g);old=g[c.FIELD][parent.RID];cl=m['predecessor_dual_closure']
 need(m['schema']=='PV1_B24_SHARED_GPU0_OWNER_FACTS_COLD_REPAIR_V1' and m['scope_sha256']==SCOPE_SHA
      and m['replacement_request_id']==RID and m['replacement_submission_nonce']==NONCE and m['control_id']==CONTROL
      and m['replacement_run_uuid']==RUN and m['automatic_retry'] is False and late.T0<=m['created_epoch']<late.LATEST
      and re.fullmatch('[0-9a-f]{40}',m['replacement_source_sha']) is not None,'EXACT_SINGLE_COLD_ENGINEERING_REPAIR_BOUNDARY')
 need(old['status']==c.STATES[3] and old['native_exit_status']==1 and c.progress(old)[:2]==(0,0)
      and old['query_attempts']==old['model_load_attempts']==0 and old['unknown_query_intents']==[]
      and old['conservative_GPU_s']>0 and old['conservative_CPU_s']>0
      and cl['request_id']==parent.RID and cl['terminal_native_sha256']==old['terminal_native_sha256']
      and cl['GPU_lease_status']==cl['physical_lease_status']=='CLOSED' and cl['native_exit_status']==1
      and all(cl[k]==0 for k in ('remaining_holds','active_native_actors','actual_valid_conditions','actual_model_loads','actual_queries','actual_physics','actual_training_updates')),
      'EXACT_KNOWN_PAID_ZERO_MODEL_PARENT_AND_BOTH_CLOSED_LEASES')
 if RID in g[c.FIELD]:
  r=g[c.FIELD][RID];allowed(r);need(r['source_sha']==m['replacement_source_sha'] and r['startup_recovery_sha256']==c.digest(m),'EXACT_NEW_SOURCE_SEALED_BOUNDARY')
 return m
def pair(g,l):
 m=validate(g);need(l.get(FIELD)==m,'SAME_DUAL_REPAIR_BOUNDARY');return m
def check_request(manifest,r,g=None):
 allowed(r)
 for p,h in ((PLAN,SCOPE_SHA),(parent.PLAN,parent.GRANT_SHA),(late.PLAN,late.GRANT_SHA)):
  need(manifest['execution']['sealed_inputs'].get(p)==r['sealed_inputs'].get(p)==h,'ALL_EXACT_HUMAN_AND_REPAIR_INPUTS_SEALED')
 if g is not None:
  m=validate(g);need(m is not None and m['replacement_source_sha']==r['source_sha'],'ACTUAL_ADOPTED_NEW_SOURCE')
 return late.LATEST
def install(g,l,raw,*,source_sha,closure,now_epoch):
 from . import evaluation_core_accounting as c
 scope(raw);c.checked(g);c.scope.validate_pair(g,l)
 need(FIELD not in g and FIELD not in l and RID not in g[c.FIELD] and late.T0<=now_epoch<late.LATEST
      and all(r['status']==c.STATES[3] for r in g[c.FIELD].values()) and all(x['status']=='CLOSED' for x in g['V4_DEV_control_reservations'].values())
      and g['unknown_effects']==l['unknown_effects']==[] and g['pending_updates']==l['pending_updates']==[],'ALL_ACTUAL_FAILED_EFFECTS_CLOSED_INACTIVE_FRESH_WINDOW')
 g,l=deepcopy(g),deepcopy(l);m=dict(schema='PV1_B24_SHARED_GPU0_OWNER_FACTS_COLD_REPAIR_V1',scope_sha256=SCOPE_SHA,replacement_request_id=RID,replacement_submission_nonce=NONCE,replacement_run_uuid=RUN,replacement_source_sha=source_sha,control_id=CONTROL,created_epoch=now_epoch,predecessor_dual_closure=deepcopy(closure),automatic_retry=False)
 for v in (g,l):v[FIELD]=deepcopy(m)
 pair(g,l);c.checked(g);c.scope.validate_pair(g,l);return g,l
