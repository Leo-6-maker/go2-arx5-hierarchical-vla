"""Exact one-request human clock exception; CPU fixtures are not native GO."""
import base64,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import evaluation_core_B24_1000 as s,evaluation_core_accounting as c
ROOT=Path(__file__).resolve().parents[2]
GRANT=ROOT/'reports/PI0294/design/parallel-science-20261022/V4A_B24_ONLY_1000_EXACT_HUMAN_AUTHORIZATION_20261025.json'
CONTROL='feed1234-1234-4234-8234-123456789abc'
def request():return dict(request_id=s.RID,run_uuid=s.RUN,evaluation_candidate='B_COVERAGE',stage='P4_DEV',noise_bank=294101,late_start_grant_sha256=s.GRANT_SHA)
def test_exact_grant_and_original_global_deadline_remain():
 assert s.grant(GRANT.read_bytes())['latest_native_registration_and_start_epoch']==s.LATEST
 assert s.start_limit({})==c.old.STOP_NEW==s.ORIGINAL_STOP_NEW
 assert s.start_limit(request())==s.LATEST
 with pytest.raises(PermissionError):s.grant(GRANT.read_bytes()+b' ')
@pytest.mark.parametrize('key,value',[('request_id',CONTROL),('run_uuid','other'),('evaluation_candidate','A_CONTROL'),('stage','P5'),('noise_bank',294102),('late_start_grant_sha256','f'*64),('queue_id',CONTROL)])
def test_no_other_request_arm_panel_or_queue_gets_late_time(key,value):
 r=request();r[key]=value
 with pytest.raises(PermissionError):s.start_limit(r)
def test_request_requires_both_sealed_exact_grant_and_actual_adoption():
 r=request();r['sealed_inputs']={s.PLAN:s.GRANT_SHA};m={'execution':{'sealed_inputs':{s.PLAN:s.GRANT_SHA}}}
 assert s.check_request(m,r)==s.LATEST
 m['execution']['sealed_inputs'][s.PLAN]='f'*64
 with pytest.raises(PermissionError):s.check_request(m,r)
def captured():
 p=ROOT.parent.parent/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01/POST_STOP_NEW_EXACT_NATIVE_1792890212484119550.private.json'
 if not p.exists():pytest.skip('private captured actual dual closure bytes unavailable')
 x=json.loads(p.read_bytes())['value']
 return json.loads(base64.b64decode(x['global_base64'])),json.loads(base64.b64decode(x['LOCAL_base64']))
def adopted():
 g,l=captured();return s.adopt(g,l,GRANT.read_bytes(),source_sha='f'*40,control_id=CONTROL,now_epoch=s.LATEST-1200)
def test_captured_actual_costs_unchanged_effects_once_and_no_IO():
 before,local=captured();g,l=adopted();assert s.validate_pair(g,l)==g[s.FIELD]
 for k in ('used','reserved','PBS_submissions','optimizer_updates','training_runs',c.FIELD):assert g[k]==before[k]
 assert g['CPU_s']==before['CPU_s']+7200 and g['new_storage_current_bytes']==before['new_storage_current_bytes']+64*1024**2
 assert l['network_transfer_accounted_bytes']==local['network_transfer_accounted_bytes']+256*1024**2
 for site in ('CoreLab','Hopper','local_PC'):assert l['CPU_s'][site]==local['CPU_s'][site]+7200
 assert g['V4_DEV_control_reservations'][s.PREP]['status']=='CLOSED'
 with pytest.raises(PermissionError):s.adopt(g,l,GRANT.read_bytes(),source_sha='f'*40,control_id=CONTROL,now_epoch=s.LATEST-1100)
@pytest.mark.parametrize('now',[s.LATEST,s.LATEST+1,float('nan'),True])
def test_expired_invalid_real_time_denied(now):
 g,l=captured()
 with pytest.raises(PermissionError):s.adopt(g,l,GRANT.read_bytes(),source_sha='f'*40,control_id=CONTROL,now_epoch=now)
@pytest.mark.parametrize('fault',['dual','cost','transfer','source','orphan'])
def test_marker_cost_and_exact_boundary_cannot_drift(fault):
 g,l=adopted()
 if fault=='dual':l[s.FIELD]['source_sha']='e'*40
 if fault=='cost':g['CPU_s']-=7200
 if fault=='transfer':l['network_transfer_accounted_bytes']-=256*1024**2
 if fault=='source':g[s.FIELD]['source_sha']='invalid'
 if fault=='orphan':del g[s.FIELD];g[c.FIELD][s.RID]=request()
 with pytest.raises(PermissionError):s.validate_pair(g,l)
