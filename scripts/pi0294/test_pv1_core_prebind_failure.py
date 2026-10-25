"""Captured actual failed unit, pure closeout; never a model/GPU release."""
import base64,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import evaluation_core_prebind_failure as s,evaluation_core_accounting as c
ROOT=Path(__file__).resolve().parents[2]
def fixture():
 p=ROOT.parent.parent/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01/B24_1000_RESIDENT_EXACT_NATIVE_1792891068091504848.private.json'
 if not p.exists():pytest.skip('private exact captured failed unit bytes unavailable')
 v=json.loads(p.read_bytes())['value'];g=json.loads(base64.b64decode(v['global_read']['global_base64']));l=json.loads(base64.b64decode(v['LOCAL_base64']))
 proof=dict(schema='PV1_CORE_PREBIND_SYSTEMD_KNOWN_ZERO_TERMINAL_V1',request_id=s.RID,request_sha256=s.REQUEST_SHA,source_sha=s.SOURCE,actual_NATIVE_PROCESS_FINAL_sha256=s.FINAL,native_unit='pv1-v4a-b4096-n294101-read-recovery-02-neural.service',native_unit_MainPID=0,native_unit_ActiveState='failed',native_exit_status=1,InvocationID='9908918021a640d4896fdc46c93d590c',ExecMainPID=258302,native_process_started_receipt_exists=False,native_model_or_phase_release_exists=False,all_actual_workers_terminal=True,stable_inventory=True,unknown_effects=[],actual_model_loads=0,actual_queries=0,actual_physics=0,actual_training_updates=0,started_epoch=1792891052.722064,finished_epoch=1792891053.2458,observed_epoch=1792891068,conservative_GPU_s=61,conservative_CPU_s=1,new_resident_allocated_upper_bytes=80*1024**2)
 return g,l,proof
def test_known_unbound_failure_settles_without_inventing_process_handle():
 g,l,p=fixture();z=s.propose(g,l,proof=p,now_epoch=p['observed_epoch']);a,b=z['proposed_global'],z['proposed_LOCAL'];r=a[c.FIELD][s.RID]
 assert z['IO_performed'] is z['new_admission'] is False and r['native_handle'] is None
 assert r['status']==c.STATES[3] and 'proc_start' not in r and r['model_load_attempts']==r['query_attempts']==0
 assert a['used']['H200_s']==g['used']['H200_s']+61 and a['CPU_s']==g['CPU_s']+1
 assert a['PBS_submissions']==g['PBS_submissions'] and a['optimizer_updates']==g['optimizer_updates']
 assert b['Core_native_GPU_reservations'][s.RID]['status']=='SETTLED_KNOWN_NATIVE_TERMINAL'
 with pytest.raises(PermissionError):s.propose(a,b,proof=p,now_epoch=p['observed_epoch'])
@pytest.mark.parametrize('key,value',[('actual_model_loads',1),('actual_queries',1),('unknown_effects',['x']),('native_unit_MainPID',1),('native_unit_ActiveState','active'),('InvocationID','f'*32),('ExecMainPID',1),('native_process_started_receipt_exists',True),('native_model_or_phase_release_exists',True),('stable_inventory',False),('conservative_GPU_s',60),('actual_NATIVE_PROCESS_FINAL_sha256','f'*64)])
def test_no_unknown_refund_no_reuse_no_qualification_alias(key,value):
 g,l,p=fixture();p[key]=value
 with pytest.raises(PermissionError):s.propose(g,l,proof=p,now_epoch=p['observed_epoch'])
