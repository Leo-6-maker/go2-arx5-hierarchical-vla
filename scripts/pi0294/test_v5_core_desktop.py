"""Exact desktop exception CPU checks; actual native GPU admission is separate."""
from copy import deepcopy
from unittest.mock import patch
import pytest,time
from scripts.pi0294.pv1 import mix_core_desktop as n,evaluation_core_identity as original,successor_cd as s
from scripts.pi0294.test_pv1_core_evaluation_identity import DEVICE

def args():
 r=dict(authority_release=s.TASK_SOURCE,scope_grant_sha256=s.GRANT_SHA,GPU_uuid=n.GPU,
   evaluation_candidate='C50',noise_bank=294101,remaining_conditions=24,native_platform='CORELAB_RTX4080',optimizer_update_limit=0,optimizer_updates=0,**{n.FIELD:n.GRANT_SHA})
 p=dict(boot_id=n.BOOT,observed_epoch=time.time(),present=True,pid=n.PID,uid=n.UID,proc_start=n.START,state='S',comm='nxnode.bin')
 line=f'{n.GPU}, {n.PID}, {n.NAME}, 302\n'
 return r,p,line

def test_only_exact_desktop_removed_and_raw_receipt_retained():
 r,p,line=args();out,proof=n.filter_processes(line,r,proof=p,now_epoch=p['observed_epoch']);assert out==''and proof['authorized_desktop_rows'][0]['pid']==n.PID

@pytest.mark.parametrize('key,value',[('uid',2009),('proc_start',0),('boot_id','new'),('state','Z'),('comm','python')])
def test_wrong_or_reused_process_refused(key,value):
 r,p,line=args();p[key]=value
 with pytest.raises(PermissionError):n.filter_processes(line,r,proof=p,now_epoch=p['observed_epoch'])

@pytest.mark.parametrize('line',[f'{n.GPU}, {n.PID}, /tmp/python, 302\n',f'{n.GPU}, {n.PID}, {n.NAME}, 1025\n'])
def test_wrong_executable_or_memory_growth_refused(line):
 r,p,_=args()
 with pytest.raises(PermissionError):n.filter_processes(line,r,proof=p,now_epoch=p['observed_epoch'])

def test_process_absent_does_not_require_desktop_remain_alive():
 r,p,_=args();p['present']=False;out,proof=n.filter_processes('',r,proof=p,now_epoch=p['observed_epoch']);assert out==''and proof['authorized_desktop_rows']==[]

def test_stale_kernel_snapshot_refused():
 r,p,line=args()
 with pytest.raises(PermissionError):n.filter_processes(line,r,proof=p,now_epoch=p['observed_epoch']+121)

@pytest.mark.parametrize('candidate',['D25','C50'])
def test_original_parser_allows_only_new_explicit_desktop_grant(candidate):
 r,p,line=args();r['evaluation_candidate']=candidate
 with patch.object(n,'capture',return_value=p):
  dev=original.parse_GPU_rows(DEVICE,line,GPU_uuid=n.GPU,desktop_request=r)
 assert dev['exact_desktop_coexistence']['grant_sha256']==n.GRANT_SHA
 with pytest.raises(PermissionError):original.parse_GPU_rows(DEVICE,line,GPU_uuid=n.GPU)

def test_other_foreign_compute_still_refused():
 r,p,line=args();line+=f'{n.GPU}, 1234, /usr/bin/python, 1000\n'
 with patch.object(n,'capture',return_value=p):
  with pytest.raises(PermissionError):original.parse_GPU_rows(DEVICE,line,GPU_uuid=n.GPU,desktop_request=r)

def test_no_old_b24_or_other_science_uses_desktop_exception():
 r,p,line=args()
 for key,value in [('evaluation_candidate','B_COVERAGE'),(n.FIELD,'0'*64),('optimizer_update_limit',1)]:
  bad=dict(r,**{key:value})
  with pytest.raises(PermissionError):n.filter_processes(line,bad,proof=p,now_epoch=p['observed_epoch'])


def test_only_current_own_model_context_is_allowed_after_loading():
 r,p,line=args();line+=f'{n.GPU}, 1234, /usr/bin/python, 16000\n'
 with patch.object(n,'capture',return_value=p):
  dev=original.parse_GPU_rows(DEVICE,line,GPU_uuid=n.GPU,desktop_request=r,own_pid=1234)
  assert dev['exact_desktop_coexistence']['grant_sha256']==n.GRANT_SHA
  with pytest.raises(PermissionError):original.parse_GPU_rows(DEVICE,line,GPU_uuid=n.GPU,desktop_request=r,own_pid=5678)


def recorded():
 r,proof,line=args();proof['observed_epoch']-=1200
 raw=dict(devices=dict(stdout=DEVICE,returncode=0),processes=dict(stdout=line,returncode=0))
 with patch('time.time',return_value=proof['observed_epoch']):
  device=original.parse_GPU_rows(DEVICE,line,GPU_uuid=n.GPU,desktop_request=r,desktop_proof=proof)
 p=dict(boot_id=n.BOOT,started_epoch=proof['observed_epoch']-.1,
        observed_epoch=proof['observed_epoch']+.1,pid=12345,device=device,native_GPU_raw=raw)
 return r,p,proof,line


def test_saved_initial_device_matches_after_600s_without_live_permission():
 r,p,proof,line=recorded()
 assert n.recorded_device(p,r)==p['device']
 with pytest.raises(PermissionError,match='FRESH_ACTUAL_KERNEL_READ'):
  original.parse_GPU_rows(DEVICE,line,GPU_uuid=n.GPU,desktop_request=r,desktop_proof=proof)


@pytest.mark.parametrize('change',['wrong-boot','unbound-earlier','future-proof','old-at-recording','wrong-uid','new-foreign','short-memory'])
def test_historical_read_still_requires_exact_bound_initial_observation(change):
 r,p,proof,line=recorded()
 if change=='wrong-boot':p['boot_id']='wrong'
 elif change=='unbound-earlier':p['started_epoch']=proof['observed_epoch']+1
 elif change=='future-proof':proof['observed_epoch']=p['observed_epoch']+1
 elif change=='old-at-recording':p['observed_epoch']=proof['observed_epoch']+121
 elif change=='wrong-uid':proof['uid']=999
 elif change=='new-foreign':p['native_GPU_raw']['processes']['stdout']+=f'{n.GPU}, 9999, /usr/bin/python, 100\n'
 else:p['native_GPU_raw']['devices']['stdout']=DEVICE.replace('32760','16376')
 with pytest.raises(PermissionError):n.recorded_device(p,r)
