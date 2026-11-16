import json,multiprocessing,time,threading
from pathlib import Path
import numpy as np
import pytest
from scripts.pi0292.durable import write,OwnerLock
from go2arx5_vla.world_model.durable_control_v2 import short_lock,read_control,valid_dispatch,launch_phase
from go2arx5_vla.world_model.mfree_recovery_v2 import register,RecoveryLedger
from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze

def hold(path,queue):
 with short_lock(path):queue.put(True);time.sleep(.25)

def test_two_process_registration_contention_is_bounded(tmp_path):
 q=multiprocessing.Queue();p=multiprocessing.Process(target=hold,args=(tmp_path/'lock',q));p.start();q.get(timeout=3);t=time.monotonic()
 with short_lock(tmp_path/'lock',timeout=2):assert time.monotonic()-t>=.15
 p.join();assert p.exitcode==0

def test_read_before_publish_waits_but_corrupt_does_not_retry(tmp_path):
 p=tmp_path/'DISPATCH';t=threading.Thread(target=lambda:(time.sleep(.1),write(p,dict(schema='MFREE_RECOVERY_DISPATCH_V2',generation='g',ready=True,epoch=time.time(),collectors=[]))))
 t.start();assert valid_dispatch(p,'g',timeout=1)['ready'];t.join()
 p.write_text('{corrupt')
 with pytest.raises(json.JSONDecodeError):read_control(p)
 write(p,dict(schema='other',generation='g',ready=True,epoch=time.time(),collectors=[]))
 with pytest.raises(ValueError):valid_dispatch(p,'g')

def spec_at(tmp_path):
 control=tmp_path/'world';control.mkdir();case=tmp_path/'case';case.mkdir();(case/'control').mkdir();reg=tmp_path/'registry';write(reg,dict(entries=[]))
 parent=dict(world_control=str(control),registry=str(reg),campaign_root=str(tmp_path));value=dict(owner='owner',source_sha='sha',role='role',deadline_epoch=time.time()+3600,ROS_DOMAIN_ID=198)
 return parent,case,value

def test_registered_exit_without_launch_and_idempotent_partial_registration(tmp_path):
 parent,case,s=spec_at(tmp_path);register(parent,[(case,s)])
 assert launch_phase(case,None,parent_cgroup_empty=True)=='REGISTERED_NOT_LAUNCHED'
 register(parent,[(case,s)]);assert len(read_control(parent['registry'])['entries'])==1
 # Confirmed completed imports do not acquire a second owner.
 write(case/'RESULT.private.json',dict(completed=True));assert launch_phase(case,dict(status='COMPLETE'),parent_cgroup_empty=True)=='TERMINAL_IMPORT_ONLY'

def test_popen_intent_without_running_is_never_relaunched(tmp_path):
 case=tmp_path/'case';case.mkdir()
 assert launch_phase(case,dict(status='INTENT'),parent_cgroup_empty=True)=='UNKNOWN_LAUNCH_INTENT_NO_REPLAY'
 assert launch_phase(case,None,parent_cgroup_empty=False)=='WAIT_OWNER_RECONCILIATION'

def test_technical_continuation_keeps_old_condition_and_budget(tmp_path):
 ledger=RecoveryLedger(tmp_path);ledger.create([],deadline_epoch=time.time()+7200)
 with ledger.transaction()as v:
  v['attempts']=[dict(id='72H-F1-T3-A003',state='CONFIRMED_NOT_STARTED',worker='old',cell='F1-T3',family='F1',layout_id='T3',ordinal=3,reset=10603,seed=123,physical_requests=0,native_integrations=0,reconciliation=dict(original='UNKNOWN'))]
  v['snapshots']['C20']={}
 r=ledger.claim('new');assert r['id'].endswith('-TECH1')and(r['reset'],r['seed'])==(10603,123)
 with pytest.raises(PermissionError):ledger.claim('new')
 v=read_control(ledger.path);assert len(v['attempts'])==2 and v['attempts'][0]['state']=='CONFIRMED_NOT_STARTED'
 assert v['attempts'][0]['reconciliation']['original']=='UNKNOWN'

def test_exact_H40_internal_and_boundary_acceleration_attribution():
 raw=np.tile(np.r_[0.,0.,0.,1.,0.,0.,0.,1.,0.,.5],(40,1));state=raw[0].copy()
 assert analyze(raw,state,np.eye(4),np.eye(4))['original_full_H40_pass']
 raw[20:,0]=.01;d=analyze(raw,state,np.eye(4),np.eye(4));assert not d['original_full_H40_pass']
 assert d['first_temporal_exceedance']['region']=='SUFFIX'
