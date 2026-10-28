"""Known pre-qsub boundary and actual child environment, no native dispatch."""
import json,os,subprocess,sys
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_export_wrapper as w

def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
@pytest.fixture
def boundary(tmp_path):
 r=tmp_path/'campaign/children/cpu';c=r/'control';c.mkdir(parents=True)
 (r/'RUN_MANIFEST.json').write_bytes(b'original-manifest')
 (r/'control-code').mkdir();(r/'control-code/amd_export_owner.py').write_bytes(b'original-owner')
 (c/'CPU_OWNER_CONFIG.private.json').write_bytes(b'original-config')
 pin=dict(control_source_sha='f'*40,parent_manifest_sha256=w.sha(b'original-manifest'),
  owner_code_sha256=w.sha(b'original-owner'),configuration_sha256=w.sha(b'original-config'))
 put(c/'CPU_OWNER_BINDING.private.json',pin)
 put(c/'CPU_OWNER_OS_EXIT.json',dict(returncode=1,control_source='f'*40,site_control_CPU_s=10.287214531))
 (c/'CPU_OWNER.stderr').write_text('Traceback\nPermissionError: '+w.FAILURE+'\n')
 for i,n in enumerate(('CPU_OWNER_WRAPPER_STARTED.json','CPU_OWNER_LAUNCHED_PROCESS.json')):put(c/n,dict(pid=11+i,procstart='123'))
 put(r.parent.parent/'HOPPER_RESOURCE_LEDGER.json',dict(PBS_submissions=[dict(request_id='same-request',job_id=None,run_manifest_sha256=pin['parent_manifest_sha256'])]))
 lease=tmp_path/'lease.json';put(lease,dict(status='ACTIVE'));put(r/'OUTPUT_BINDING.private.json',dict(lease=str(lease)))
 put(c/w.REPAIR/'BINDING.private.json',dict(request_id='same-request',expected_GLOBAL_sha256=w.sha((r.parent.parent/'HOPPER_RESOURCE_LEDGER.json').read_bytes()),old_OS_exit_sha256=w.sha((c/'CPU_OWNER_OS_EXIT.json').read_bytes()),old_stderr_sha256=w.sha((c/'CPU_OWNER.stderr').read_bytes()),old_control_CPU_s=10.287214531,additional_prepay=0,original_control_upper_CPU_s=600))
 return r

def test_same_request_preserves_all_original_bytes(boundary):
 before={str(p):p.read_bytes()for p in boundary.parent.parent.rglob('*')if p.is_file()}
 repair,pin=w.previous_boundary(boundary,current=lambda pid:None)
 assert repair.name==w.REPAIR and pin['request_id']=='same-request'
 assert before=={str(p):p.read_bytes()for p in boundary.parent.parent.rglob('*')if p.is_file()}

@pytest.mark.parametrize('bad',['live','journal','release','qsub','changed_GLOBAL','wrong_failure','source','cost'])
def test_fail_closed_before_child_or_journal(boundary,bad):
 c=boundary/'control'
 if bad=='journal':(c/'original-cpu-owner').mkdir()
 elif bad=='release':put(c/'CPU_RUNNING_RELEASE.json',{})
 elif bad=='qsub':(boundary/'pbs/alloc-01').mkdir(parents=True)
 elif bad=='changed_GLOBAL':put(boundary.parent.parent/'HOPPER_RESOURCE_LEDGER.json',{})
 elif bad=='wrong_failure':(c/'CPU_OWNER.stderr').write_text('different failure')
 elif bad=='source':(boundary/'control-code/amd_export_owner.py').write_text('changed')
 elif bad=='cost':
  p=c/w.REPAIR/'BINDING.private.json';v=json.loads(p.read_bytes());v['old_control_CPU_s']=0;put(p,v)
 with pytest.raises(PermissionError):w.previous_boundary(boundary,current=lambda pid:'123'if bad=='live'else None)
 assert not(c/w.REPAIR/'CPU_OWNER_LAUNCHED_PROCESS.json').exists()

def test_real_child_receives_all_offline_flags_without_parent_mutation():
 original=dict(os.environ,HF_HUB_OFFLINE='0');before=original.copy()
 env=w.environment(original)
 v=subprocess.run([sys.executable,'-c','import json,os; print(json.dumps({k:os.environ.get(k)for k in '+repr(list(w.OFFLINE))+'}))'],env=env,capture_output=True,text=True,check=True)
 assert json.loads(v.stdout)==w.OFFLINE and original==before
