"""Finish only the actual cold02 preparation's relocated-audit failure.

The existing role/source/manifest stay immutable. Invoked by the original
finite CPU child after publication; no canonical ledger, model or qsub effect.
"""

PROGRAM = r'''
import hashlib,json,os,resource,sys,time
from pathlib import Path
from copy import deepcopy
P=json.loads(sys.stdin.read());began=time.process_time();sys.dont_write_bytecode=True
resource.setrlimit(resource.RLIMIT_CPU,(178,179));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
assert os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-')and not os.environ.get('PBS_JOBID')and os.environ.get('CUDA_VISIBLE_DEVICES')==''and time.time()<1793404800
raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
failed=Path(P['failed_stage']);eb=(failed/'OS_EXIT.json').read_bytes();fb=(failed/'FAILED_OR_UNKNOWN.json').read_bytes()
assert sha(eb)=='923d13b0c9803b50ca5a83fab8e531d7961c8284e9eef79518d2871755372ec4'and sha(fb)=='f3f41f77df1554e0a23669b112df415ea29f3ffe82d43887877d2385c1829396'
assert read(failed/'OS_EXIT.json')['returncode']==1 and read(failed/'OS_EXIT.json')['timeout']is False
assert read(failed/'FAILED_OR_UNKNOWN.json')['type']=='FileNotFoundError'
for row,pk,sk in[(read(failed/'LAUNCHED.json'),'wrapper_pid','wrapper_procstart'),(read(failed/'CHILD_STARTED.json'),'pid','procstart')]:
 p=Path('/proc',str(row[pk]),'stat');f=p.read_text().rsplit(')',1)[1].split()if p.exists()else None
 assert not(f and f[0]not in('Z','X')and int(f[19])==int(row[sk]))
root=Path(P['root']);mb=(root/'RUN_MANIFEST.json').read_bytes();assert sha(mb)=='2801961013d2918ed7402c235c0f0fa5cacb05b2e9d7db4db10bc271850603f8';m=json.loads(mb)
assert root.name=='arx5-minimal-e2-only-raw-fixed512-02-cold02'and root.is_absolute()and not any(p.is_symlink()for p in(root,*root.parents))
binding=read(root/'OUTPUT_BINDING.private.json');assert binding['run_manifest_sha256']==sha(mb)
assert not Path(binding['lease']).exists()and not(root/'control/resident-owner').exists()and not list((root/'pbs').iterdir())
source=root/m['code']['capsule_relative'];seal=read(root/'SOURCE_SEAL.json');assert seal['source_sha']==m['code']['vla_commit']=='84a8422bd6a0154090df8a2a471f66e02f1fe70c'
assert sha((root/'SOURCE_SEAL.json').read_bytes())==m['code']['source_seal_sha256']
for name,pin in seal['files'].items():assert sha((source/name).read_bytes())==pin
small={name:(root/name).read_bytes()for name in m['execution']['sealed_inputs']}
assert all(sha(body)==m['execution']['sealed_inputs'][name]for name,body in small.items())
sys.path[:0]=[str(source),str(source/'src')]
from scripts.pi0294.generalization import minimal_e2only_cold2 as n,minimal_e2only_execution as x
from scripts.pi0294.generalization.minimal_e2only_cold2_prepare import metadata,lease
from scripts.pi0294.generalization.minimal_saved512_inputs import Inputs
from scripts.pi0294.generalization.minimal_e2only_physical import Guard,CONTROL_GROWTH_RESERVE
from scripts.pi0294.generalization import minimal_remaining_physical as inventory
from scripts.pi0294.dec1.source_cpu_once_io import commit
scope=json.loads(small[n.BINDING]);old=Path(scope['old_manifest']['output_root']);oldb=read(old/'OUTPUT_BINDING.private.json');lb=Path(oldb['lease']).read_bytes()
assert sha(lb)==n.LEASE and inventory.identity(inventory.prefix_inventory(old))==scope['closed_role']['snapshot_identity_sha256']
assert read(failed/'FAILED_OR_UNKNOWN.json')['message']=="[Errno 2] No such file or directory: '"+str(old/'control/COLD_INPUT_AUDIT.private.json')+"'"
auditpath=old/'control/closed-nameerror-owner-01/COLD_INPUT_AUDIT.private.json';auditb=auditpath.read_bytes();old_audit=json.loads(auditb)
assert sha(auditb)==scope['old_request']['cold_input_audit_sha256']=='6fae118a970bc688fc613a82797e161d6d0a32758c09c9884c065b3d83cc994e'and old_audit['actual_twelve_reader_rows']==7212
regpath,envpath=Path(binding['registry']),Path(binding['environment']);regb,envb=regpath.read_bytes(),envpath.read_bytes()
assert sha(regb)==P['registry_sha256']and sha(envb)==P['environment_sha256']
reg,env,newb=metadata(json.loads(regb),json.loads(envb),oldb,sha(mb));assert newb==binding
envname='plans/A40_ENVIRONMENT_BINDING.private.json';oldenv=(old/envname).read_bytes();expected=json.loads(oldenv);expected['environment_registry_sha256']=sha(raw(env));assert raw(expected)==small[envname]
for path,key in[(regpath,'E2-COLD02-REGISTER-ROLE'),(envpath,'E2-COLD02-ENV-PROFILE')]:
 assert not path.with_name(key+'.INTENT.json').exists()and not path.with_name(key+'.RECEIPT.json').exists()
gp=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';gb=gp.read_bytes();assert sha(gb)==n.GLOBAL
assert not any(read(root/'MIX_OWNER_CONFIG.private.json')['request_template']['request_id']==r.get('request_id')for r in json.loads(gb)['PBS_submissions'])
Inputs(root,m,guard=lambda:None).verify_current()
audit=dict(status='PASS_ACTUAL_MINIMAL_FIRST512_COLD_INPUTS_NO_MODEL',source_sha=seal['source_sha'],run_manifest_sha256=sha(mb),source_files=len(seal['files']),sealed_inputs=len(small),actual_twelve_reader_rows=7212,
 original_reader_qualification_reused=dict(audit_sha256=sha(auditb),own_input_bytes_reverified=True),relocated_original_audit=True,new_GPU_PBS_model_query_update_physics=0,observed_epoch=time.time(),CPU_actual_s=time.process_time()-began)
request=deepcopy(read(root/'MIX_OWNER_CONFIG.private.json')['request_template']);request.update(run_manifest_sha256=sha(mb),cold_input_audit_sha256=sha(raw(audit)),created_epoch=time.time());x.request(request)
physical=Guard(root,request,m,lambda **_:None).inspect(next_output_upper_bytes=CONTROL_GROWTH_RESERVE)
import base64
local=base64.b64decode(json.loads(small['plans/MINIMAL_CORE_LOCAL_CAPTURE.private.json'])['LOCAL_base64'])
proposal=n.propose(gb,local,scope=scope,manifest_bytes=mb,physical_bytes=small[x.PHYSICAL],request=request,environment_binding_bytes=(oldenv,small[envname]))
assert gp.read_bytes()==gb and regpath.read_bytes()==regb and envpath.read_bytes()==envb
def put(path,body):
 path=Path(path);path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,'wb')as f:f.write(body);f.flush();os.fsync(f.fileno())
for name,body in [('COLD_INPUT_AUDIT.private.json',raw(audit)),('COLD_FIXED_REQUEST_PREVIEW.private.json',raw(request)),('COLD_FULL_FUTURE_PREVIEW.private.json',raw(proposal['full_future_proof'])),('PROPOSED_GLOBAL_NOT_CANONICAL.private.json',raw(proposal['proposed_global']))]:put(root/'control'/name,body)
put(binding['lease'],raw(lease(json.loads(lb),binding)))
for path,before,after,key in[(regpath,regb,raw(reg),'E2-COLD02-REGISTER-ROLE'),(envpath,envb,raw(env),'E2-COLD02-ENV-PROFILE')]:
 commit(path,expected_before_sha256=sha(before),proposed_bytes=after,request_id=key,intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_suffix('.owner.lock'))
 assert path.read_bytes()==after
assert(root/'RUN_MANIFEST.json').read_bytes()==mb and gp.read_bytes()==gb
assert all(sha((root/name).read_bytes())==pin for name,pin in m['execution']['sealed_inputs'].items())
result=dict(status='ACTUAL_E2_COLD02_FINITE_PREPARATION_FULL14_NO_NATIVE',root=str(root),source_sha=seal['source_sha'],manifest_sha256=sha(mb),scope_sha256=n.digest(scope),proposed_GLOBAL_sha256=sha(raw(proposal['proposed_global'])),physical_proof=physical,full14_shortfalls={},CPU_s=time.process_time()-began,observed_epoch=time.time(),new_model_GPU_PBS_physics=0,native_permission=False,canonical_adoption=False,automatic_retry=False,failed_preparation_OS_sha256=sha(eb),relocated_audit_sha256=sha(auditb))
put(Path(P['stage'])/'RESULT.json',raw(result));print(json.dumps(result))
'''
