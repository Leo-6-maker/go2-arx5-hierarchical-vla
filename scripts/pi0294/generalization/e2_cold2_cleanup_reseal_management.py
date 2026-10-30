"""Reseal only the unregistered cold02 stopped at the authorized-cleanup guard.

The earlier tentative bytes and source are archived, not hot patched. The same
unused request, original science, closed histories and gross costs remain.
"""

PROGRAM = r'''
import base64,hashlib,json,os,resource,sys,time
from copy import deepcopy
from pathlib import Path
P=json.loads(sys.stdin.read());began=time.process_time();sys.dont_write_bytecode=True
resource.setrlimit(resource.RLIMIT_CPU,(178,179));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
assert os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-')and not os.environ.get('PBS_JOBID')and os.environ.get('CUDA_VISIBLE_DEVICES')==''and time.time()<1793404800
raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
failed=Path(P['failed_stage']);eb=(failed/'OS_EXIT.json').read_bytes()
assert sha(eb)=='7ff18ee9c123dc5b0ad4d1ae0957a12153d35a6d026e56b230ae0336912a7b31'and json.loads(eb)['returncode']==1 and json.loads(eb)['timeout']is False
assert 'MINIMAL_PORTABLE_SAME_FILESYSTEM_PATHS_INODES_STATS_AND_FULL_CONTENTS'in(failed/'stderr').read_text()
for row,pk,sk in[(read(failed/'LAUNCHED.json'),'wrapper_pid','wrapper_procstart'),(read(failed/'CHILD_STARTED.json'),'pid','procstart')]:
 p=Path('/proc',str(row[pk]),'stat');f=p.read_text().rsplit(')',1)[1].split()if p.exists()else None
 assert not(f and f[0]not in('Z','X')and int(f[19])==int(row[sk]))
root=Path(P['root']);before_manifest=(root/'RUN_MANIFEST.json').read_bytes();assert sha(before_manifest)=='2801961013d2918ed7402c235c0f0fa5cacb05b2e9d7db4db10bc271850603f8'
tentative=json.loads(before_manifest);before_binding=(root/'OUTPUT_BINDING.private.json').read_bytes();tb=json.loads(before_binding)
assert root.name=='arx5-minimal-e2-only-raw-fixed512-02-cold02'and not any(p.is_symlink()for p in(root,*root.parents))
assert not Path(tb['lease']).exists()and not(root/'control/resident-owner').exists()and not list((root/'pbs').iterdir())
for name in('COLD_INPUT_AUDIT.private.json','COLD_FIXED_REQUEST_PREVIEW.private.json','COLD_FULL_FUTURE_PREVIEW.private.json','PROPOSED_GLOBAL_NOT_CANONICAL.private.json'):assert not(root/'control'/name).exists()
for name,pin in tentative['execution']['sealed_inputs'].items():assert sha((root/name).read_bytes())==pin
oldseal=read(root/'SOURCE_SEAL.json');oldcode=root/tentative['code']['capsule_relative']
assert oldseal['source_sha']==tentative['code']['vla_commit']=='84a8422bd6a0154090df8a2a471f66e02f1fe70c'
for name,pin in oldseal['files'].items():assert sha((oldcode/name).read_bytes())==pin
source=Path(P['source_root']);sb=(source/'SOURCE_SEAL.json').read_bytes();seal=json.loads(sb)
assert seal['source_sha']=='6ec5327e7f805fc9cd9b7ed17604dc489123ed4b'
for name,pin in seal['files'].items():
 assert not Path(name).is_absolute()and '..'not in Path(name).parts
 f=source/'code'/name;assert not f.is_symlink()and f.stat().st_uid==os.getuid()and f.stat().st_mode&0o777==0o400 and sha(f.read_bytes())==pin
sys.path[:0]=[str(source/'code'),str(source/'code/src')]
from scripts.pi0294.generalization import minimal_e2only_cold2 as n,minimal_e2only_execution as x,minimal_remaining_physical as inventory
from scripts.pi0294.generalization.minimal_e2only_cold2_prepare import metadata,lease,ENV,LOCAL,PACKET
from scripts.pi0294.generalization.minimal_saved512_inputs import Inputs
from scripts.pi0294.generalization.minimal_e2only_physical import Guard,CONTROL_GROWTH_RESERVE
from scripts.pi0294.pv1.model_inputs import preflight
from scripts.pi0294.dec1.source_cpu_once_io import commit
old=Path(P['old_root']);mb=(old/'RUN_MANIFEST.json').read_bytes();m=json.loads(mb);oldb=read(old/'OUTPUT_BINDING.private.json');lb=Path(oldb['lease']).read_bytes()
assert sha(mb)==n.OLD_MANIFEST and sha(lb)==n.LEASE and json.loads(lb)['status']=='CLOSED'
snapshot=P['snapshot'];closed=n.closed_role(P['closed_capture']['record']);current=inventory.prefix_inventory(old)
assert sha(raw(snapshot))==closed['snapshot_sha256']and inventory.identity(current)==inventory.identity(snapshot)==closed['snapshot_identity_sha256']
small={name:(old/name).read_bytes()for name in m['execution']['sealed_inputs']};assert all(sha(body)==m['execution']['sealed_inputs'][name]for name,body in small.items())
prior=P['old_request'];cfg=json.loads(small['MIX_OWNER_CONFIG.private.json']);assert not any(str(old)in body for body in cfg['programs'].values())
gp=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';gb=gp.read_bytes();local=base64.b64decode(P['LOCAL_base64'],validate=True)
scope=n.prepare_scope(gb,local,authority_bytes=base64.b64decode(P['authority_base64'],validate=True),closed_capture=P['closed_capture'],old_request=prior,old_manifest=m,old_physical_custody=json.loads(small[x.PHYSICAL]),source_seal_bytes=sb,frozen=m['code']['frozen_numerical_files'],source_readback=P['source_readback'],now_epoch=time.time())
regpath,envpath=Path(oldb['registry']),Path(oldb['environment']);regb,envb=regpath.read_bytes(),envpath.read_bytes();assert sha(regb)==P['registry_sha256']and sha(envb)==P['environment_sha256']
reg,env,binding=metadata(json.loads(regb),json.loads(envb),oldb,'0'*64)
for path,key in[(regpath,'E2-COLD02-REGISTER-ROLE'),(envpath,'E2-COLD02-ENV-PROFILE')]:assert not path.with_name(key+'.INTENT.json').exists()and not path.with_name(key+'.RECEIPT.json').exists()
oldenv=small[ENV];v=json.loads(oldenv);v['environment_registry_sha256']=sha(raw(env));small[ENV]=raw(v)
p=json.loads(small[x.PHYSICAL]);p['run_uuid']=n.RUN;p['retained_initial_bytes']+=max(snapshot['allocated_bytes'],current['allocated_bytes']);p['closed_pre_native_roots'].append(dict(root=str(old),closure=closed,snapshot_file=n.SNAPSHOT))
small[x.PHYSICAL]=raw(p);assert small[x.PHYSICAL]==(root/x.PHYSICAL).read_bytes()
small[n.BINDING]=raw(scope);small[n.SNAPSHOT]=raw(snapshot);v=json.loads(small[LOCAL]);v.update(LOCAL_base64=base64.b64encode(local).decode(),LOCAL_sha256=sha(local),source='ACTUAL_CANONICAL_CORE_LOCAL_READONLY',capture_is_fresh_native_authority=False);small[LOCAL]=raw(v)
for name in('SOURCE_SEAL.json','MIX_OWNER_CONFIG.private.json',PACKET):small.pop(name,None)
small[PACKET]=raw({name:sha(body)for name,body in small.items()})
request=deepcopy(prior);request.update(x.fixed_request_fields(2,2),source_sha=seal['source_sha'],pre_native_owner_CPU_upper_s=2600,closed_pre_native_role=closed,**{n.TAG:n.digest(scope)},physical_custody_sha256=sha(small[x.PHYSICAL]),frozen_input_packet_sha256=sha(small[PACKET]))
template=deepcopy(request);template.pop('run_manifest_sha256',None);cfg.update(source_sha=seal['source_sha'],request_template=template);small['MIX_OWNER_CONFIG.private.json']=raw(cfg);small['SOURCE_SEAL.json']=sb
manifest=deepcopy(m);manifest.update(run_uuid=n.RUN,output_root=str(root));manifest['code'].update(vla_commit=seal['source_sha'],capsule_relative='capsules/'+seal['source_sha']+'/code',source_seal_sha256=sha(sb));manifest['execution']['sealed_inputs']={name:sha(body)for name,body in small.items()}
binding['run_manifest_sha256']=sha(raw(manifest));request['run_manifest_sha256']=binding['run_manifest_sha256'];assert not Path(binding['lease']).exists()
# Before any role mutation, check the actual original inputs and complete gross
# physical lineage with the published cleanup-aware guard, retaining old costs.
Inputs(root,tentative,guard=lambda:None).verify_current();preflight(root,json.loads(small['NEURAL_BINDINGS.private.json']))
auditb=(old/'control/closed-nameerror-owner-01/COLD_INPUT_AUDIT.private.json').read_bytes();assert sha(auditb)==prior['cold_input_audit_sha256']and json.loads(auditb)['actual_twelve_reader_rows']==7212
audit=dict(status='PASS_ACTUAL_MINIMAL_FIRST512_COLD_INPUTS_NO_MODEL',source_sha=seal['source_sha'],run_manifest_sha256=sha(raw(manifest)),source_files=len(seal['files']),sealed_inputs=len(small),actual_twelve_reader_rows=7212,original_reader_qualification_reused=dict(audit_sha256=sha(auditb),own_input_bytes_reverified=True),new_GPU_PBS_model_query_update_physics=0,observed_epoch=time.time(),CPU_actual_s=time.process_time()-began)
request.update(cold_input_audit_sha256=sha(raw(audit)),created_epoch=time.time());x.request(request)
physical=Guard(root,request,manifest,lambda **_:None).inspect(next_output_upper_bytes=CONTROL_GROWTH_RESERVE)
proposal=n.propose(gb,local,scope=scope,manifest_bytes=raw(manifest),physical_bytes=small[x.PHYSICAL],request=request,environment_binding_bytes=(oldenv,small[ENV]))
assert gp.read_bytes()==gb and regpath.read_bytes()==regb and envpath.read_bytes()==envb
def put(path,body):
 path=Path(path);path.parent.mkdir(mode=0o700,parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,'wb')as f:f.write(body);f.flush();os.fsync(f.fileno())
archive=root/'control/TENTATIVE84_BEFORE_CLEANUP_RESEAL';assert not archive.exists();archive.mkdir(mode=0o700)
changes={name:body for name,body in small.items()if(root/name).read_bytes()!=body};changes.update({'RUN_MANIFEST.json':raw(manifest),'OUTPUT_BINDING.private.json':raw(binding)})
assert set(changes)=={'SOURCE_SEAL.json','MIX_OWNER_CONFIG.private.json',n.BINDING,PACKET,'RUN_MANIFEST.json','OUTPUT_BINDING.private.json'}
for index,(name,body)in enumerate(changes.items()):put(archive/(str(index)+'-'+Path(name).name),(root/name).read_bytes())
code=root/manifest['code']['capsule_relative'];assert not code.exists()
for name in seal['files']:put(code/name,(source/'code'/name).read_bytes())
for index,(name,body)in enumerate(changes.items()):
 path=root/name;before=path.read_bytes();key='E2-COLD02-CLEANUP-COLD-RESEAL-'+str(index)
 commit(path,expected_before_sha256=sha(before),proposed_bytes=body,request_id=key,intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_suffix('.owner.lock'));assert path.read_bytes()==body
assert all(sha((root/name).read_bytes())==pin for name,pin in manifest['execution']['sealed_inputs'].items())and all(sha((code/name).read_bytes())==pin for name,pin in seal['files'].items())
for name,body in [('COLD_INPUT_AUDIT.private.json',raw(audit)),('COLD_FIXED_REQUEST_PREVIEW.private.json',raw(request)),('COLD_FULL_FUTURE_PREVIEW.private.json',raw(proposal['full_future_proof'])),('PROPOSED_GLOBAL_NOT_CANONICAL.private.json',raw(proposal['proposed_global']))]:put(root/'control'/name,body)
put(binding['lease'],raw(lease(json.loads(lb),binding)))
for path,before,after,key in[(regpath,regb,raw(reg),'E2-COLD02-REGISTER-ROLE'),(envpath,envb,raw(env),'E2-COLD02-ENV-PROFILE')]:
 commit(path,expected_before_sha256=sha(before),proposed_bytes=after,request_id=key,intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_suffix('.owner.lock'));assert path.read_bytes()==after
assert gp.read_bytes()==gb
result=dict(status='ACTUAL_E2_COLD02_FINITE_PREPARATION_FULL14_NO_NATIVE',root=str(root),source_sha=seal['source_sha'],manifest_sha256=sha(raw(manifest)),scope_sha256=n.digest(scope),proposed_GLOBAL_sha256=sha(raw(proposal['proposed_global'])),physical_proof=physical,full14_shortfalls={},CPU_s=time.process_time()-began,observed_epoch=time.time(),new_model_GPU_PBS_physics=0,native_permission=False,canonical_adoption=False,automatic_retry=False,preserved_tentative_manifest_sha256=sha(before_manifest),failed_guard_OS_sha256=sha(eb),resealed_files=len(changes))
put(Path(P['stage'])/'RESULT.json',raw(result));print(json.dumps(result))
'''
