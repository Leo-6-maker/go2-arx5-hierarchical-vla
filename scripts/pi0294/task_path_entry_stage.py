"""Finite original-entry qualification in one newly registered S1 directory.

Management program only. Uses the existing canonical registry and onceIO;
never starts a Teacher, solver, simulator, renderer or neural model. The
qualification lease closes after the original entry's interpreter/cwd probe.
"""

PROGRAM = r'''
import base64,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'and not os.environ.get('PBS_JOBID')
began=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
sha=lambda b:hashlib.sha256(b).hexdigest();raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
approval=base64.b64decode(P['approval_base64'],validate=True)
assert sha(approval)=='8a55153f728c0431116e2f9932f5286f6a0647d362252991712541638b27484f'
assert set(P['code_files'])=={'scripts/pi0294/task_path_entry.py','scripts/pi0294/dec1/source_cpu_once_io.py'}
code={name:base64.b64decode(v,validate=True)for name,v in P['code_files'].items()}
assert all(sha(b)==P['source_members'][name]for name,b in code.items())
entry_ns={};exec(compile(code['scripts/pi0294/task_path_entry.py'],'<published-S1-entry-binding>','exec'),entry_ns)
files={name:base64.b64decode(b,validate=True)for name,b in P['entry_files'].items()}
root=Path(P['root']);plan=entry_ns['prepare_standard_entry'](approval,files,run_root=str(root),source_sha=P['source_sha'],now_epoch=time.time())
registry=Path(P['registry']);before=registry.read_bytes();assert sha(before)==P['registry_before_sha256']
value=json.loads(before);old=[x for x in value['entries']if x.get('role')=='PI0294_TASK_DATA_FIRST_4080']
assert len(old)==1 and root.parent==Path(old[0]['path']).parent and not root.exists()and not root.is_symlink()
assert not any(x.get('role')==P['role']or x.get('path')==str(root)for x in value['entries'])
assert root.parent.is_dir()and root.parent.stat().st_uid==2009
mount=subprocess.run(['findmnt','-T',str(root.parent),'-J','-o','TARGET,SOURCE,FSTYPE'],capture_output=True,text=True,timeout=3)
assert mount.returncode==0 and json.loads(mount.stdout)['filesystems'][0]['fstype']=='ext4'
assert P['budget_proof']['shortfalls']=={} and P['budget_proof']['old_work_matrix_and_final_protection_preserved']is True
assert P['budget_proof']['S1_allowance_spendable_by_old_work']is False
assert P['budget_proof']['new_GR00T_queries']==P['budget_proof']['new_optimizer_updates']==P['budget_proof']['new_PBS_submissions']==0
assert P['request_sha256']==sha(raw(P['request']))and P['request']['purpose']=='ORIGINAL_ENTRY_PATH_QUALIFICATION_ONLY'
assert P['request']['scope_approval_sha256']==sha(approval)and P['request']['source_sha']==P['source_sha']
def put(path,b,mode=0o400):
 p=Path(path);b=b if isinstance(b,bytes)else raw(b)
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
 with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 assert p.read_bytes()==b
root.mkdir(mode=0o700)
manifest=P['manifest'];assert manifest['run_uuid']==root.name and manifest['owner']=='EXISTING_LOCAL_PC'
assert manifest['code']['vla_commit']==P['source_sha']and manifest['execution']['profile']=='4080-simulation'
put(root/'RUN_MANIFEST.json',manifest)
for name in plan['directories']:
 Path(name).mkdir(parents=True,mode=0o700,exist_ok=False)
capsule=root/'entry/code/vla'
for name,b in code.items():
 path=capsule/name;path.parent.mkdir(parents=True,mode=0o700,exist_ok=True);put(path,b)
(capsule/'src').mkdir(mode=0o700)
for name,b in files.items():put(root/'entry/infrastructure'/name,b,0o500 if name=='enter.sh'else 0o400)
seal=dict(source_sha=P['source_sha'],files=P['source_members'],capsule_purpose='ENTRY_QUALIFICATION_ONLY_NOT_COMPLETE_TASK_RUNTIME')
put(root/'SOURCE_SEAL.json',seal);put(root/'S1_HUMAN_APPROVAL.json',approval);put(root/'PREPARATION_REQUEST.json',P['request'])
put(root/'FULL14_PREPARATION_PROOF.json',P['budget_proof'])
lease_path=root/'control/ENTRY_QUALIFICATION_LEASE.json'
lease=dict(run_uuid=root.name,owner='EXISTING_LOCAL_PC',status='ACTIVE',purpose='ENTRY_PATH_QUALIFICATION_ONLY',
 approval_sha256=sha(approval),deadline_epoch=plan['deadline_epoch'],new_physics_authorized=False)
put(lease_path,lease)
scope=dict(schema='S1_APPROVED_FINITE_SCOPE_PREPARATION_ACCOUNT_V1',approval_sha256=sha(approval),
 original_GLOBAL_sha256=P['budget_observation']['GLOBAL']['sha256'],original_LOCAL_sha256=P['budget_observation']['LOCAL']['sha256'],
 deadline_epoch=plan['deadline_epoch'],starts=0,integrations=0,RGB_pairs=0,GR00T_queries=0,optimizer_updates=0,PBS=0,
 Core_CPU_upper_charged_s=60,PC_CPU_upper_charged_s=P['PC_CPU_upper_s'],
 transfer_upper_charged_bytes=P['transfer_upper_bytes'],Core_staging_upper_charged_bytes=P['Core_staging_upper_bytes'],
 approved_scope_caps=dict(starts=4,integrations=62012,RGB_pairs=2084,total_Core_and_PC_CPU_s=14400,
 Core_output_staging_bytes=8*1024**3,PC_evidence_bytes=1024**3,transfer_bytes=8*1024**3),
 old_costs_reset=False,old_shared_caps_rewritten=False,ten_day_extension_adopted=False,native_attempts_admitted=False)
assert scope['Core_CPU_upper_charged_s']+scope['PC_CPU_upper_charged_s']<=14400
assert P['transfer_upper_bytes']<=8*1024**3 and P['Core_staging_upper_bytes']<=8*1024**3
put(root/'control/S1_PREPARATION_ACCOUNT.json',scope)
once_ns={};exec(compile(code['scripts/pi0294/dec1/source_cpu_once_io.py'],'<unchanged-original-onceIO>','exec'),once_ns)
commit=once_ns['commit'];key='S1-ORIGINAL-ENTRY-REGISTER-20261029'
new=json.loads(before);new['entries'].append(dict(role=P['role'],path=str(root),run_uuid=root.name,
 executor_node='corelab-4080s',owner='EXISTING_LOCAL_PC',scope='APPROVED_S1_ENTRY_PREPARATION_ONLY',
 authority_sha256=sha(approval),source_sha=P['source_sha'],local_lease=str(lease_path)))
receipt=commit(registry,expected_before_sha256=sha(before),proposed_bytes=raw(new),request_id=key,
 intent_path=registry.with_name(key+'.INTENT.json'),receipt_path=registry.with_name(key+'.RECEIPT.json'),lock_path=registry.with_name('PATH_REGISTRY.owner.lock'))
after=registry.read_bytes();assert sha(after)==receipt['after_sha256']
entry_ns['registered_snapshot'](after,expected_sha256=sha(after),run_root=str(root),role=P['role'])
put(root/'control/CANONICAL_REGISTRY_READBACK.json',after)
put(root/'control/REGISTRATION_RECEIPT.json',receipt)
binding=dict(run_root=str(root),run_uuid=root.name,registered_parent=str(root.parent),role=P['role'],
 registry=str(registry),runtime_registry_snapshot=str(root/'control/CANONICAL_REGISTRY_READBACK.json'),
 registry_snapshot_sha256=sha(after),lease=str(lease_path),snapshot_is_new_registry=False)
put(root/'OUTPUT_BINDING.private.json',binding)
env={k:v for k,v in os.environ.items()if k not in('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT')}
env.update(TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),PYTHONNOUSERSITE='1',
 PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
probe='import os,sys,json;print(json.dumps(dict(host=os.uname().nodename,uid=os.getuid(),cwd=os.getcwd(),python=sys.executable,pythonpath=os.environ.get("PYTHONPATH"),sys_path=sys.path)))'
q=subprocess.run(['/bin/bash',plan['entry_path'],'4080-simulation','python','-B','-c',probe],env=env,capture_output=True,text=True,timeout=10,cwd=str(root))
assert q.returncode==0,q.stderr
actual=json.loads(q.stdout);assert actual['cwd']==plan['entry_cwd']and actual['python']==plan['profile_python']and actual['uid']==2009
for name in actual['sys_path']:
 p=Path(name)
 if p.is_absolute()and p.exists():
  qmount=subprocess.run(['findmnt','-T',str(p.resolve()),'-J','-o','TARGET,SOURCE,FSTYPE'],capture_output=True,text=True,timeout=3)
  assert qmount.returncode==0 and json.loads(qmount.stdout)['filesystems'][0]['fstype']=='ext4'
put(root/'outputs/ACTUAL_ORIGINAL_ENTRY_PROBE.json',actual)
old_lease=lease_path.read_bytes();lease['status']='CLOSED';lease['terminal_reason']='ENTRY_PROBE_COMPLETE_NO_NATIVE_TASK_ADMITTED'
close_key='S1-ORIGINAL-ENTRY-PROBE-CLOSE-20261029'
close_receipt=commit(lease_path,expected_before_sha256=sha(old_lease),proposed_bytes=raw(lease),request_id=close_key,
 intent_path=lease_path.with_name(close_key+'.INTENT.json'),receipt_path=lease_path.with_name(close_key+'.RECEIPT.json'),lock_path=lease_path.with_name('ENTRY_QUALIFICATION_LEASE.owner.lock'))
children=resource.getrusage(resource.RUSAGE_CHILDREN)
result=dict(status='ACTUAL_ORIGINAL_STANDARD_ENTRY_NODE_LOCAL_PROBE_PASS_ONLY',observed_epoch=time.time(),
 run_uuid=root.name,source_sha=P['source_sha'],entry_files_sha256=entry_ns['ENTRY_FILES'],
 source_members=P['source_members'],registry_before_sha256=sha(before),registry_after_sha256=sha(after),
 registration_receipt_sha256=sha(raw(receipt)),closed_lease_sha256=sha(lease_path.read_bytes()),
 actual_entry=actual,CPU_s=time.process_time()-began+children.ru_utime+children.ru_stime,
 Core_CPU_upper_charged_s=60,PC_CPU_upper_charged_s=P['PC_CPU_upper_s'],new_physics=0,new_neural_models=0,
 new_GR00T_queries=0,new_PBS=0,new_optimizer_updates=0,native_task_admission=False,
 full_runtime_dependencies_qualified=False,old_run_or_capsule_modified=False,new_global_registry=False)
put(root/'ENTRY_QUALIFICATION_RESULT.json',result)
print(json.dumps(result))
'''
