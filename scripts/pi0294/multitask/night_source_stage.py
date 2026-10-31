"""Cold admission through the original registry/LOCAL account; no physics.

The existing root owner supplies fully captured original ledgers and an
already closed PC preparation-custody record. Runtime uses one local account,
one local lease and a sealed readback of the original canonical registry.
"""

PROGRAM=r'''
import base64,gzip,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'and not os.environ.get('PBS_JOBID')
resource.setrlimit(resource.RLIMIT_CPU,(160,160));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});started=time.process_time()
raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();sha=lambda b:hashlib.sha256(b).hexdigest()
root=Path(P['root']);registry=Path(P['registry']);local_path=Path(P['LOCAL_path'])
suffix='-night-source-book01'if P.get('task_kind','BOOK')=='BOOK'else'-night-source-four01'
assert root.is_absolute()and root.parent.name=='PI0294'and root.name.endswith(suffix)and not root.exists()
rb=registry.read_bytes();lb=local_path.read_bytes()
assert sha(rb)==P['registry_sha256']and sha(lb)==P['LOCAL_sha256']
reg=json.loads(rb);local=json.loads(lb)
assert local['pending_queries']==local['pending_updates']==local['unknown_effects']==[]
assert not any(x.get('role')==P['role']or x.get('path')==str(root)for x in reg['entries'])
assert 'NIGHT_SOURCE30_BOUNDED_AUTONOMY'not in local or P.get('predecessor_root')==str(Path(local['NIGHT_SOURCE30_BOUNDED_AUTONOMY']['account_path']).parents[1])
for unit in P['idle_D25_units']:
 q=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','LoadState'],capture_output=True,text=True,timeout=4)
 assert dict(x.split('=',1)for x in q.stdout.splitlines()if '='in x)==dict(MainPID='0',LoadState='not-found')
q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
assert q.returncode==0
if not P.get('gpu0_shared'):assert not q.stdout.strip()
else:
 assert P['execution_policy_sha256']=='b57b75cacdc3d8611b9f8d13cd5f4f4e15ed0d4dccd9a69b7469bc62f4c9b99a'
 q=subprocess.run(['nvidia-smi','--id=0','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
 assert q.returncode==0
 gpu_uuid,free_mib=q.stdout.strip().split(',');assert gpu_uuid==P['GPU_uuid']and int(free_mib)>=2048
for unit in P.get('idle_owned_units',[]):
 q=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID'],capture_output=True,text=True,timeout=4)
 assert q.returncode==0 and q.stdout.strip()=='MainPID=0'
disk=os.statvfs(root.parent);assert disk.f_bavail*disk.f_frsize>=20*1024**3
for line in Path('/proc/self/mountinfo').read_text().splitlines():
 before,after=line.split(' - ',1)
 if before.split()[4]=='/var/cache':assert after.split()[0]in('ext4','xfs','btrfs')
if 'source_delta'in P:
 donor=Path(P['predecessor_root'])/'entry/code/vla'
 delta={n:base64.b64decode(b,validate=True)for n,b in P['source_delta'].items()}
 assert set(delta)<=set(P['source_members'])and donor.is_dir()and not donor.is_symlink()
 code={n:delta[n]if n in delta else(donor/n).read_bytes()for n in P['source_members']}
else:code={n:base64.b64decode(b,validate=True)for n,b in json.loads(gzip.decompress(base64.b64decode(P['source_gzip_base64']))).items()}
assert all(n.endswith('.py')and not Path(n).is_absolute()and '..'not in Path(n).parts for n in code)
assert {n:sha(b)for n,b in code.items()}==P['source_members']
bundle_ns={};exec(compile(code['scripts/pi0294/multitask/source_stage_bundle.py'],'<pinned-source-bundle>','exec'),bundle_ns)
for name,row in P['libero_binary_files'].items():
 if 'base64'not in row:
  asset=Path(P['predecessor_root'])/'assets'/name
  assert asset.is_file()and not asset.is_symlink()
  body=asset.read_bytes();assert len(body)==row['bytes']and sha(body)==row['sha256']
  row['base64']=base64.b64encode(body).decode()
bundle=bundle_ns['unpack'](P)
once_ns={};exec(compile(code['scripts/pi0294/dec1/source_cpu_once_io.py'],'<original-onceIO>','exec'),once_ns)
put=once_ns['_exclusive'];commit=once_ns['commit']
scope_ns={};exec(compile(code['scripts/pi0294/multitask/night_scope.py'],'<published-night-scope>','exec'),scope_ns)
approval=base64.b64decode(P['approval_base64']);extension=base64.b64decode(P['successor_authority_base64'],validate=True)if P.get('successor_authority_base64')else None
assert extension is None or bundle['task_kind']in('ALL_PINNED','ROBOCASA_CABINET')or P.get('book_route_revision')
v7_policy=base64.b64decode(P['V7_policy_base64'],validate=True)if P.get('v7_finite_source_job')else None
if v7_policy is not None:
 scope_ns['authority'](approval,now=time.time(),v7_policy_body=v7_policy,job=P['v7_finite_source_job'])
else:scope_ns['authority'](approval,now=time.time(),successor_body=extension)
pc=base64.b64decode(P['PC_closed_account_base64']);account=json.loads(pc)
scope_ns['validate_account'](account)
assert account['custody']in('CLOSED_CARRIED_TO_NATIVE_OWNER_PENDING_INDEPENDENT_READBACK','V7_CLOSED_SOURCE_CUSTODY_PENDING_NATIVE_READBACK')
assert account['native_adopted']is False
if scope_ns.get('V7_CARRY')in account:
 assert account[scope_ns['V7_CARRY']]['destination']==str(root)
 if v7_policy is not None:assert account[scope_ns['V7_FINITE']]==P['v7_finite_source_job']
elif bundle['task_kind']=='ROBOCASA_CABINET':
 if P.get('cabinet_port_revision')=='ROBOCASA_ARX5_PORT05_SLOWER_TURN_REFERENCE':
  assert 'closed_cabinet_turn_speed_port05_carry'in account and account['charged_upper']['attempts']==5
 elif P.get('cabinet_port_revision')=='ROBOCASA_ARX5_PORT04_MEASURED_HINGE_ANCHOR':
  assert 'closed_cabinet_hinge_port04_carry'in account and account['charged_upper']['attempts']==4
 elif P.get('cabinet_port_revision')=='ROBOCASA_ARX5_PORT03_FIXED_CAMERA_HOME':
  assert 'closed_cabinet_camera_port03_carry'in account and account['charged_upper']['attempts']==3
 elif P.get('cabinet_port_revision')=='ROBOCASA_ARX5_PORT02_GEOMETRY_REPAIR':
  assert 'closed_cabinet_geometry_repair02_carry'in account and account['charged_upper']['attempts']==2
 else:assert 'closed_book_unrun_cabinet_carry'in account and account['charged_upper']['attempts']==1
else:assert account['charged_upper']['attempts']==0
assert P['PC_closed_account_sha256']==sha(pc)and account['native_destination']==str(root)
assert all(time.time()-P['observation'][k]['observed_epoch']<120 for k in('GLOBAL','LOCAL'))
root.mkdir(mode=0o700)
for name in('entry/infrastructure','entry/code/vla','control','tmp','cache','logs','outputs','inputs','assets'):
 (root/name).mkdir(parents=True,mode=0o700,exist_ok=True)
capsule=root/'entry/code/vla'
for n,b in code.items():
 path=capsule/n;path.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
 if 'source_delta'in P and n not in delta:os.link(donor/n,path)
 else:put(path,b)
sys.path[:0]=[str(capsule),str(capsule/'src')]
from scripts.pi0294.multitask import night_scope as scope
if extension is not None:put(root/'SOURCE_SUCCESSOR_AUTHORITY.json',extension)
if v7_policy is not None:
 put(root/'SOURCE_V7_EXECUTION_POLICY.md',v7_policy)
 put(root/'SOURCE_V7_FINITE_JOB.json',raw(P['v7_finite_source_job']))
scope=scope.for_root(root)
from scripts.pi0294.task_path_entry import ENTRY_FILES,node_local_paths,registered_snapshot
gb=gzip.decompress(base64.b64decode(P['GLOBAL_gzip_base64']));assert sha(gb)==P['observation']['GLOBAL']['sha256']
if scope.V7_CARRY in account:
 from scripts.pi0294.generalization.minimal_scope import first512_forecast
 budget=dict(original_forecast=first512_forecast(gb,lb),
  V7_project_internal_shortfalls_are_observations_not_new_run_approval_gates=True,
  old_balances_and_protected_work_unchanged=True,source_pool_caps_unchanged=True)
 if v7_policy is not None:budget.update(v7_finite_source_job=P['v7_finite_source_job'],historical_source_caps_retained_not_reissued=True)
 storage=budget['original_forecast']['dimensions']['Core_storage_bytes']
 future=max(0,storage['projected']-storage['used'])
 retained_and_next=account['charged_upper']['Core_peak_bytes']+536870912+67108864
 source_capacity=max(5368709120,((retained_and_next+1073741823)//1073741824)*1073741824)
 free=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
 assert free>=future+source_capacity+2147483648
 put(root/'ACTUAL_SOURCE_CAPACITY.json',raw(dict(old_future_bytes=future,source_retained_and_next_upper_bytes=retained_and_next,source_capacity_upper_bytes=source_capacity,free_bytes=free,margin_bytes=2147483648,financial_pool_caps_unchanged=True)))
else:
 budget=scope.forecast(approval,gb,lb,observation=P['observation'],now=time.time())
 assert budget==P['full14_proof']
put(root/'FULL14_PROOF.json',raw(budget));put(root/'ORIGINAL_GLOBAL_CAPTURE.private.json',gb)
put(root/'CLOSED_PC_PREPARATION_ACCOUNT.json',pc);put(root/'NIGHT_HUMAN_APPROVAL.json',approval)
put(root/'SOURCE_SEAL.json',raw(dict(source_sha=P['source_sha'],files=P['source_members'])))
for name,encoded in P['entry_files'].items():
 b=base64.b64decode(encoded);assert sha(b)==ENTRY_FILES[name]
 put(root/'entry/infrastructure'/name,b,0o500 if name=='enter.sh'else 0o400)
assets=dict(robot_root=P['robot_asset_root'],files={},libero_binary_names=list(P['libero_binary_files']))
for name,pin in P['robot_asset_files'].items():
 path=Path(P['robot_asset_root'])/name;b=path.read_bytes();assert sha(b)==pin['sha256']and len(b)==pin['bytes']
 assets['files']['robot/'+name]=dict(path=str(path),sha256=sha(b),bytes=len(b))
for name,b in bundle['binaries'].items():
 path=root/'assets'/name;path.parent.mkdir(parents=True,mode=0o700,exist_ok=True);put(path,b)
 assets['files'][name]=dict(path=str(path),sha256=sha(b),bytes=len(b))
for name,b in bundle['definitions'].items():
 path=root/'inputs'/name;path.parent.mkdir(parents=True,mode=0o700,exist_ok=True);put(path,b)
 assets['files']['definition/'+name]=dict(path=str(path),sha256=sha(b),bytes=len(b))
assets.update(robocasa_definition_members=bundle['robocasa_definition_members'],robocasa_texture_members=bundle['robocasa_texture_members'])
put(root/'ASSET_BINDINGS.private.json',raw(assets));put(root/'FIXED_TASK_CARDS.json',raw(P['cards']))
put(root/'ORIGINAL_ACTION_SEMANTICS_ACTUAL.json',raw(P['action_semantics']))
put(root/'SOURCE_QUALIFICATION.json',raw(P['qualification']))
lease_path=root/'control/NATIVE_SOURCE_LEASE.json'
lease=dict(run_uuid=root.name,owner='EXISTING_LOCAL_PC',status='ACTIVE',approval_sha256=scope.APPROVAL_SHA,
 deadline_epoch=scope.DEADLINE,scope=scope.SCOPE,automatic_retry=False,new_GR00T_training_queries_PBS=0)
if v7_policy is not None:lease.update(execution_policy_sha256=scope_ns['V7_POLICY'],finite_job_profile_sha256=sha(raw(P['v7_finite_source_job'])))
put(lease_path,raw(lease))
if scope.V7_CARRY in account:
 carry=account[scope.V7_CARRY];previous=Path(P['predecessor_root'])
 previous_account=previous/'control/NIGHT_SOURCE_ACCOUNT.json'
 for key,name in dict(account='control/NIGHT_SOURCE_ACCOUNT.json',lease='control/NATIVE_SOURCE_LEASE.json',terminal='outputs/SOURCE_CHAIN_TERMINAL.json').items():
  assert(previous/name).read_bytes()==base64.b64decode(carry[key+'_base64'],validate=True)
 q=subprocess.run(['systemctl','--user','show',previous.name+'.service','-p','MainPID','-p','SubState'],capture_output=True,text=True,timeout=4)
 assert q.returncode==0 and dict(x.split('=',1)for x in q.stdout.splitlines()if '='in x)==dict(MainPID='0',SubState='exited')
 for name,pin in P.get('predecessor_preserved_files',{}).items():assert sha((previous/name).read_bytes())==pin
 before=previous_account.read_bytes();closed=scope.transfer_development_custody(before,pc,destination=str(root))
 key='V7-SOURCE-CUSTODY-TRANSFER-'+root.name
 transfer=commit(previous_account,expected_before_sha256=sha(before),proposed_bytes=raw(closed),request_id=key,intent_path=previous_account.with_name(key+'.INTENT.json'),receipt_path=previous_account.with_name(key+'.RECEIPT.json'),lock_path=previous_account.with_name('NIGHT_SOURCE_ACCOUNT.owner.lock'))
 put(root/'PREDECESSOR_CUSTODY_TRANSFER.json',raw(transfer))
put(root/'control/NIGHT_SOURCE_ACCOUNT.json',raw(account))
reg_new=json.loads(rb);reg_new['entries'].append(dict(role=P['role'],path=str(root),run_uuid=root.name,
 executor_node='corelab-4080s',owner='EXISTING_LOCAL_PC',scope=scope.SCOPE,
 authority_sha256=scope.APPROVAL_SHA,source_sha=P['source_sha'],local_lease=str(lease_path)))
def cas(path,before,after,key):
 if scope.V7_CARRY in account:key=key+'-'+root.name
 return commit(path,expected_before_sha256=before,proposed_bytes=raw(after),request_id=key,
  intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),
  lock_path=path.with_name(path.name+'.owner.lock'))
registration=cas(registry,sha(rb),reg_new,'NIGHT30-REGISTER-ONE-SOURCE-OWNER')
snapshot=registry.read_bytes();registered_snapshot(snapshot,expected_sha256=sha(snapshot),run_root=str(root),role=P['role'])
put(root/'control/CANONICAL_REGISTRY_READBACK.json',snapshot);put(root/'control/ORIGINAL_REGISTRATION_RECEIPT.json',raw(registration))
account.update(custody='ONE_NATIVE_CORE_SOURCE_OWNER',native_adopted=True,qualification_sha256=sha(raw(P['qualification'])))
marker=dict(schema='NIGHT_SOURCE30_DISTINCT_ADDITIVE_ALLOWANCE_V1',approval_sha256=scope.APPROVAL_SHA,
 run_uuid=root.name,source_sha=P['source_sha'],account_path=str(root/'control/NIGHT_SOURCE_ACCOUNT.json'),
 initial_native_account_sha256=sha(raw(account)),PC_closed_account_sha256=sha(pc),
 original_GLOBAL_sha256=sha(gb),original_LOCAL_before_sha256=sha(lb),full14_proof_sha256=sha(raw(budget)),
 registry_after_sha256=sha(snapshot),stop_new_epoch=scope.STOP_NEW,deadline_epoch=scope.DEADLINE,
 caps=scope.CAPS,all_original_financial_fields_unchanged=True,old_work_cannot_spend_this_allowance=True)
if v7_policy is not None:marker.update(execution_policy_sha256=scope_ns['V7_POLICY'],finite_job_profile_sha256=sha(raw(P['v7_finite_source_job'])),historical_caps_retained_not_reissued=True)
if 'NIGHT_SOURCE30_BOUNDED_AUTONOMY'in local:marker['closed_predecessor_source_marker']=local['NIGHT_SOURCE30_BOUNDED_AUTONOMY']
new_local=json.loads(lb);new_local['NIGHT_SOURCE30_BOUNDED_AUTONOMY']=marker
adoption=cas(local_path,sha(lb),new_local,'NIGHT30-ADOPT-DISTINCT-SOURCE-ALLOWANCE')
put(root/'SOURCE_SCOPE_ADOPTION.json',raw(dict(marker=marker,LOCAL_once_receipt=adoption,original_registry_once_receipt=registration)))
ac=cas(root/'control/NIGHT_SOURCE_ACCOUNT.json',sha(pc),account,'NIGHT30-ADOPT-ONE-NATIVE-SOURCE-ACCOUNT')
binding=dict(run_root=str(root),run_uuid=root.name,role=P['role'],registry=str(registry),
 runtime_registry_snapshot=str(root/'control/CANONICAL_REGISTRY_READBACK.json'),registry_snapshot_sha256=sha(snapshot),
 lease=str(lease_path),snapshot_is_new_authority=False)
put(root/'OUTPUT_BINDING.private.json',raw(binding))
paths={k:[]for k in('code','entry','doctor','profile','python','libraries','model','assets','data','cwd','tmp','cache','control','lease','journal','heartbeat','logs','output')}
for name in('NATIVE_JOURNAL.json','NATIVE_OWNER_HEARTBEAT.json'):put(root/'control'/name,raw(dict(status='COLD_NOT_STARTED',run_uuid=root.name)))
paths.update(code=[str(capsule)],entry=[str(root/'entry/infrastructure/enter.sh')],doctor=[str(root/'entry/infrastructure/doctor.py')],
 profile=[str(root/'entry/infrastructure/ENVIRONMENTS.json')],python=[P['python']],libraries=P['library_paths'],
 model=[str(root/'inputs'/name)for name in bundle['definitions']],
 assets=[r['path']for r in assets['files'].values()],data=[str(root/'FIXED_TASK_CARDS.json')],cwd=[str(capsule)],
 tmp=[str(root/'tmp')],cache=[str(root/'cache')],control=[str(root/'control')],lease=[str(lease_path)],
 journal=[str(root/'control/NATIVE_JOURNAL.json')],heartbeat=[str(root/'control/NATIVE_OWNER_HEARTBEAT.json')],logs=[str(root/'logs')],output=[str(root/'outputs')])
put(root/'RUNTIME_PATHS.private.json',raw(paths));locality=node_local_paths(paths)
put(root/'ACTUAL_LOCAL_PATHS.json',raw(locality))
sealed_names=['NIGHT_HUMAN_APPROVAL.json','SOURCE_SCOPE_ADOPTION.json','SOURCE_QUALIFICATION.json','FIXED_TASK_CARDS.json',
 'ASSET_BINDINGS.private.json','RUNTIME_PATHS.private.json','ORIGINAL_ACTION_SEMANTICS_ACTUAL.json','OUTPUT_BINDING.private.json',
 'FULL14_PROOF.json']+['inputs/'+name for name in bundle['definitions']]
if extension is not None:sealed_names.append('SOURCE_SUCCESSOR_AUTHORITY.json')
if v7_policy is not None:sealed_names+=['SOURCE_V7_EXECUTION_POLICY.md','SOURCE_V7_FINITE_JOB.json']
manifest=dict(schema='RUN_MANIFEST_V1',run_uuid=root.name,owner='EXISTING_LOCAL_PC',scope=scope.SCOPE,
 source_sha=P['source_sha'],capsule_relative='entry/code/vla',sealed_inputs={n:sha((root/n).read_bytes())for n in sealed_names},
 authority_sha256=scope.APPROVAL_SHA,original_entry_unchanged=True,original_registry_adopted=True,
 local_lease=str(lease_path),deadline_epoch=scope.DEADLINE,task_kind=bundle['task_kind'],
 predeclared_tasks=[x['side']for x in bundle['plan']]if bundle['task_kind']=='BOOK'else bundle['plan'],attempts_this_chain_max=len(bundle['plan']),
 source_port_not_original_robot_benchmark=True,new_GR00T_training_queries_PBS=0)
if P.get('cabinet_port_revision'):manifest['cabinet_port_revision']=P['cabinet_port_revision']
for key in('condition_side','book_route_revision','book_seed_index','gpu0_shared','GPU_uuid','execution_policy_sha256','idle_owned_units'):
 if key in P:manifest[key]=P[key]
put(root/'RUN_MANIFEST.json',raw(manifest))
env={k:v for k,v in os.environ.items()if k in('PATH','HOME','USER','LOGNAME','LANG','LC_ALL',
 'XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS')}
env.update(TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),XDG_CONFIG_HOME=str(root/'cache'),
 CUDA_CACHE_PATH=str(root/'cache'),__GL_SHADER_DISK_CACHE_PATH=str(root/'cache'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
 MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',CUDA_VISIBLE_DEVICES=P.get('GPU_uuid','GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'),
 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
put(root/'PROCESS_ENVIRONMENT.private.json',raw(env))
probe='import os,sys,json,numpy,mujoco,PIL;print(json.dumps(dict(uid=os.getuid(),host=os.uname().nodename,cwd=os.getcwd(),python=sys.executable,origins={m.__name__:m.__file__ for m in[numpy,mujoco,PIL]},model_queries_physics=0)))'
q=subprocess.run(['/bin/bash',str(root/'entry/infrastructure/enter.sh'),'4080-simulation','python','-B','-c',probe],
 env=env,cwd=str(capsule),capture_output=True,text=True,timeout=15)
assert q.returncode==0,q.stderr;entry=json.loads(q.stdout);assert entry['cwd']==str(capsule)and entry['python']==P['python']
put(root/'ACTUAL_ORIGINAL_ENTRY_PROBE.json',raw(entry))
result=dict(status='ACTUAL_NIGHT_SOURCE_SCOPE_REGISTERED_LOCAL_AND_COLD_BOUND_NO_PHYSICS',run_uuid=root.name,
 source_sha=P['source_sha'],manifest_sha256=sha(raw(manifest)),account_sha256=ac['after_sha256'],
 LOCAL_before_sha256=sha(lb),LOCAL_after_sha256=adoption['after_sha256'],registry_after_sha256=sha(snapshot),
 source_members=len(code),binary_asset_bytes=bundle['total_binary_bytes'],task_kind=bundle['task_kind'],predeclared_tasks=bundle['plan'],
 actual_entry=entry,all14_old_shortfalls=budget['original_forecast']['cap_changes_required'],
 independent_readback_required_before_launch=True,new_physics=0,new_model_queries_updates_PBS=0,
 CPU_s=time.process_time()-started+resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
 observed_epoch=time.time())
put(root/'COLD_ADOPTION_RESULT.json',raw(result));print(json.dumps(result))
'''

READBACK=r'''
import hashlib,json,os,resource,sys,time
from pathlib import Path
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
resource.setrlimit(resource.RLIMIT_CPU,(15,15));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
sha=lambda b:hashlib.sha256(b).hexdigest();root=Path(P['root']);result=json.loads((root/'COLD_ADOPTION_RESULT.json').read_bytes())
assert result['manifest_sha256']==P['manifest_sha256']and result['source_sha']==P['source_sha']
mb=(root/'RUN_MANIFEST.json').read_bytes();assert sha(mb)==P['manifest_sha256'];m=json.loads(mb)
code=root/m['capsule_relative'];sys.path[:0]=[str(code),str(code/'src')]
from scripts.pi0294.multitask import night_scope as scope
from scripts.pi0294.multitask.source_stage_bundle import checked_manifest
from scripts.pi0294.multitask.night_book_chain import checked_plan
scope=scope.for_root(root)
from scripts.pi0294.task_path_entry import node_local_paths,registered_snapshot
scope.authority((root/'NIGHT_HUMAN_APPROVAL.json').read_bytes(),now=time.time())
for name,pin in json.loads((root/'SOURCE_SEAL.json').read_bytes())['files'].items():assert sha((code/name).read_bytes())==pin
for name,pin in m['sealed_inputs'].items():assert sha((root/name).read_bytes())==pin
kind,planned=checked_manifest(m);assert [dict(task_kind=k,side=s)for k,s in checked_plan(root,kind)]==planned
assets=json.loads((root/'ASSET_BINDINGS.private.json').read_bytes())
for name,pin in assets['files'].items():
 b=Path(pin['path']).read_bytes();assert len(b)==pin['bytes']and sha(b)==pin['sha256']
binding=json.loads((root/'OUTPUT_BINDING.private.json').read_bytes());registry=Path(binding['registry']).read_bytes()
assert sha(registry)==binding['registry_snapshot_sha256']==result['registry_after_sha256']
assert (root/'control/CANONICAL_REGISTRY_READBACK.json').read_bytes()==registry
registered_snapshot(registry,expected_sha256=sha(registry),run_root=str(root),role=binding['role'])
localb=Path(P['LOCAL_path']).read_bytes();assert sha(localb)==result['LOCAL_after_sha256'];local=json.loads(localb)
adoption=json.loads((root/'SOURCE_SCOPE_ADOPTION.json').read_bytes())
assert local['NIGHT_SOURCE30_BOUNDED_AUTONOMY']==adoption['marker']
assert sha((root/'control/NIGHT_SOURCE_ACCOUNT.json').read_bytes())==result['account_sha256']
account=scope.validate_account(json.loads((root/'control/NIGHT_SOURCE_ACCOUNT.json').read_bytes()))
assert account['native_adopted']is True
expected_attempts=5 if m.get('cabinet_port_revision')=='ROBOCASA_ARX5_PORT05_SLOWER_TURN_REFERENCE'else 4 if m.get('cabinet_port_revision')=='ROBOCASA_ARX5_PORT04_MEASURED_HINGE_ANCHOR'else 3 if m.get('cabinet_port_revision')=='ROBOCASA_ARX5_PORT03_FIXED_CAMERA_HOME'else 2 if m.get('cabinet_port_revision')=='ROBOCASA_ARX5_PORT02_GEOMETRY_REPAIR'else 1 if kind=='ROBOCASA_CABINET'else 0
if scope.V7_CARRY not in account:assert account['charged_upper']['attempts']==expected_attempts
assert account['qualification_sha256']==sha((root/'SOURCE_QUALIFICATION.json').read_bytes())
lease=json.loads(Path(binding['lease']).read_bytes());assert lease['status']=='ACTIVE'and lease['run_uuid']==root.name
paths=node_local_paths(json.loads((root/'RUNTIME_PATHS.private.json').read_bytes()))
assert not(root/'STOP.json').exists()and not(root/'control/NATIVE_LAUNCH_INTENT.json').exists()
print(json.dumps(dict(status='INDEPENDENT_ALL_SOURCE_ASSET_ACCOUNT_REGISTRY_LOCAL_LEASE_AND_ENTRY_BYTES_VERIFIED',
 source_sha=m['source_sha'],manifest_sha256=sha(mb),account_sha256=result['account_sha256'],LOCAL_sha256=sha(localb),
 registry_sha256=sha(registry),asset_files=len(assets['files']),source_members=result['source_members'],
 local_paths=len(paths['paths']),new_physics=0,new_neural_queries_updates_PBS=0,observed_epoch=time.time())))
'''

LAUNCH=r'''
import hashlib,json,os,subprocess,time
from pathlib import Path
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
sha=lambda b:hashlib.sha256(b).hexdigest();raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
available=sorted(os.sched_getaffinity(0));assert len(available)>=4
cpus=available[-3:];os.sched_setaffinity(0,{available[0]})
root=Path(P['root']);mb=(root/'RUN_MANIFEST.json').read_bytes();assert sha(mb)==P['manifest_sha256'];m=json.loads(mb)
assert m['source_sha']==P['source_sha']
import sys
code=root/m['capsule_relative'];sys.path[:0]=[str(code),str(code/'src')]
from scripts.pi0294.multitask.source_stage_bundle import checked_manifest
from scripts.pi0294.multitask.night_book_chain import checked_plan
from scripts.pi0294.multitask.night_scope import for_root
scope=for_root(root);scope.authority((root/'NIGHT_HUMAN_APPROVAL.json').read_bytes(),now=time.time())
kind,planned=checked_manifest(m);assert [dict(task_kind=k,side=s)for k,s in checked_plan(root,kind)]==planned
for key in('gpu0_shared','GPU_uuid','execution_policy_sha256','idle_owned_units'):
 if key in m:assert P.get(key)==m[key]
assert 0<=time.time()-P['independent_readback_epoch']<120
assert sha(Path(P['LOCAL_path']).read_bytes())==P['LOCAL_sha256']
assert sha((root/'control/NIGHT_SOURCE_ACCOUNT.json').read_bytes())==P['account_sha256']
lease=json.loads((root/'control/NATIVE_SOURCE_LEASE.json').read_bytes())
assert lease['status']=='ACTIVE'and lease['run_uuid']==root.name and not(root/'STOP.json').exists()
assert lease['deadline_epoch']==scope.DEADLINE and time.time()+1300<lease['deadline_epoch']and time.time()<scope.STOP_NEW
for unit in P['idle_D25_units']:
 q=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','LoadState'],capture_output=True,text=True,timeout=4)
 assert dict(x.split('=',1)for x in q.stdout.splitlines()if '='in x)==dict(MainPID='0',LoadState='not-found')
q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
assert q.returncode==0
if not P.get('gpu0_shared'):assert not q.stdout.strip()
else:
 assert P['execution_policy_sha256']=='b57b75cacdc3d8611b9f8d13cd5f4f4e15ed0d4dccd9a69b7469bc62f4c9b99a'
 q=subprocess.run(['nvidia-smi','--id=0','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
 assert q.returncode==0
 gpu_uuid,free_mib=q.stdout.strip().split(',');assert gpu_uuid==P['GPU_uuid']and int(free_mib)>=2048
for unit in P.get('idle_owned_units',[]):
 q=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID'],capture_output=True,text=True,timeout=4)
 assert q.returncode==0 and q.stdout.strip()=='MainPID=0'
unit=root.name+'.service';intent=root/'control/NATIVE_LAUNCH_INTENT.json'
fd=os.open(intent,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
with os.fdopen(fd,'wb')as f:f.write(raw(dict(unit=unit,manifest_sha256=sha(mb),source_sha=m['source_sha'],
 independent_readback_sha256=P['independent_readback_sha256'],automatic_retry=False,observed_epoch=time.time())));f.flush();os.fsync(f.fileno())
env=json.loads((root/'PROCESS_ENVIRONMENT.private.json').read_bytes())
assert len(cpus)==3
args=['/usr/bin/systemd-run','--user','--unit='+unit,'--property=RuntimeMaxSec=1250',
 '--property=TimeoutStopSec=15','--property=KillMode=control-group','--property=CPUQuota=300%',
 '--property=CPUAffinity='+' '.join(map(str,cpus)),'--property=RemainAfterExit=yes',
 '--property=WorkingDirectory='+str(root/'entry/code/vla'),
 '--property=StandardOutput=append:'+str(root/'logs/owner.log'),'--property=StandardError=inherit']
for k,v in env.items():args.append('--setenv='+k+'='+v)
args+=['/usr/bin/env','-i']+[k+'='+v for k,v in env.items()]
args+=['/bin/bash',str(root/'entry/infrastructure/enter.sh'),'4080-simulation','python','-B','-m',
 'scripts.pi0294.multitask.'+('night_source_chain'if kind=='ALL_PINNED'else'night_cabinet_chain'if kind=='ROBOCASA_CABINET'else'night_book_chain'),str(root)]
q=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10)
assert q.returncode==0,'NATIVE_START_RESULT_UNKNOWN_NO_REPLAY'
q=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','InvocationID','-p','ControlGroup','-p','ActiveState'],
 env=env,capture_output=True,text=True,timeout=4)
assert q.returncode==0;facts=dict(x.split('=',1)for x in q.stdout.splitlines()if '='in x)
print(json.dumps(dict(status='ONE_FINITE_SOURCE_NATIVE_UNIT_START_ACKNOWLEDGED_REQUIRES_ACTUAL_PROCESS_READBACK',
 unit=unit,native=facts,launch_intent_sha256=sha(intent.read_bytes()),automatic_retry=False,
 task_kind=kind,predeclared_tasks=planned,new_GR00T_training_queries_PBS=0,observed_epoch=time.time())))
'''
