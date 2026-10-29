"""Cold S1 native capsule adoption through the existing registry and onceIO.

The finite program stages already-published source, probes the original entry
and reserves preparation costs. It cannot launch a native simulator or worker.
"""

PROGRAM = r'''
import ast,base64,hashlib,io,json,os,resource,subprocess,sys,tarfile,time
from pathlib import Path
from datetime import datetime
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
resource.setrlimit(resource.RLIMIT_CPU,(180,180));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
began=time.process_time();sha=lambda b:hashlib.sha256(b).hexdigest()
raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
root=Path(P['root']);registry=Path(P['registry']);before_registry=registry.read_bytes()
assert sha(before_registry)==P['registry_before_sha256']
approval=(root/'S1_HUMAN_APPROVAL.json').read_bytes()
assert sha(approval)=='8a55153f728c0431116e2f9932f5286f6a0647d362252991712541638b27484f'
a=json.loads(approval);deadline=datetime.fromisoformat(a['window_end_UTC']).timestamp()
assert time.time()<deadline and json.loads((root/'control/ENTRY_QUALIFICATION_LEASE.json').read_bytes())['status']=='CLOSED'
assert sha((root/'control/ENTRY_QUALIFICATION_LEASE.json').read_bytes())==P['closed_entry_lease_sha256']
assert not (root/'native-entry').exists() and not (root/'NATIVE_RUN_MANIFEST.json').exists()
forecast=P['forecast'];observation=P['observation']
assert forecast['shortfalls']=={} and forecast['requested']['engine_starts']==96
assert forecast['requested']['episode_starts']==4 and forecast['requested']['physics_ticks']==62012
assert forecast['total_new_Core_and_PC_CPU_s']==14400 and not forecast['S1_allowance_spendable_by_old_work']
assert all(0<=time.time()-observation[k]['observed_epoch']<=120 for k in ('GLOBAL','LOCAL'))
assert forecast['original_forecast']['GLOBAL_sha256']==observation['GLOBAL']['sha256']
assert forecast['original_forecast']['historical_LOCAL_sha256']==observation['LOCAL']['sha256']
source=base64.b64decode(P['source_tar_base64'],validate=True)
assert sha(source)==P['source_tar_sha256'] and len(source)<100*1024**2
archive=tarfile.open(fileobj=io.BytesIO(source),mode='r:');members={}
for member in archive:
 path=Path(member.name)
 assert not path.is_absolute() and '..'not in path.parts and path.parts[0]in('scripts','src')
 assert member.isdir()or member.isfile()
 if member.isfile():
  b=archive.extractfile(member).read();assert len(b)<8*1024**2
  assert member.name not in members;members[member.name]=b
assert set(members)==set(P['source_members']) and all(sha(b)==P['source_members'][n]for n,b in members.items())
entry_ns={};exec(compile(members['scripts/pi0294/task_path_entry.py'],'<published-S1-entry>','exec'),entry_ns)
entry_ns['authority'](approval,now_epoch=time.time())
once_ns={};exec(compile(members['scripts/pi0294/dec1/source_cpu_once_io.py'],'<original-onceIO>','exec'),once_ns)
commit=once_ns['commit']
def put(path,value,mode=0o400):
 p=Path(path);b=value if isinstance(value,bytes)else raw(value)
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
 with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 assert p.read_bytes()==b
account_path=root/'control/S1_PREPARATION_ACCOUNT.json';old_account=account_path.read_bytes()
assert sha(old_account)==P['account_before_sha256']
account=json.loads(old_account);assert not account.get('active_native_ticket') and account['starts']==account['integrations']==account['RGB_pairs']==0
account['Core_CPU_upper_charged_s']+=P['Core_CPU_upper_s'];account['PC_CPU_upper_charged_s']+=P['PC_CPU_upper_s']
account['transfer_upper_charged_bytes']+=P['transfer_upper_bytes'];account['Core_staging_upper_charged_bytes']+=P['staging_upper_bytes']
assert account['Core_CPU_upper_charged_s']+4*3300<=13800 and account['PC_CPU_upper_charged_s']<=600
assert account['Core_CPU_upper_charged_s']+account['PC_CPU_upper_charged_s']+4*3300<=14400
assert account['transfer_upper_charged_bytes']<=8*1024**3 and account['Core_staging_upper_charged_bytes']<=8*1024**3
key='S1-NATIVE-COLD-PREPARATION-20261029'
commit(account_path,expected_before_sha256=sha(old_account),proposed_bytes=raw(account),request_id=key,
 intent_path=account_path.with_name(key+'.INTENT.json'),receipt_path=account_path.with_name(key+'.RECEIPT.json'),lock_path=account_path.with_name('S1_SCOPE_ACCOUNT.owner.lock'))
code=root/'native-entry/code/vla';infra=root/'native-entry/infrastructure'
code.mkdir(parents=True,mode=0o700);infra.mkdir(mode=0o700)
for name,b in members.items():
 p=code/name;p.parent.mkdir(parents=True,mode=0o700,exist_ok=True);put(p,b)
for name,pin in entry_ns['ENTRY_FILES'].items():
 b=(root/'entry/infrastructure'/name).read_bytes();assert sha(b)==pin
 put(infra/name,b,0o500 if name=='enter.sh'else 0o400)
for name in ('native-s1','native-s1/episodes','cache/cuda','cache/nvidia','cache/torch','cache/torch-extensions','cache/hf','cache/matplotlib','cache/config'):
 (root/name).mkdir(parents=True,mode=0o700,exist_ok=False)
sys.path[:0]=[str(code),str(code/'src')]
from scripts.pi0294.task_path_planning import derive_private_urdf
from scripts.pi0294.task_path_runtime import cards
urdf,srdf,meshes=derive_private_urdf(P['inputs']['planner_source_urdf'],root/'native-s1/planning-assets')
inputs=P['inputs'];inputs['planner_urdf']=str(urdf);inputs['planner_srdf']=str(srdf)
inputs['path_roles']['code']=[str(code)];inputs['path_roles']['entry']=[str(infra/'enter.sh')]
inputs['path_roles']['doctor']=[str(infra/'doctor.py')];inputs['path_roles']['profile']=[str(infra/'ENVIRONMENTS.json')]
inputs['path_roles']['cwd']=[str(code),inputs['native_root']]
inputs['path_roles']['data'] += [str(urdf),str(srdf)]
inputs['path_roles']['lease']=[str(root/'control/NATIVE_S1_LEASE.json')]
lease=dict(run_uuid=root.name,status='ACTIVE',owner='EXISTING_LOCAL_PC',source_sha=P['source_sha'],deadline_epoch=deadline,
 purpose='ONLY_DIRECTLY_APPROVED_FOUR_S1_ATTEMPTS',old_qualification_lease_reopened=False)
put(root/'control/NATIVE_S1_LEASE.json',lease)
put(root/'control/NATIVE_OWNER_HEARTBEAT.json',dict(scope='S1_NATIVE_LOCAL_OWNER',status='PREPARED_NOT_RUNNING',deadline_epoch=deadline))
put(root/'control/NATIVE_JOURNAL.json',dict(status='NO_NATIVE_LAUNCH',source_sha=P['source_sha']))
for name,pin in inputs['frozen_files'].items():assert sha(Path(name).read_bytes())==pin
entry_ns['node_local_paths'](inputs['path_roles'])
put(root/'NATIVE_INPUTS.private.json',inputs)
seal=dict(source_sha=P['source_sha'],files=P['source_members'],complete_native_runtime=True)
put(root/'NATIVE_SOURCE_SEAL.json',seal)
manifest=dict(schema='APPROVED_S1_NATIVE_TASK_MANIFEST_V1',run_uuid=root.name,owner='EXISTING_LOCAL_PC',
 teacher_version='S1_NATIVE_STAGE_PATH_EEF10_V1',approval_sha256=sha(approval),deadline_epoch=deadline,
 code=dict(vla_commit=P['source_sha'],capsule_relative='native-entry/code/vla'),entry_relative='native-entry/infrastructure',
 inputs_sha256=sha((root/'NATIVE_INPUTS.private.json').read_bytes()),source_seal_sha256=sha(raw(seal)))
put(root/'NATIVE_RUN_MANIFEST.json',manifest)
put(root/'NATIVE_FULL14_FORECAST.json',forecast)
put(root/'native-s1/FIXED_SERIES_CARDS.json',dict(cards=cards(),source_sha=P['source_sha'],failed_seed_replacement=False))
safe_parent_keys=('PATH','USER','LOGNAME','HOME','LANG','LC_ALL','XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS')
env={k:os.environ[k]for k in safe_parent_keys if k in os.environ}
assert set(inputs['runtime_environment'])<=set(('PYTHONPATH','LD_LIBRARY_PATH','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH',
 'ROS_DISTRO','ROS_VERSION','RMW_IMPLEMENTATION'))
env.update(inputs['runtime_environment'])
env.update(PYTHONPATH=str(code)+':'+str(code/'src')+':'+env.get('PYTHONPATH',''),PYTHONDONTWRITEBYTECODE='1',
 PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
 TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),XDG_CONFIG_HOME=str(root/'cache/config'),
 CUDA_CACHE_PATH=str(root/'cache/cuda'),__GL_SHADER_DISK_CACHE_PATH=str(root/'cache/nvidia'),
 TORCH_HOME=str(root/'cache/torch'),TORCH_EXTENSIONS_DIR=str(root/'cache/torch-extensions'),HF_HOME=str(root/'cache/hf'),
 MPLCONFIGDIR=str(root/'cache/matplotlib'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
env.pop('LD_PRELOAD',None);env.pop('LD_AUDIT',None)
if not env.get('RMW_IMPLEMENTATION'):env.pop('RMW_IMPLEMENTATION',None)
probe='import os,sys,json,importlib;mods='+repr(['numpy','mujoco','rclpy','PIL','rosidl_runtime_py','ocs2_msgs','std_msgs','nav_msgs','sensor_msgs','scripts.pi0294.task_path_worker'])+';print(json.dumps(dict(cwd=os.getcwd(),python=sys.executable,path=sys.path,origins={n:importlib.import_module(n).__file__ for n in mods},physics=0,models_constructed=0)))'
q=subprocess.run(['/bin/bash',str(infra/'enter.sh'),'4080-simulation','python','-B','-c',probe],env=env,
 cwd=str(root),capture_output=True,text=True,timeout=20)
assert q.returncode==0,q.stderr
probe_value=json.loads(q.stdout);assert probe_value['cwd']==str(code)and probe_value['python']==inputs['runtime_python']
assert probe_value['origins']['scripts.pi0294.task_path_worker']==str(code/'scripts/pi0294/task_path_worker.py')
put(root/'NATIVE_ENTRY_AND_IMPORT_PROBE.json',probe_value)
value=json.loads(before_registry);rows=[r for r in value['entries']if r.get('role')==P['role']]
assert len(rows)==1 and rows[0]['path']==str(root)
put(root/'NATIVE_PREVIOUS_REGISTRY_ROW.json',rows[0])
rows[0].update(scope='APPROVED_S1_NATIVE_TASK',source_sha=P['source_sha'],local_lease=str(root/'control/NATIVE_S1_LEASE.json'))
key='S1-NATIVE-ROLE-ADOPTION-20261029'
receipt=commit(registry,expected_before_sha256=sha(before_registry),proposed_bytes=raw(value),request_id=key,
 intent_path=registry.with_name(key+'.INTENT.json'),receipt_path=registry.with_name(key+'.RECEIPT.json'),lock_path=registry.with_name('PATH_REGISTRY.owner.lock'))
put(root/'control/NATIVE_REGISTRY_READBACK.json',registry.read_bytes())
put(root/'NATIVE_OUTPUT_BINDING.private.json',dict(role=P['role'],snapshot=str(root/'control/NATIVE_REGISTRY_READBACK.json'),
 snapshot_sha256=receipt['after_sha256'],source_sha=P['source_sha'],original_canonical_registry=True))
adoption=dict(status='FULL14_AND_NODE_LOCAL_CPU_READY_FOR_FIRST_RESERVED_NATIVE_START',source_sha=P['source_sha'],
 approval_sha256=sha(approval),forecast_sha256=sha(raw(forecast)),entry_probe_sha256=sha(raw(probe_value)),
 first_native_start_not_performed=True,GR00T_queries=0,training_updates=0,PBS=0,physics_integrations=0)
put(root/'NATIVE_SCOPE_ADOPTION.json',adoption)
put(root/'NATIVE_PROCESS_ENVIRONMENT.private.json',env)
prior=account_path.read_bytes();current=json.loads(prior)
assert not current.get('active_native_ticket') and current['starts']==0
current.update(native_attempts_admitted=True,native_scope_source_sha=P['source_sha'],
 current_native_dynamic_qualification=False,native_scope_adoption_sha256=sha(raw(adoption)))
key='S1-NATIVE-SCOPE-ADOPT-20261029'
commit(account_path,expected_before_sha256=sha(prior),proposed_bytes=raw(current),request_id=key,
 intent_path=account_path.with_name(key+'.INTENT.json'),receipt_path=account_path.with_name(key+'.RECEIPT.json'),lock_path=account_path.with_name('S1_SCOPE_ACCOUNT.owner.lock'))
children=resource.getrusage(resource.RUSAGE_CHILDREN)
result=dict(status='S1_COMPLETE_NATIVE_CAPSULE_AND_ORIGINAL_ENTRY_PREPARED_NOT_STARTED',source_sha=P['source_sha'],
 registry_before_sha256=sha(before_registry),registry_after_sha256=receipt['after_sha256'],
 manifest_sha256=sha(raw(manifest)),inputs_sha256=manifest['inputs_sha256'],scope_adoption_sha256=sha(raw(adoption)),
 files=len(members),CPU_s=time.process_time()-began+children.ru_utime+children.ru_stime,
 Core_CPU_upper_charged_s=P['Core_CPU_upper_s'],PC_CPU_upper_charged_s=P['PC_CPU_upper_s'],
 physics_integrations=0,GR00T_queries=0,training_updates=0,PBS=0,full_native_dynamic_qualification=False)
put(root/'NATIVE_COLD_PREPARATION_RESULT.json',result)
print(json.dumps(result))
'''
