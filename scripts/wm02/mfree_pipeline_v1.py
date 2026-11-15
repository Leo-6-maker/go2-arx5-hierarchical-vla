"""This campaign only: baseline8, seal TRAIN, one FM1000, fresh RGB, TRAIN8/DEV16.
Existing Journal/once, registry, native owner and one neural owner are reused.
No retries, second training arm, WBC change, system configuration or new scheduler.
"""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
read=lambda p:json.loads(Path(p).read_bytes())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec_path=Path(sys.argv[1]);s=read(spec_path);root=Path(s['registered_root']);R=Path(s['source_root']);parent=s['trusted_source_parent']
assert digest(__file__)==s['pipeline_program_sha256']
sys.path[:0]=[str(R),str(R/'src'),parent['source_root'],parent['source_root']+'/src']
import go2arx5_vla.world_model,scripts.wm02
go2arx5_vla.world_model.__path__.insert(0,str(R/'src/go2arx5_vla/world_model'));scripts.wm02.__path__.insert(0,str(R/'scripts/wm02'))
from scripts.wm02.mobile_boundary_runner_v1 import registered
from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.journal import Journal
lease=registered(s,spec_path)
if(root/'SUPERVISOR.private.json').exists():raise PermissionError('NO_PIPELINE_REPLAY_OR_UNKNOWN_RELAUNCH')
write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),epoch=time.time(),program_sha256=digest(__file__)))
journal=Journal(root/'control/journal',root.name);children=[];control=Path(s['world_control']);collection=Path(s['collection_root']);H=Path(s['H']);deadline=s['deadline_epoch']
for pin in s['input_pins']:
 if digest(pin['path'])!=pin['sha256']:raise PermissionError('SEALED_PIPELINE_INPUT_CHANGED:'+pin['path'])
for name,pin in s['source_files'].items():
 if digest(R/name)!=pin:raise PermissionError('SEALED_SCIENTIFIC_SOURCE_CHANGED:'+name)
native_seed=read(s['native_template']['path']);training_seed=read(s['training_template']['path']);resident_seed=read(s['resident_template']['path'])
population=read(collection/'POPULATION.private.json');scenes={(v['family'],v['layout_id']):v for v in read(s['scene_catalog']['path'])}
registry_path=Path(s['registry']);native_python=native_seed['python'];sla_python=training_seed['python'];dataset=root/'dataset';rows=[]

def finite():
 if time.time()>=deadline-900:raise TimeoutError('FINITE_PIPELINE_DEADLINE_NO_NEW_EFFECT')
 if(root/'STOP.json').exists():raise RuntimeError('PIPELINE_OPERATOR_STOP_NO_NEW_DISPATCH')

def progress(phase,**detail):write(root/'PROGRESS.private.json',dict(phase=phase,epoch=time.time(),**detail))

def dirs(path):
 path.mkdir()
 for name in ('control','logs','tmp','cache','data'):(path/name).mkdir()

def commit_entries(specs):
 lock=OwnerLock(control/'registration.lock')
 try:
  registry=read(registry_path);entries=[]
  for path,value in specs:
   assert not(path/'EXECUTION_SPEC.json').exists()
   write(path/'EXECUTION_SPEC.json',value)
   write(path/'control/LEASE.json',dict(status='ACTIVE',owner=s['owner'],spec_sha256=digest(path/'EXECUTION_SPEC.json'),deadline_epoch=deadline))
   entries.append(dict(path=str(path),run_uuid=path.name+'-'+root.name,role=value['role'],source_sha=s['source_sha'],owner=s['owner'],scope='FIXED_MFREE_C20_FINITE_AUTHORIZED_PIPELINE',executor_node='corelab-4080s',local_lease=str(path/'control/LEASE.json'),ROS_DOMAIN_ID=value['ROS_DOMAIN_ID']))
  registry['entries'].extend(entries);write(registry_path,registry)
 finally:lock.close()

def launch(path,value,argv,env,tag):
 finite();created=[]
 def effect():
  with(path/'logs'/f'{tag}.log').open('xb')as log:
   child=subprocess.Popen(argv,cwd=R,env=dict(os.environ,**env),stdout=log,stderr=subprocess.STDOUT,
       preexec_fn=lambda:os.sched_setaffinity(0,set(value['cpus'])))
  created.append(child);children.append(child)
  return dict(pid=child.pid,epoch=time.time(),spec_sha256=digest(path/'EXECUTION_SPEC.json'))
 journal.once(tag,dict(spec_sha256=digest(path/'EXECUTION_SPEC.json')),effect)
 if not created:raise PermissionError('PRIOR_EFFECT_REQUIRES_INDEPENDENT_RECONCILIATION')
 return created[0]

def wait(child):
 while child.poll()is None:
  if time.time()>=deadline-120:raise TimeoutError('FINITE_PIPELINE_CLOSEOUT')
  time.sleep(5)
 return child.returncode

def source_fields():return {k:s[k]for k in('source_sha','source_root','source_files','trusted_source_parent','source_seal','registry','owner')}

def environment_native():
 env=dict(native_seed['native_environment']);env.update(ROS_DOMAIN_ID='196',PYTHONPATH=str(R)+':'+str(R/'src')+':'+env['PYTHONPATH'],PYTHONDONTWRITEBYTECODE='1')
 return env

def environment_sla(value):
 runtime=value['SLA_runtime'];rule=read(runtime['initial_function_rule']['path'])
 return dict(CUDA_VISIBLE_DEVICES=s['GPU_uuid'],JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
  XLA_PYTHON_CLIENT_MEM_FRACTION=str(runtime['JAX_memory_fraction']),XLA_FLAGS=rule['XLA_FLAGS'],
  PYTHONPATH=str(R)+':'+str(R/'src')+':'+runtime['upstream_root']+'/src:'+runtime['upstream_root']+'/packages/openpi-client/src',
  PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')

def capacity(kind):
 while True:
  finite();available=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
  # Account for collection worlds between children as well as currently live
  # children. Their future committed capacity cannot be borrowed by a model.
  reserve=0
  for family in ('F1','F2'):
   g=collection/family
   if read(g/'control/LEASE.json')['status']=='CLOSED':continue
   unit='wm02-mfree-c20-'+family.lower()+'-20261115-02'
   raw=subprocess.check_output(['systemctl','--user','show',unit,'--property=MemoryCurrent','--value'],text=True).strip()
   current=int(raw)if raw.isdigit()else 0;reserve+=max(0,s['native_peak_bound_bytes']-current)
  required=s['model_host_peak_bound_bytes']+s['host_margin_bytes']+reserve+(s['native_peak_bound_bytes']if kind=='panel'else s['frame_cache_bytes'])
  csv=subprocess.check_output(['nvidia-smi','--id='+s['GPU_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
  free=int(csv.strip())*1024**2
  if available>=required and free>s['allocator_limit_bytes']+1024**3:
   try:lock=OwnerLock(s['neural_owner_lock'])
   except BlockingIOError:lock=None
   if lock is not None:
    lock.close();write(root/'CAPACITY_LAST_ACTUAL.private.json',dict(epoch=time.time(),kind=kind,host_available=available,required=required,GPU_free_bytes=free,other_user_jobs_untouched=True));return
  progress('RESOURCE_QUEUE_'+kind,host_available=available,required=required,GPU_free_bytes=free)
  time.sleep(30)

def register_panel(tag,conditions,policies):
 path=root/tag;dirs(path);native=path/'native';server=path/'resident';dirs(native);dirs(server);(native/'runs').mkdir();(server/'requests').mkdir()
 base=dict(native_seed,**source_fields());base.update(ROS_DOMAIN_ID=196,cpus=[16,17],deadline_epoch=deadline,wall_limit_s=2700,max_ticks=35703,
  policy_queries=120,mobile_stage='L1',mobile_experiment_version='MFREE_STUDENT_60S_V1',student_goal_offset_m=0.,
  accepted_boundary_version='MOBILE_ACCEPTED_BOUNDARY_050_V1',grasp_goal_raise_m=0.,expert_training_eligible=False,
  automatic_formal_training=False,mobile_group=str(native),mobile_world_lock=str(control/'ros-196.lock'),native_environment=environment_native())
 runtime=dict(resident_seed['SLA_runtime']);runtime['compilation_cache_dir']=str(root/'cache'/tag);Path(runtime['compilation_cache_dir']).mkdir()
 gpu=dict(base,registered_root=str(server),role=(root.name+'-'+tag+'-RESIDENT').upper(),cpus=[12,13],physics=False,
  policies=policies,max_queries=120*len(conditions),neural_owner_lock=s['neural_owner_lock'],python=sla_python,SLA_runtime=runtime,
  GPU_uuid=s['GPU_uuid'],checkpoint=resident_seed['checkpoint'],checkpoint_manifest_path=resident_seed['checkpoint_manifest_path'],
  checkpoint_parameter_bytes=resident_seed['checkpoint_parameter_bytes'],allocator_limit_bytes=s['allocator_limit_bytes'])
 server_pin=dict(path=str(server/'EXECUTION_SPEC.json'),sha256=hashlib.sha256((json.dumps(gpu,indent=2,allow_nan=False)+'\n').encode()).hexdigest())
 specs=[(server,gpu)];cases=[]
 for index,condition in enumerate(conditions):
  family=condition['family'];layout=condition.get('layout_id','T0');scene=scenes[(family,layout)];case=native/'runs'/f'{index:02d}-{family}-{condition["method"]}-r{condition["reset"]}';dirs(case)
  value=dict(base,registered_root=str(case),role=(root.name+'-'+tag+'-'+case.name).upper(),model_path=scene['path'],
   mobile_condition=dict(family=family,level='L1',mode='M_FREE',method=condition['method'],axis='+x'if family=='F1'else'+y',reset=condition['reset'],layout=layout,split=condition.get('split','DEVELOPMENT_BASELINE')),
   resident_server_spec=server_pin,language=('Pick up the red cube and hold it at least two centimeters above the table.'if family=='F1'else'Approach the red cube from the side, pick it up and hold it above the table.'),
   stage_identity_sha256=hashlib.sha256((root.name+'|'+tag+'|'+case.name).encode()).hexdigest())
  value['frozen_files']=dict(base['frozen_files']);value['frozen_files'][scene['path']]=scene['sha256']
  for name in ('logs','tmp','cache','data'):value[name]=str(case/name)
  specs.append((case,value));cases.append(str(case/'EXECUTION_SPEC.json'))
 batch=dict(base,registered_root=str(native),role=(root.name+'-'+tag+'-NATIVE').upper(),cases=cases,
  resident_server_spec=server_pin,maximum_physical_requests=len(cases),unsafe_quarantine_scope='MODEL_BY_FAMILY')
 specs.append((native,batch));commit_entries(specs)
 if digest(server/'EXECUTION_SPEC.json')!=server_pin['sha256']:raise ValueError('SERVER_SPEC_SEAL_BEFORE_PHYSICS')
 return path,native,server,batch,gpu

def panel(tag,conditions,policies,prior_quarantine=()):
 capacity('panel');planned=len(conditions);quarantined=set(prior_quarantine)
 for family in ('F1','F2'):
  path=collection/family/'RESULT.private.json'
  if path.exists()and read(path)['unsafe']:
   quarantined.update(family+':'+r['method']for r in conditions if r['family']==family)
 excluded=[r for r in conditions if r['family']+':'+r['method']in quarantined]
 conditions=[r for r in conditions if r not in excluded]
 if not conditions:
  record=dict(phase=tag,planned_physical_requests=planned,physical_requests=0,native_integrations=0,model_queries=0,
   task_successes=0,grasp_conditions=0,unsafe=0,technical_failure=None,unstarted=planned,
   quarantined_axes=sorted(quarantined),quarantined_unstarted=excluded,reason='ALL_AFFECTED_UNITS_NATIVE_SAFETY_QUARANTINED')
  rows.append(record);write(root/'PHASES_ACTUAL.private.json',rows);return record
 path,native,server,batch,gpu=register_panel(tag,conditions,policies)
 progress('RUNNING_'+tag,planned_physical_requests=len(conditions))
 model=launch(server,gpu,[sla_python,'-B',str(R/'scripts/wm02/mfree_neural_bootstrap_v1.py'),'--server',str(server/'EXECUTION_SPEC.json')],environment_sla(gpu),'LAUNCH-'+tag+'-MODEL')
 physical=launch(native,batch,[native_python,'-B',str(R/'scripts/wm02/mobile_boundary_bootstrap_v1.py'),'--batch',str(native/'EXECUTION_SPEC.json')],environment_native(),'LAUNCH-'+tag+'-NATIVE')
 wait(physical);wait(model)
 result=read(native/'RESULT.private.json');models=read(server/'RESULT.private.json')
 record=dict(phase=tag,native_result=dict(path=str(native/'RESULT.private.json'),sha256=digest(native/'RESULT.private.json')),
  model_result=dict(path=str(server/'RESULT.private.json'),sha256=digest(server/'RESULT.private.json')),
  planned_physical_requests=planned,physical_requests=result['physical_requests'],native_integrations=result['native_integrations'],model_queries=result['new_model_generations'],
  task_successes=sum(bool((o['result'].get('detail')or{}).get('task_success'))for o in result['outcomes']),
  grasp_conditions=sum(bool((o['result'].get('detail')or{}).get('ever_grasp'))for o in result['outcomes']),
  unsafe=sum(o['result']['unsafe']for o in result['outcomes']),technical_failure=result['failure'],unstarted=planned-result['physical_requests'],quarantined_unstarted=excluded,quarantined_axes=sorted(quarantined|set(result['quarantined_axes'])))
 rows.append(record);write(root/'PHASES_ACTUAL.private.json',rows);return record

def export_data():
 path=root/'export';dirs(path);value=dict(s,registered_root=str(path),role=(root.name+'-EXPORT').upper(),dataset=str(dataset),cpus=[12,13])
 commit_entries([(path,value)]);env=dict(environment_sla(dict(SLA_runtime=resident_seed['SLA_runtime'])),CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
 child=launch(path,value,[sla_python,'-B',str(R/'scripts/wm02/mfree_neural_bootstrap_v1.py'),'--export',str(path/'EXECUTION_SPEC.json')],env,'EXPORT-SEALED-SUCCESSFUL-TRAIN')
 code=wait(child);write(path/'control/LEASE.json',dict(read(path/'control/LEASE.json'),status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
 if code or not(dataset/'AUDIT.private.json').exists():raise RuntimeError('DATA_EXPORT_FAILED_PRESERVED_NO_AUTOMATIC_REPLAY')
 return read(dataset/'AUDIT.private.json')

def train_and_restore(audit):
 capacity('train');path=root/'candidate1000';dirs(path);cache=root/'cache/train';cache.mkdir()
 value=dict(training_seed,**source_fields());value.update(registered_root=str(path),role=(root.name+'-FM1000').upper(),
  operation='MFREE_C20_FINAL_FM1000',scope='MFREE_NEW_LIVE_SUCCESSFUL_TRAIN_FM1000',physics=False,cpus=[12,13],ROS_DOMAIN_ID=196,
  deadline_epoch=deadline-600,max_walltime_s=int(s['max_walltime_s']),max_gpu_hours=s['max_walltime_s']/3600.,
  dataset=str(dataset),dense_manifest_sha256=digest(dataset/'MANIFEST.private.json'),target_audit=dict(path=str(dataset/'AUDIT.private.json'),sha256=digest(dataset/'AUDIT.private.json')),
  planned_candidate_updates=1000,planned_target_visits=32000,planned_source_visits=0,microbatch=8,training_matmul_precision='highest',
  parent_host_cast_v7=True,resume=None,continuation=None,frame_cache_bytes=s['frame_cache_bytes'],
  compilation_cache_dir=str(cache),GPU_uuid=s['GPU_uuid'],neural_owner_lock=s['neural_owner_lock'],
  JAX_memory_fraction=.45,allocator_limit_bytes=s['allocator_limit_bytes'],closeout_reserve_s=900,
  native_execution_admitted=False,save_and_exit_on_limit=True,model_service_timeout_s=600,RPC_timeout_s=600,
  action_representation_audit=None)
 runtime=dict(resident_seed['SLA_runtime']);runtime['compilation_cache_dir']=str(cache);value['SLA_runtime']=runtime
 binding=read(training_seed['site_binding']['path']);binding.update(allocation_deadline_epoch=deadline+120,
  host_headroom_bytes=s['model_host_peak_bound_bytes']+s['host_margin_bytes']+s['frame_cache_bytes'],
  paths={k:str(Path(value[k]).resolve())for k in ('parent_params','dataset','compilation_cache_dir')},
  allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=str(root),neural_owner_lock=s['neural_owner_lock']))
 write(path/'SITE_BINDING.private.json',binding);value['site_binding']=dict(path=str(path/'SITE_BINDING.private.json'),sha256=digest(path/'SITE_BINDING.private.json'))
 commit_entries([(path,value)]);progress('RUNNING_FIXED1000',dataset_id=read(dataset/'MANIFEST.private.json')['dataset_id'],optimizer_updates_target=1000)
 with_lock=OwnerLock(s['neural_owner_lock'])
 try:
  child=launch(path,value,[sla_python,'-B',str(R/'scripts/wm02/mfree_neural_bootstrap_v1.py'),'--train',str(path/'EXECUTION_SPEC.json')],environment_sla(value),'TRAIN-ONE-FM1000')
  code=wait(child)
 finally:with_lock.close()
 result=read(path/'RESULT.private.json');write(path/'control/LEASE.json',dict(read(path/'control/LEASE.json'),status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
 if code or result.get('step')!=1000 or result.get('target_visits')!=32000:raise RuntimeError('MOBILE_FIXED1000_NOT_COMPLETE_NO_RETRAIN:'+str(result.get('status')))
 # Fresh ordinary RGB process loads no Adam. It reads only sealed frames and
 # the exported model/statistics, while all training arrays have been released.
 fresh=root/'freshRGB';dirs(fresh)
 replay=dict(value,registered_root=str(fresh),role=(root.name+'-FRESH-RGB').upper(),training_root=str(path),training_result=dict(path=str(path/'RESULT.private.json'),sha256=digest(path/'RESULT.private.json')))
 commit_entries([(fresh,replay)]);replay_path=fresh/'EXECUTION_SPEC.json'
 capacity('train');lock=OwnerLock(s['neural_owner_lock'])
 try:
  child=launch(fresh,replay,[sla_python,'-B',str(R/'scripts/wm02/mfree_neural_bootstrap_v1.py'),'--replay',str(replay_path)],environment_sla(replay),'FRESH-ORDINARY-RGB-RESTORE')
  code=wait(child)
 finally:lock.close()
 write(fresh/'control/LEASE.json',dict(read(fresh/'control/LEASE.json'),status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
 if code or not read(fresh/'REPLAY_RESULT.private.json')['all_bit_exact']:raise RuntimeError('FRESH_ORDINARY_RGB_RESTORE_FAILED_NO_ENDPOINT_REPLACEMENT')
 rows.append(dict(phase='FM1000',training_result=dict(path=str(path/'RESULT.private.json'),sha256=digest(path/'RESULT.private.json')),
  step=result['step'],target_visits=result['target_visits'],source_visits=0,physical_requests=0,
  replay=dict(path=str(fresh/'REPLAY_RESULT.private.json'),sha256=digest(fresh/'REPLAY_RESULT.private.json'))))
 write(root/'PHASES_ACTUAL.private.json',rows);return result

failure=None;code=1
try:
 # Model and collection can share when actual host/GPU capacity supports all
 # committed worlds. Ordinary old-policy failures never gate the new learner.
 baseline=[dict(r,method=r['model'])for r in population['baseline8']]
 panel('baseline8',baseline,resident_seed['policies'])
 while any(read(collection/f/'control/LEASE.json')['status']!='CLOSED'for f in ('F1','F2')):
  finite();progress('WAIT_SEALED_COLLECTION');time.sleep(30)
 audit=export_data();trained=train_and_restore(audit)
 policy={'MFREE_C20_FINAL_FM1000':dict(package=trained['package'])}
 train8=[dict(r,method='MFREE_C20_FINAL_FM1000',split='TRAIN_DIAGNOSTIC')for r in population['train8']]
 dev16=[dict(r,method='MFREE_C20_FINAL_FM1000')for r in population['dev16']]
 train_result=panel('TRAIN8',train8,policy);panel('DEV16',dev16,policy,prior_quarantine=train_result['quarantined_axes']);code=0
except BaseException:failure=traceback.format_exc()
finally:
 # Do not silently leave a neural child orphaned on an orchestration error.
 # Let an already-running finite protected job close under its original quota.
 for child in children:
  if child.poll()is None:
   try:wait(child)
   except BaseException:pass
 result=dict(schema='MFREE_C20_FINITE_PIPELINE_ACTUAL_V1',exit_code=code,failure=failure,phases=rows,
  optional_successor_started=False,automatic_retry=False,WBC_changes=False,safety_changes=False,
  real_hardware=False,RL=False,old_RP8_passed=False,old_JOIN10_retried=False)
 write(root/'RESULT.private.json',result);write(root/'WORKER_EXIT.private.json',dict(exit_code=code))
 write(root/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
raise SystemExit(code)
