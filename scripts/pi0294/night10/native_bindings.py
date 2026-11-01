"""Bind the supplied DAG to existing sealed native jobs, never replay effects."""
import fcntl,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
P=Path(__file__).parent
CONFIG=json.loads((P/'BINDINGS.json').read_bytes());M=Path(CONFIG['matched_manager_root'])
sys.path[:0]=[str(P/'source'),str(P/'source/src')]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def unit(root):
 q=subprocess.run(['systemctl','--user','show',Path(root).name+'.service','-p','MainPID','-p','SubState','-p','InvocationID'],capture_output=True,text=True,timeout=6)
 v=dict(x.split('=',1)for x in q.stdout.splitlines()if '='in x)
 if q.returncode not in(0,4)or'MainPID'not in v:raise RuntimeError('UNKNOWN_NATIVE_UNIT:'+str(root))
 return v
def job(arm,steps):return M/'jobs'/(arm+('_native10'if steps==10 else'_formal4096'))
def prefix(arm,steps):return 'BC299_'+arm+'_S'+str(steps)+'_'
def control(folder,module,action,*arguments):
 """Exactly one canonical metadata writer, even with two live GPU workers."""
 lock=(P/'NATIVE_CONTROL.owner.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  label=folder.name+'_'+module.removesuffix('.py')+'_'+action
  if arguments:label+='_'+'_'.join(map(str,arguments))
  record=P/'control_steps'/(label+'.json');record.parent.mkdir(exist_ok=True)
  if record.exists():raise RuntimeError('NATIVE_STEP_INTENT_ALREADY_EXISTS_NO_REPLAY:'+label)
  argv=[CONFIG['python'],'-B',str(folder/module),action,*map(str,arguments)]
  write(record,dict(status='INTENT',argv=argv,epoch=time.time()))
  with record.with_suffix('.stdout').open('xb')as out,record.with_suffix('.stderr').open('xb')as err:q=subprocess.run(argv,cwd=folder,stdout=out,stderr=err,timeout=900)
  write(record.with_suffix('.RESULT.json'),dict(returncode=q.returncode,epoch=time.time()))
  if q.returncode:raise RuntimeError(label+':'+record.with_suffix('.stderr').read_text(errors='replace')[-1600:])
 finally:lock.close()
def check_staged(path):
 v=read(path);r=Path(v['root'])
 if sha(r/'RUN_MANIFEST.json')!=v['manifest_sha256']:raise PermissionError('EXACT_EXISTING_RUN_MANIFEST')
 return v,r
def wait(r):
 start=time.time()
 while True:
  u=unit(r)
  if int(u['MainPID'])==0:
   if not(r/'RESULT.json').exists():raise RuntimeError('NATIVE_TERMINAL_WITHOUT_RESULT')
   return read(r/'RESULT.json')
  if time.time()-start>CONFIG['maximum_native_wait_s']:raise TimeoutError('FINITE_PREDECLARED_NATIVE_WAIT')
  time.sleep(20)
def staged_all_closed():
 for f in M.glob('jobs/*/*_STAGED.private.json'):
  _,r=check_staged(f)
  if int(unit(r)['MainPID'])or read(r/'control/LEASE.json')['status']!='CLOSED':return False
 for f in [M/'GPU1/STAGED.json']:
  if f.exists():
   _,r=check_staged(f)
   if int(unit(r)['MainPID'])or read(r/'control/LEASE.json')['status']!='CLOSED':return False
 return True
def safe_failure_close(folder,arm,steps):
 """Settle a proven terminal failed native job, retaining real costs and STOP."""
 sys.path.insert(0,str(folder));import resident_common as b
 staged=read(folder/(prefix(arm,steps)+'STAGED.private.json'));r=Path(staged['root']);manifest=read(r/'RUN_MANIFEST.json')
 if int(unit(r)['MainPID']):raise PermissionError('FAILED_JOB_STILL_LIVE')
 result=read(r/'RESULT.json')
 if result['status']!='FAILED_BOOK_FINITE_OPTIMIZATION':raise PermissionError('EXACT_FAILED_OPTIMIZATION_SCHEMA')
 if read(r/'control/LEASE.json')['status']=='CLOSED':return
 journal=subprocess.run(['journalctl','--user','-u',r.name+'.service','-o','json','--no-pager'],capture_output=True,text=True,check=True,timeout=8)
 rows=list(map(json.loads,journal.stdout.splitlines()));cpu=[int(x['CPU_USAGE_NSEC'])/1e9 for x in rows if'CPU_USAGE_NSEC'in x];begin=[int(x['__REALTIME_TIMESTAMP'])for x in rows if x.get('MESSAGE','').startswith('Started '+r.name+'.service')];end=[int(x['__REALTIME_TIMESTAMP'])for x in rows if'CPU_USAGE_NSEC'in x]
 if len(cpu)!=1 or len(begin)!=1 or len(end)!=1:raise PermissionError('TERMINAL_USAGE_NOT_PROVEN')
 files={n:sha(r/n)for n in('RESULT.json','FIT_PROGRESS.json','STOP.json')if(r/n).exists()};base=read(folder/(prefix(arm,steps)+'PREPARED.private.json'))
 actual=dict(model_loads=result['model_load_attempts'],optimizer_updates=result['optimizer_updates'],TRAIN_window_visits=result['TRAIN_visits'],generation_queries=result['generation_attempts'],completed_generations=result['completed_generations'],physics_integrations=0,formal_DEV_conditions=0,CPU_s=cpu[0],GPU_allocated_s_upper=math.ceil((end[0]-begin[0])/1e6),result_sha256=files['RESULT.json'],error=result['error'],all_costs_retained=True)
 import ast
 tree=ast.parse((folder/'resident_fit_close.py').read_text());code=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(a,ast.Name)and a.id=='CLOSE'for a in n.targets))
 code=code.replace("json.loads(result)['status']=='COMPLETE_BOOK5_FINITE_TARGET_LEARNING'","json.loads(result)['status']=='FAILED_BOOK_FINITE_OPTIMIZATION'")
 staged.update(LOCAL=base['LOCAL'],registry=base['registry'],LOCAL_before_sha256=sha(base['LOCAL']),actual=actual,files=files)
 lock=(P/'NATIVE_CONTROL.owner.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  closed=b.call('NIGHT10_'+arm+'_S'+str(steps)+'_FAILED_CLOSE_ONCE',code,staged,effect='MUTATION',CPU=15,reply=16000)
  write(P/'receipts'/(arm+'_S'+str(steps)+'_FAILED_CLOSED.json'),closed)
 finally:lock.close()
def fit(arm,steps):
 folder=job(arm,steps);pre=prefix(arm,steps)
 if(folder/(pre+'FINAL_INDEPENDENT.private.json')).exists():
  v,r=check_staged(folder/(pre+'STAGED.private.json'))
  if read(r/'control/LEASE.json')['status']!='CLOSED'or int(unit(r)['MainPID']):raise PermissionError('EXACT_OLD_COMPLETION_NOT_LIVE')
  return read(folder/(pre+'COMPLETED_CLOSED.private.json'))
 for action,member in [('prepare','PREPARED'),('stage','STAGED')]:
  if not(folder/(pre+member+'.private.json')).exists():control(folder,'resident_fit.py',action)
 v,r=check_staged(folder/(pre+'STAGED.private.json'))
 if not(r/'LAUNCH_INTENT.json').exists():
  if not(folder/(pre+'INDEPENDENT.private.json')).exists():control(folder,'resident_fit.py','read')
  control(folder,'resident_fit.py','start')
 result=wait(r)
 if result['status']!='COMPLETE_BOOK5_FINITE_TARGET_LEARNING':
  safe_failure_close(folder,arm,steps);raise RuntimeError('ACTUAL_CLOSED_FINITE_MODEL_FAILURE:'+str(result.get('error')))
 for action,member in [('collect','RAW_INDEPENDENT'),('audit','INDEPENDENT_AUDIT'),('close','COMPLETED_CLOSED'),('read','FINAL_INDEPENDENT')]:
  if not(folder/(pre+member+'.private.json')).exists():control(folder,'resident_fit_close.py',action)
 audit=read(folder/(pre+'INDEPENDENT_AUDIT.private.json'))
 if not audit['capacity']['capacity_pass']:raise PermissionError('ACTUAL_TRAINING_CAPACITY_FAILED')
 return read(folder/(pre+'COMPLETED_CLOSED.private.json'))
def training(arm):
 fit(arm,10);closed=fit(arm,4096)
 return dict(status='COMPLETE_TRAINING_CLOSED',arm=arm,optimizer_updates=4096,native_effects_closed=True,closed=closed)
def evaluation(arm):
 folder=job(arm,4096);train=read(folder/(prefix(arm,4096)+'FINAL_INDEPENDENT.private.json'))
 # Existing helpers perform the original manifest/lease/asset and safety checks.
 for module,pref,actions in [('resident_diagnostic.py','BC299_'+arm+'_T32_',('prepare','stage','read','start'))]:
  for action,member in zip(actions,('PREPARED','STAGED','INDEPENDENT','STARTED')):
   if not(folder/(pref+member+'.private.json')).exists():control(folder,module,action)
  _,r=check_staged(folder/(pref+'STAGED.private.json'));result=wait(r)
  if result['status']!='COMPLETE_MATCHED_TRAIN32_PHASE_DIAGNOSTIC':raise RuntimeError('TRAIN32_FAILED_REQUIRES_RECONCILIATION')
  for action,member in [('collect','RAW_INDEPENDENT'),('close','COMPLETED_CLOSED'),('closed_read','FINAL_INDEPENDENT')]:
   if not(folder/(pref+member+'.private.json')).exists():control(folder,module,action)
 pre='BC299_'+arm+'_DEV_'
 for action,member in [('prepare','PREPARED'),('stage','STAGED'),('read','INDEPENDENT'),('start','STARTED')]:
  if not(folder/(pre+member+'.private.json')).exists():control(folder,'resident_dev.py',action,4096)
 _,r=check_staged(folder/(pre+'STAGED.private.json'));result=wait(r)
 if not(folder/(pre+'RAW_HEADER.private.json')).exists():control(folder,'resident_dev_close.py','collect',4096)
 for i in range(len(result['conditions'])):
  if not(folder/(pre+'RAW_CASE'+str(i)+'.private.json')).exists():control(folder,'resident_dev_close.py','case',4096,i)
 for action,member in [('audit','INDEPENDENT_AUDIT'),('close','COMPLETED_CLOSED'),('read','FINAL_INDEPENDENT')]:
  if not(folder/(pre+member+'.private.json')).exists():control(folder,'resident_dev_close.py',action,4096)
 return dict(status='COMPLETE_DEV_CLOSED',arm=arm,native_effects_closed=True,audit=read(folder/(pre+'INDEPENDENT_AUDIT.private.json')),result_sha256=sha(r/'RESULT.json'))
def audit_data():
 src=Path(CONFIG['source_manager_root']);state=read(src/'STATE.json');qualification=read(src/'JOINT_FINAL.RESULT.json');data=read(M/'admitted/ADMITTED_ARMS.json')
 if state['status']!='COMPLETE'or int(unit(src)['MainPID'])or qualification['status']!='PASS_OFFICIAL_TRAIN_CODEC_ALL_WINDOWS_AND_REAL_RGB':raise PermissionError('COMPLETE_ACTUAL_SOURCE_AND_OFFICIAL_QUALIFICATION')
 if sha(src/'expanded_TRAIN/DATASET_MANIFEST.json')!=qualification['manifest_sha256']or sha(src/'expanded_TRAIN/statistics.json')!=qualification['statistics_sha256']:raise PermissionError('FROZEN_DATA_IDENTITY')
 for a in data['arms'].values():
  if a['status']=='ADMITTED_EXACT_OFFICIAL_TRAIN_SUBSET':
   for n,k in [('DATASET_MANIFEST.json','manifest_sha256'),('statistics.json','statistics_sha256'),('TRAIN_POPULATION.json','population_sha256')]:
    if sha(Path(a['root'])/n)!=a[k]:raise PermissionError('FROZEN_ARM_CHANGED')
 return dict(status='COMPLETE_ACTUAL_DATA_AUDIT',native_effects_closed=True,manifest_sha256=qualification['manifest_sha256'],statistics_sha256=qualification['statistics_sha256'],entries=len(qualification['episodes']),windows=qualification['windows'],arms={a:dict(status=v['status'],entries=len(v.get('entries',[])),windows=v.get('windows'),new_qualified_entries=v['new_qualified_entries'])for a,v in data['arms'].items()},source_attempts=18)
def policy():
 from scripts.pi0294.night10.policy_review import review
 card=read(P/'POLICY_REVIEW.json');v=review(card);v['native_effects_closed']=True
 write(P/'receipts/policy_review.json',v);return v
def gpu1():
 folder=P/'GPU1'
 if not(folder/'DECISION.json').exists():control(folder,'gpu1_inference.py','prepare')
 if read(folder/'DECISION.json')['status']=='NOT_ADMITTED_ACTUAL_RESOURCE':return dict(status='SKIPPED_ACTUAL_GPU1_RESOURCE',native_effects_closed=True,decision=read(folder/'DECISION.json'))
 for action,member in [('stage','STAGED'),('read','INDEPENDENT'),('start','STARTED')]:
  if not(folder/(member+'.json')).exists():control(folder,'gpu1_inference.py',action)
 _,r=check_staged(folder/'STAGED.json');wait(r)
 if not(folder/'FINAL_INDEPENDENT.json').exists():control(folder,'gpu1_inference.py','finish')
 return dict(status='COMPLETE_GPU1_MODEL_DIAGNOSTIC_CLOSED',native_effects_closed=True,result=read(folder/'FINAL_INDEPENDENT.json'))
def preflight():
 if int(unit(M)['MainPID']):raise PermissionError('OLD_MANAGER_STILL_ACTIVE')
 if read(Path(CONFIG['source_manager_root'])/'STATE.json')['status']!='COMPLETE':raise PermissionError('SOURCE_NOT_TERMINAL_COMPLETE')
 # Any preceding physical evaluator must really be CLOSED, even on another model.
 for f in M.glob('jobs/*/*_DEV_STAGED.private.json'):
  _,r=check_staged(f)
  if int(unit(r)['MainPID'])or read(r/'control/LEASE.json')['status']!='CLOSED':raise PermissionError('PRIOR_PHYSICS_NOT_COLD')
 print(json.dumps(dict(status='ACTUAL_COLD_OWNER_FOR_NEW_MODEL_DEV',epoch=time.time())))
def main(action,arm=None):
 if action=='preflight':return preflight()
 key=action+('_'+arm if arm else'');receipt=P/'receipts'/(key+'.json')
 if receipt.exists():raise FileExistsError('NIGHT10_TASK_NO_REPLAY:'+key)
 if int(unit(M)['MainPID']):raise PermissionError('TWO_METADATA_MANAGERS_FORBIDDEN')
 try:
  if action=='train':v=training(arm)
  elif action=='eval':v=evaluation(arm)
  elif action=='data':v=audit_data()
  elif action=='gpu1':v=gpu1()
  elif action=='policy':
   v=policy();return
  else:raise ValueError(action)
  write(receipt,v)
 except BaseException as exc:
  write(receipt,dict(status='FAILED',native_effects_closed=staged_all_closed(),error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),epoch=time.time()));raise
if __name__=='__main__':main(*sys.argv[1:])
