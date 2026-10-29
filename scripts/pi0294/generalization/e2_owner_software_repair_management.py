"""One exact cold software repair of the proved unsubmitted resident owner.

No action on import. The original finite launcher, Journal, onceIO, prepaid
control and native admission are retained. No model, qsub or GLOBAL write.
"""

PROGRAM = r'''
import base64,hashlib,json,math,os,resource,sys,time,traceback
from copy import deepcopy
from pathlib import Path
sys.dont_write_bytecode=True;P=json.loads(sys.stdin.read());stage=Path(P['stage']);began=time.process_time()
resource.setrlimit(resource.RLIMIT_CPU,(178,179));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();sha=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def put(p,b):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
try:
 assert os.getuid()==22182 and not os.environ.get('PBS_JOBID')and os.environ['CUDA_VISIBLE_DEVICES']==''
 os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
 root=Path(P['root']);oldbase=root/'control/resident-owner';eb=(root/'control/RESIDENT_OWNER_OS_EXIT.json').read_bytes();fb=(oldbase/'PRESERVED_STOP.private.json').read_bytes()
 assert sha(eb)=='53d90d76e6ca3a78e2ab8402c2782bc60a441b2eede8033f11cfcb07e11a9afa'
 assert sha(fb)=='54bb49f87d7fa5de9bfb67a251eab44b144ddd77b69ef7846bf8ca721eb45d43'
 failure=json.loads(fb);assert json.loads(eb)['returncode']==1 and failure['error_type']=='NameError'and failure['message']=="name 'r' is not defined"
 for name,pk,sk in [('control/RESIDENT_WRAPPER_PROCESS.private.json','wrapper_pid','wrapper_proc_start_ticks'),('control/resident-owner/SERVER_OWNER_PROCESS.private.json','pid','proc_start_ticks')]:
  row=read(root/name);p=Path('/proc',str(row[pk]),'stat');f=p.read_text().rsplit(')',1)[1].split()if p.exists()else None
  assert not(f and f[0]not in('Z','X')and int(f[19])==int(row[sk]))
 for name in ['SOURCE_REQUEST_INTENT','SOURCE_GLOBAL_RESERVED','SOURCE_QSUB_INTENT','SOURCE_QSUB_REPLY','SOURCE_JOB_BOUND']:
  assert not(oldbase/(name+'.private.json')).exists()
 assert not(root/'A40_NATIVE_EFFECT_LEDGER.json').exists()
 phases=[]
 for p in(oldbase/'setup-rpc').glob('*-INTENT.json'):
  v=read(p);intent=v['intent'];assert intent['effect']=='READ_ONLY'and intent['phase']in('AUTH','QUOTA','FRESH_EFFECTIVE_QUEUE')
  receipt=read(p.with_name(p.name.replace('-INTENT.json','-CONFIRMED.json')));assert receipt['intent']==intent;phases.append(intent['phase'])
 assert sorted(phases)==['AUTH','FRESH_EFFECTIVE_QUEUE','QUOTA']
 saved=read(oldbase/'SOURCE_REQUEST.private.json');old_request=saved['request'];assert old_request['request_id']==P['request_id']and old_request['source_sha']==P['old_source']
 prepared=root/'pbs/alloc-42';assert sorted(p.name for p in prepared.iterdir())==['EFFECTIVE_QUEUE_LIMITS.json','REQUEST.json']
 assert read(prepared/'REQUEST.json')==old_request
 gp=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';gb=gp.read_bytes();g=json.loads(gb)
 assert sha(gb)==P['GLOBAL_sha256']and g['pending_updates']==g['unknown_effects']==[]and all(r['status']=='NATIVE_TERMINAL'for r in g['PBS_submissions'])
 assert not any(r.get('request_id')==P['request_id']for r in g['PBS_submissions'])
 mb=(root/'RUN_MANIFEST.json').read_bytes();manifest=json.loads(mb);bb=(root/'OUTPUT_BINDING.private.json').read_bytes();binding=json.loads(bb)
 cfgb=(root/'MIX_OWNER_CONFIG.private.json').read_bytes();cfg=json.loads(cfgb);sb=(root/'SOURCE_SEAL.json').read_bytes();seal=json.loads(sb)
 leasepath=Path(binding['lease']);lb=leasepath.read_bytes();lease=json.loads(lb)
 assert sha(mb)==binding['run_manifest_sha256']==lease['manifest_sha256']==old_request['run_manifest_sha256']
 assert lease['status']=='ACTIVE'and manifest['code']['vla_commit']==cfg['source_sha']==seal['source_sha']==P['old_source']
 assert sha(sb)==manifest['code']['source_seal_sha256'];oldcode=root/manifest['code']['capsule_relative']
 oldaudit=(root/'control/COLD_INPUT_AUDIT.private.json').read_bytes();assert sha(oldaudit)==old_request['cold_input_audit_sha256']
 for name,pin in manifest['execution']['sealed_inputs'].items():assert sha((root/name).read_bytes())==pin
 frozen=manifest['code']['frozen_numerical_files'];files=dict(seal['files']);code=root/'capsules'/P['source']/'code';assert not code.exists()
 for name,pin in files.items():
  f=oldcode/name;body=f.read_bytes();assert not f.is_symlink()and f.stat().st_mode&0o777==0o400 and sha(body)==pin
  if name in P['delta']:
   row=P['delta'][name];body=base64.b64decode(row['base64']);assert sha(body)==row['sha256'];files[name]=row['sha256']
  assert name not in frozen or sha(body)==frozen[name];put(code/name,body)
 assert set(P['delta'])<=set(files)
 newseal=dict(seal,source_sha=P['source'],published_source_commit=P['source'],files=files,
  previous_unsubmitted_source=P['old_source'],previous_source_seal_sha256=sha(sb))
 sys.path[:0]=[str(code),str(code/'src')]
 from scripts.pi0294.dec1.source_cpu_once_io import commit
 def once(path,before,after,key):return commit(path,expected_before_sha256=sha(before),proposed_bytes=after,request_id=key,
  intent_path=stage/(key+'.INTENT.json'),receipt_path=stage/(key+'.RECEIPT.json'),lock_path=root/'control/COLD_SOURCE_REPAIR.owner.lock')
 history=root/'control/closed-nameerror-owner-01';assert not history.exists();history.mkdir(mode=0o700)
 for name,body in [('RUN_MANIFEST.private.json',mb),('SOURCE_SEAL.private.json',sb),('OUTPUT_BINDING.private.json',bb),('CONFIG.private.json',cfgb),('LEASE.private.json',lb)]:put(history/name,body)
 start=read(root/'control/E2_PRE_NATIVE30_CREDENTIAL_OWNER_LAUNCH_INTENT.json')['epoch'];wall=json.loads(eb)['finished_epoch']-start
 assert 0<wall and math.ceil(wall*4)<=400
 for name in ['resident-owner','resident-owner.job','resident-owner.stdout','resident-owner.stderr','RESIDENT_WRAPPER_PROCESS.private.json','RESIDENT_OWNER_OS_EXIT.json','COLD_INPUT_AUDIT.private.json','COLD_FULL_FUTURE_PREVIEW.private.json','COLD_FIXED_REQUEST_PREVIEW.private.json']:
  p=root/'control'/name;s=p.lstat();os.rename(p,history/name);assert(history/name).lstat().st_ino==s.st_ino
 s=prepared.lstat();os.rename(prepared,history/'prepared-alloc-42');assert(history/'prepared-alloc-42').lstat().st_ino==s.st_ino
 cfg['source_sha']=P['source'];cfg['request_template'].update(source_sha=P['source'],pre_native_owner_CPU_upper_s=2600)
 manifest['code'].update(vla_commit=P['source'],capsule_relative='capsules/'+P['source']+'/code',source_seal_sha256=sha(raw(newseal)))
 manifest['execution']['sealed_inputs'].update({'SOURCE_SEAL.json':sha(raw(newseal)),'MIX_OWNER_CONFIG.private.json':sha(raw(cfg))})
 binding['run_manifest_sha256']=sha(raw(manifest));lease['manifest_sha256']=binding['run_manifest_sha256']
 for name,before,after,key in [('SOURCE_SEAL.json',sb,raw(newseal),'E2-NAMEERROR30-SEAL'),('MIX_OWNER_CONFIG.private.json',cfgb,raw(cfg),'E2-NAMEERROR30-CONFIG'),('RUN_MANIFEST.json',mb,raw(manifest),'E2-NAMEERROR30-MANIFEST'),('OUTPUT_BINDING.private.json',bb,raw(binding),'E2-NAMEERROR30-BINDING')]:once(root/name,before,after,key)
 once(leasepath,lb,raw(lease),'E2-NAMEERROR30-OWN-LEASE')
 # onceIO and every qualification module use the new complete sealed capsule;
 # the preserved old source remains read-only provenance.
 sys.path[:0]=[str(code),str(code/'src')]
 from scripts.pi0294.generalization import minimal_e2only_execution as x,minimal_e2only_physical as physical,minimal_e2only_pre_native as cold,minimal_native_launch as launch
 from scripts.pi0294.pv1.model_inputs import preflight
 preflight(root,read(root/'NEURAL_BINDINGS.private.json'))
 for name,pin in files.items():assert sha((code/name).read_bytes())==pin
 for name,pin in manifest['execution']['sealed_inputs'].items():assert sha((root/name).read_bytes())==pin
 audit=dict(json.loads(oldaudit),source_sha=P['source'],run_manifest_sha256=sha(raw(manifest)),source_files=len(files),
  original_closed_reader_audit_sha256=sha(oldaudit),original_input_members_unchanged=True,observed_epoch=time.time(),CPU_actual_s=time.process_time()-began)
 effective=launch.collect_effective_queue(walltime_s=22440);request=dict(cfg['request_template'],run_manifest_sha256=sha(raw(manifest)),cold_input_audit_sha256=sha(raw(audit)),
  submit_count=len(g['PBS_submissions'])+1,created_epoch=time.time(),effective_queue_proof_sha256=sha(raw(effective)))
 x.request(request);guard=physical.Guard(root,request,manifest,lambda **_:None);guard.inspect(next_output_upper_bytes=physical.CONTROL_GROWTH_RESERVE)
 proof=dict(GLOBAL_sha256=sha(gb),request_id=P['request_id'],all_recorded_actors_terminal=True,native_requests=0,
  only_readonly_setup_phases=True,owner_OS_sha256=sha(eb),owner_stop_sha256=sha(fb),old_owner_four_core_wall_upper_s=400,
  original_setup_phases=phases,old_manifest_sha256=sha(mb),observed_epoch=time.time())
 local=base64.b64decode(read(root/'plans/MINIMAL_CORE_LOCAL_CAPTURE.private.json')['LOCAL_base64'])
 proposal=cold.propose_software_repair(gb,local,old_request=old_request,new_request=request,proof=proof,now_epoch=time.time())
 target=proposal['proposed_global'];delta={k:v for k,v in target.items()if k not in g or g[k]!=v};assert set(delta)=={'revision',x.PREPAID}
 put(root/'control/COLD_INPUT_AUDIT.private.json',raw(audit));put(root/'control/COLD_FIXED_REQUEST_PREVIEW.private.json',raw(request));put(root/'control/COLD_FULL_FUTURE_PREVIEW.private.json',raw(proposal['full_future_proof']))
 put(stage/'SOFTWARE_REPAIR_GLOBAL_DELTA.private.json',raw(delta));put(stage/'CLOSED_OWNER_PROOF.private.json',raw(proof));assert gp.read_bytes()==gb
 result=dict(status='PASS_ACTUAL_UNSUBMITTED_E2_NAMEERROR_COLD_SOURCE_REPAIR_FULL14_NO_GPU',root=str(root),source=P['source'],
  manifest_sha256=sha(raw(manifest)),source_seal_sha256=sha(raw(newseal)),audit_sha256=sha(raw(audit)),source_files=len(files),sealed_inputs=len(manifest['execution']['sealed_inputs']),
  before_GLOBAL_sha256=sha(gb),after_GLOBAL_sha256=sha(raw(target)),delta_sha256=sha(raw(delta)),closed_owner_proof_sha256=sha(raw(proof)),
  old_owner_CPU_upper_s=400,new_owner_CPU_upper_s=2600,full14_shortfalls={},old_owner_history_preserved=True,new_GPU_PBS_models_queries_updates=0,
  observed_epoch=time.time(),CPU_s=time.process_time()-began)
 put(stage/'RESULT.json',raw(result));print(json.dumps(result))
except BaseException as error:
 put(stage/'FAILED_OR_UNKNOWN.json',raw(dict(type=type(error).__name__,message=str(error)[-2000:],traceback=traceback.format_exc()[-12000:],CPU_s=time.process_time()-began,originals_preserved=True)));raise
'''
