"""One finite cold role assembled from an actually closed preparation.

Exact private parameters, paid control, source readback and the original
STAGE/LAUNCH transport are supplied by the existing owner. No model, qsub or
GLOBAL write occurs here. Ordinary runtime admission remains separate.
"""

PREPARE = r'''
import base64,hashlib,json,os,resource,sys,time,traceback
from copy import deepcopy
from pathlib import Path
sys.dont_write_bytecode=True;P=json.loads(sys.stdin.read());stage=Path(P['stage']);began=time.process_time()
resource.setrlimit(resource.RLIMIT_CPU,(178,179));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();sha=lambda b:hashlib.sha256(b).hexdigest()
def put(path,b):
 path=Path(path);path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
def read(path):return json.loads(Path(path).read_bytes())
try:
 assert os.getuid()==22182 and not os.environ.get('PBS_JOBID')and os.environ['CUDA_VISIBLE_DEVICES']==''
 os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
 old=Path(P['old_root']);mb=(old/'RUN_MANIFEST.json').read_bytes();m=json.loads(mb);sb=(old/'SOURCE_SEAL.json').read_bytes();seal=json.loads(sb)
 assert sha(mb)==P['closed']['manifest_sha256']and m['code']['vla_commit']==P['closed']['source_sha']and sha(sb)==m['code']['source_seal_sha256']
 b=read(old/'OUTPUT_BINDING.private.json');lease_bytes=Path(b['lease']).read_bytes();assert sha(lease_bytes)==P['closed']['closed_lease_sha256']and json.loads(lease_bytes)['status']=='CLOSED'
 gp=old.parent.parent/'HOPPER_RESOURCE_LEDGER.json';gb=gp.read_bytes();g=json.loads(gb);assert sha(gb)==P['GLOBAL_sha256']and g['pending_updates']==g['unknown_effects']==[]
 assert all(r['status']=='NATIVE_TERMINAL'for r in g['PBS_submissions'])
 root=old.with_name(P['identity']['run_uuid']);assert not root.exists();root.mkdir(mode=0o700)
 for n in ['control','pbs','evidence','reservations','training','datasets']:(root/n).mkdir(mode=0o700)
 oldcode=old/m['code']['capsule_relative'];code=root/'capsules'/P['identity']['source_sha']/'code';files=dict(seal['files']);frozen=m['code']['frozen_numerical_files']
 for name in sorted(set(files)|set(P['delta'])):
  rel=Path(name);assert not rel.is_absolute()and '..'not in rel.parts
  if name in P['delta']:
   row=P['delta'][name];body=base64.b64decode(row['base64']);assert sha(body)==row['sha256'];files[name]=row['sha256']
  else:
   f=oldcode/name;assert not f.is_symlink()and f.stat().st_uid==os.getuid()and f.stat().st_mode&0o777==0o400
   body=f.read_bytes();assert sha(body)==files[name]
  assert name not in frozen or sha(body)==frozen[name]
  put(code/name,body)
 newseal=dict(seal,source_sha=P['identity']['source_sha'],published_source_commit=P['identity']['source_sha'],
  files=files,base_source_seal_sha256=sha(sb),base_science_source=P['closed']['source_sha'],
  published_delta_sha256=P['delta_sha256'],new_scientific_tensor_links=0,new_scientific_tensor_copies=0)
 assert sha(raw(P['delta']))==P['delta_sha256']
 sys.path[:0]=[str(code),str(code/'src')]
 from scripts.pi0294.generalization import minimal_e2only_execution as x,minimal_e2only_contract as contract,minimal_e2only_physical as physical,minimal_e2only_pre_native as cold,minimal_remaining_physical as inventory
 from scripts.pi0294.dec1.source_cpu_once_io import commit
 assert root.name==contract.run_name(2,1)and cold.retired(g)[cold.OLD_REQUEST]==P['closed']
 payment=x.paid_retry_controls(g)[P['identity']['request_id']];assert payment['preparation_only']is True and payment['preparation_plan_sha256']==P['plan_sha256']
 assert all(payment[k]==v for k,v in P['identity'].items())
 snapshot=inventory.prefix_inventory(old);assert sha(raw(snapshot))==P['closed']['snapshot_sha256']and inventory.identity(snapshot)==P['closed']['snapshot_identity_sha256']
 small={}
 for name,pin in m['execution']['sealed_inputs'].items():
  if name in ['SOURCE_SEAL.json','MIX_OWNER_CONFIG.private.json','plans/MINIMAL_FROZEN_INPUT_PACKET.private.json',x.PHYSICAL]:continue
  body=(old/name).read_bytes();assert sha(body)==pin;small[name]=body
 custody=read(old/x.PHYSICAL);custody.update(**{k:P['identity'][k]for k in ['request_id','submission_nonce','run_uuid']})
 custody['retained_initial_bytes']+=snapshot['allocated_bytes'];custody['closed_pre_native_roots']=[dict(root=str(old),closure=P['closed'],snapshot_file='plans/E2_ONLY_CLOSED_PRE_NATIVE.private.json')]
 small['plans/E2_ONLY_CLOSED_PRE_NATIVE.private.json']=raw(snapshot);small[x.PHYSICAL]=raw(custody)
 regpath=Path(b['registry']).resolve(strict=True);envpath=Path(b['environment']).resolve(strict=True);regb=regpath.read_bytes();envb=envpath.read_bytes();reg,envs=map(json.loads,(regb,envb))
 role='PI0294_ARX5_E2_ONLY_RAW512_02_COLD01';profiles=[r for r in envs['profiles']if r['name']==x.PROFILE]
 assert len(profiles)==1 and profiles[0]['manifest']==str(old/'RUN_MANIFEST.json')and not any(r.get('role')==role for r in reg['entries'])
 envs.setdefault('closed_profile_history',[]).append(deepcopy(profiles[0]));envs['profiles']=[r for r in envs['profiles']if r['name']!=x.PROFILE]
 profile=deepcopy(profiles[0]);profile.update(role=role,manifest=str(root/'RUN_MANIFEST.json'));envs['profiles'].append(profile)
 env=json.loads(small['plans/A40_ENVIRONMENT_BINDING.private.json']);env.update(environment_registry_sha256=sha(raw(envs)));small['plans/A40_ENVIRONMENT_BINDING.private.json']=raw(env)
 cfg=read(old/'MIX_OWNER_CONFIG.private.json');request=deepcopy(cfg['request_template']);request.update(x.fixed_request_fields(2,1),**P['identity'],pre_native_predecessor=P['closed'],physical_custody_sha256=sha(small[x.PHYSICAL]))
 packet=raw({n:sha(v)for n,v in small.items()});small['plans/MINIMAL_FROZEN_INPUT_PACKET.private.json']=packet;request['frozen_input_packet_sha256']=sha(packet)
 cfg.update(source_sha=P['identity']['source_sha'],request_template=request)
 # Existing programs receive the current TARGET/REQUEST from the mature owner.
 # A literal old role path would require an explicit reviewed conversion.
 assert not any(str(old)in body for body in cfg['programs'].values())
 small['MIX_OWNER_CONFIG.private.json']=raw(cfg);small['SOURCE_SEAL.json']=raw(newseal)
 manifest=deepcopy(m);manifest.update(run_uuid=root.name,output_root=str(root));manifest['code'].update(vla_commit=P['identity']['source_sha'],capsule_relative='capsules/'+P['identity']['source_sha']+'/code',source_seal_sha256=sha(raw(newseal)))
 manifest['execution'].update(profile=x.PROFILE,sealed_inputs={n:sha(v)for n,v in small.items()})
 binding=dict(b,run_root=str(root),role=role,registry=str(regpath),environment=str(envpath),lease=str(Path(b['lease']).with_name('ARX5-E2-ONLY-RAW512-02-COLD01.json')),run_manifest_sha256=sha(raw(manifest)))
 assert not Path(binding['lease']).exists()
 for name,body in small.items():put(root/name,body)
 put(root/'RUN_MANIFEST.json',raw(manifest));put(root/'OUTPUT_BINDING.private.json',raw(binding))
 from scripts.pi0294.pv1.model_inputs import prepare_alias
 alias=prepare_alias(root,json.loads(small['NEURAL_BINDINGS.private.json']))
 lease=json.loads(lease_bytes);lease={k:v for k,v in lease.items()if k not in ['closed_epoch','closure_reason','pre_native_owner_OS_sha256','PBS_submitted']};lease.update(status='ACTIVE',run_uuid=root.name,role=role,manifest_sha256=binding['run_manifest_sha256'],CPU_upper_s=3000,new_GPU_permission=False);put(binding['lease'],raw(lease))
 reg['entries'].append(dict(role=role,scope=x.VERSION,path=str(root),run_uuid=root.name,executor_node=os.uname().nodename,owner='EXISTING_SOLE_MUTATION_OWNER',lifecycle='PRE_NATIVE_COLD_NO_GPU'))
 for path,before,after,key in [(regpath,regb,raw(reg),'E2-PRE-NATIVE30-NEW-REGISTERED-ROLE'),(envpath,envb,raw(envs),'E2-PRE-NATIVE30-NEW-ENV-PROFILE')]:
  commit(path,expected_before_sha256=sha(before),proposed_bytes=after,request_id=key,intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_suffix('.owner.lock'));assert path.read_bytes()==after
 for name,pin in files.items():assert sha((code/name).read_bytes())==pin
 for name,pin in manifest['execution']['sealed_inputs'].items():assert sha((root/name).read_bytes())==pin
 from scripts.pi0294.generalization.minimal_saved512_inputs import Inputs
 Inputs(root,manifest,guard=lambda:None).verify_current()
 oldaudit=(old/'control/COLD_INPUT_AUDIT.private.json').read_bytes();assert sha(oldaudit)==P['old_audit_sha256']and json.loads(oldaudit)['actual_twelve_reader_rows']==7212
 audit=dict(status='PASS_ACTUAL_MINIMAL_FIRST512_COLD_INPUTS_NO_MODEL',source_sha=P['identity']['source_sha'],run_manifest_sha256=sha(raw(manifest)),source_files=len(files),sealed_inputs=len(small),actual_twelve_reader_rows=7212,
  original_reader_qualification_reused=dict(audit_sha256=sha(oldaudit),own_input_bytes_reverified=True),new_GPU_PBS_model_query_update_physics=0,observed_epoch=time.time(),CPU_actual_s=time.process_time()-began)
 from scripts.pi0294.generalization import minimal_native_launch as launch
 from scripts.pi0294.dec1 import vanda_campaign as queue
 from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside
 effective=launch.collect_effective_queue(walltime_s=22440);request.update(run_manifest_sha256=sha(raw(manifest)),cold_input_audit_sha256=sha(raw(audit)),submit_count=len(g['PBS_submissions'])+1,created_epoch=time.time(),effective_queue_proof_sha256=sha(raw(effective)))
 x.request(request);guard=physical.Guard(root,request,manifest,lambda **_:None);proof=guard.inspect(next_output_upper_bytes=physical.CONTROL_GROWTH_RESERVE)
 full=queue.collect_queue();rows=queue.queue_rows(full,now_epoch=time.time());own,outside,unknown=queue.classify(g,rows);assert own==unknown==[];checked_outside(rows,outside,base64.b64decode(cfg['outside_confirmation_base64']))
 zero={k:(root/'plans'/n).read_bytes()for k,n in [('lease_bytes','E2_ONLY_PREVIOUS_ZERO_LEASE.private.json'),('settled_bytes','E2_ONLY_PREVIOUS_ZERO_SETTLED.private.json'),('finite_OS_bytes','E2_ONLY_PREVIOUS_ZERO_FINITE_OS0.private.json')]}
 fresh=dict(GLOBAL_sha256=sha(gb),closed_zero_predecessor=request['zero_effect_predecessor'],no_existing_new_owner_or_dispatch=True,observed_epoch=time.time())
 local=base64.b64decode(json.loads(small['plans/MINIMAL_CORE_LOCAL_CAPTURE.private.json'])['LOCAL_base64'])
 proposal=x.propose_retry_control(gb,local,request_bytes=raw(request),closed_zero_files=zero,closed_readback=fresh,now_epoch=time.time())
 assert proposal['proposed_global']['CPU_s']==g['CPU_s']and proposal['full_future_proof']['cap_changes_required']=={}and gp.read_bytes()==gb
 put(root/'control/COLD_INPUT_AUDIT.private.json',raw(audit));put(root/'control/COLD_FIXED_REQUEST_PREVIEW.private.json',raw(request));put(root/'control/COLD_FULL_FUTURE_PREVIEW.private.json',raw(proposal['full_future_proof']))
 result=dict(status='PASS_ACTUAL_E2_PRE_NATIVE_COLD_ROLE_INPUTS_FULL14_NO_GPU',root=str(root),source=P['identity']['source_sha'],request_id=request['request_id'],manifest_sha256=sha(raw(manifest)),source_seal_sha256=sha(raw(newseal)),audit_sha256=sha(raw(audit)),source_files=len(files),sealed_inputs=len(small),full14_shortfalls={},old_preparation_retained_allocated_bytes=snapshot['allocated_bytes'],new_GPU_PBS_models_queries_updates=0,CPU_s=time.process_time()-began,observed_epoch=time.time())
 put(stage/'RESULT.json',raw(result));print(json.dumps(result))
except BaseException as error:
 put(stage/'FAILED_OR_UNKNOWN.json',raw(dict(type=type(error).__name__,message=str(error)[-2000:],traceback=traceback.format_exc()[-12000:],CPU_s=time.process_time()-began,originals_preserved=True)));raise
'''
