"""Original finite model-only uploader with explicit closed C/D1024 admission.

Token stays in stdin/process memory. A returned/unknown commit is never resent.
Source and fixed-revision remote identity checks retain the mature protocol.
"""
import base64,hashlib,io,json,os,resource,sys,time,traceback
from pathlib import Path
P=json.loads(sys.stdin.read());token=P.pop('token');root=Path(P['upload_root']);root.mkdir(mode=0o700)
SHA=lambda b:hashlib.sha256(b).hexdigest();RAW=lambda x:(json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def put(n,v):
 with (root/n).open('xb') as f:f.write(RAW(v));f.flush();os.fsync(f.fileno())
 (root/n).chmod(0o400)
def identity(path):
 p=Path(path);s=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 t=p.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns)
 return s.st_size,h.hexdigest()
began=time.process_time();started=time.time();phase='PREUPLOAD_SOURCE_VERIFY';commit_started=False
try:
 assert os.getuid()==22182 and os.environ.get('USER')=='e1192548' and not os.environ.get('PBS_JOBID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 resource.setrlimit(resource.RLIMIT_CPU,(600,600));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 assert time.time()<P['stop_new_epoch'];control=Path(P['canonical_parent'])/'HOPPER_RESOURCE_LEDGER.json';g=json.loads(control.read_bytes())
 from scripts.pi0294.pv1.mix_delivery_control import upload_admission
 assert P['candidate']==P['compiled_plan']['public_plan']['candidate']
 extra={}
 if P['candidate']in('E1_ADAPTER_ONLY','E2_SHARED_ACTION_HEAD'):
  manifest_bytes=(Path(P['control_root'])/'CONTROL_RUN_MANIFEST.json').read_bytes();manifest=json.loads(manifest_bytes)
  assert manifest['canonical_parent']==P['canonical_parent']and manifest['control_root']==P['control_root']and manifest['source_sha']==P['source_sha']and manifest['candidate']==P['candidate']
  parent=Path(P['canonical_parent']);binding=json.loads((parent/'OUTPUT_BINDING.private.json').read_bytes());lease=json.loads(Path(binding['lease']).read_bytes())
  qualified=P['compiled_plan'].get('private_restoration',{}).get('closed_evidence',{}).get('owner_observation',{}).get('qualified_saved_E1_source')
  if P['candidate']=='E1_ADAPTER_ONLY'and qualified is not None:
   from scripts.pi0294.generalization.minimal_qualified_e1_source import source_control_role
   source_control_role(P['control_root'],manifest_bytes,canonical_parent=P['canonical_parent'])
  else:
   assert SHA((parent/'RUN_MANIFEST.json').read_bytes())==manifest['parent_manifest_sha256']and lease['status']in('ACTIVE','ACTIVE_CONTROL_PLANE_ONLY')and lease['manifest_sha256']==manifest['parent_manifest_sha256']
  assert manifest['repo_id']==P['repo_id']and manifest['HF_prefix']==P['prefix']and Path(P['upload_root'])==Path(P['control_root'])/'upload'
  extra=dict(source_control_manifest_bytes=manifest_bytes,source_checkpoint_root=P['checkpoint'])
 admission=upload_admission(g,P['compiled_plan'],now_epoch=time.time(),checkpoint_complete_bytes=(Path(P['checkpoint'])/'CHECKPOINT_COMPLETE.json').read_bytes(),**extra)
 assert admission['model_or_physics_admission']is False and admission['source_CPU_upper_s']==600
 assert P['compiled_plan']['public_plan']==P['public_plan'] and P['compiled_plan']['public_rewritten_metadata_base64']==P['rewritten']
 complete=(Path(P['checkpoint'])/'CHECKPOINT_COMPLETE.json').read_bytes();assert SHA(complete)==P['checkpoint_sha256']
 plans=P['public_plan']['files'];operations=[];before={}
 from huggingface_hub import HfApi,CommitOperationAdd
 api=HfApi(token=token);assert api.whoami()['name']=='leoyuer';info=api.repo_info(P['repo_id'],repo_type='dataset',timeout=15);assert info.private is False
 parent_commit=info.sha;current=api.list_repo_files(P['repo_id'],repo_type='dataset',revision=parent_commit)
 assert len(current)<=10000 and not any(x.startswith(P['prefix']+'/') for x in current)
 for name,expected in plans.items():
  if name in P['rewritten']:
   raw=base64.b64decode(P['rewritten'][name],validate=True);assert (len(raw),SHA(raw))==(expected['bytes'],expected['sha256']);source=raw
  else:
   source=Path(P['checkpoint'])/name;before[name]=identity(source);assert before[name]==(expected['bytes'],expected['sha256'])
  operations.append(CommitOperationAdd(path_in_repo=P['prefix']+'/'+name,path_or_fileobj=source))
 for name,item in P['notices'].items():
  raw=base64.b64decode(item['base64'],validate=True);assert len(raw)==item['bytes'] and SHA(raw)==item['sha256'];operations.append(CommitOperationAdd(path_in_repo=P['prefix']+'/'+name,path_or_fileobj=raw))
 put('SOURCE_FULL_STREAMING_VERIFIED.json',dict(candidate=P['candidate'],files={n:dict(bytes=z[0],sha256=z[1]) for n,z in before.items()},CPU_s=time.process_time()-began,observed_epoch=time.time()))
 put('COMMIT_INTENT.json',dict(schema=admission.get('commit_intent_schema','V5_CD_MODEL_ONLY_PUBLIC_HF_COMMIT_ONCE_INTENT_V1'),candidate=P['candidate'],repo_id=P['repo_id'],prefix=P['prefix'],parent_commit=parent_commit,checkpoint_sha256=P['checkpoint_sha256'],public_plan_sha256=SHA(RAW(P['public_plan'])),automatic_retry=False,observed_epoch=time.time()))
 phase='ONE_CREATE_COMMIT';commit_started=True
 result=api.create_commit(repo_id=P['repo_id'],repo_type='dataset',operations=operations,commit_message='Audited '+P['candidate']+' saved'+str(P['public_plan']['checkpoint_step'])+' inference-only delivery',parent_commit=parent_commit)
 revision=result.oid;assert len(revision)==40;put('COMMIT_RETURNED.json',dict(revision=revision,repo_id=P['repo_id'],prefix=P['prefix'],commit_url=result.commit_url))
 phase='REMOTE_CONTENT_ADDRESSED_READBACK';remote={x.path:x for x in api.list_repo_tree(P['repo_id'],repo_type='dataset',revision=revision,path_in_repo=P['prefix'],recursive=True)}
 for name,expected in dict(plans,**P['notices']).items():
  member=remote[P['prefix']+'/'+name];assert member.size==expected['bytes']
  if name.endswith('.safetensors'):
   lfs=member.lfs;digest=lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256'];assert digest==expected['sha256']
  else:
   import requests
   url='https://huggingface.co/datasets/'+P['repo_id']+'/resolve/'+revision+'/'+P['prefix']+'/'+name
   response=requests.get(url,timeout=30);response.raise_for_status();assert (len(response.content),SHA(response.content))==(expected['bytes'],expected['sha256'])
 for name,old in before.items():assert identity(Path(P['checkpoint'])/name)==old
 completed=dict(status='HF_PUBLIC_MODEL_ONLY_FIXED_REVISION_FULL_IDENTITY_VERIFIED',candidate=P['candidate'],revision=revision,repo_id=P['repo_id'],prefix=P['prefix'],payload_bytes=P['public_plan']['payload_bytes'],notice_bytes=sum(z['bytes'] for z in P['notices'].values()),full_stream_source_SHA_before_and_after=True,remote_all9_content_addressed_or_full_small_SHA=True,new_model_loads=0,new_queries=0,new_training=0,CPU_s=time.process_time()-began,completed_epoch=time.time())
 if P['public_plan']['checkpoint_step']==512:
  completed.pop('remote_all9_content_addressed_or_full_small_SHA');completed.update(remote_all_declared_content_addressed_or_full_small_SHA=True,inference_member_count=len(plans),checkpoint_step=512)
 put('COMPLETED.json',completed)
except BaseException as exc:
 # No repr/traceback of SDK inputs or HTTP headers; no token may reach logs.
 put('FAILED_OR_UNKNOWN.json',dict(status='UNKNOWN_COMMIT_NO_AUTOMATIC_RESEND' if commit_started else 'KNOWN_PRECOMMIT_FAILURE_NO_COMMIT_SENT',phase=phase,error_type=type(exc).__name__,HTTP_status=getattr(getattr(exc,'response',None),'status_code',None),CPU_s=time.process_time()-began,observed_epoch=time.time(),automatic_retry=False));raise SystemExit(1)
finally:
 token=None
