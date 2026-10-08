"""The previously executed inline remote preview, saved for owner reuse. No unlink."""
import json,sys
from pathlib import Path

WORKTREE=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(WORKTREE/'scripts/pi0276_execution'))
from coordinator import remote

PROGRAM=r'''
import hashlib,json,os,re,shutil,stat,subprocess,time
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0285');uid=os.getuid()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def inventory(folder):
 out=[]
 for p in sorted(folder.iterdir()):
  s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==uid and s.st_nlink==1 and p.resolve().parent==folder.resolve(),str(p)
  out.append(dict(path=str(p),name=p.name,bytes=s.st_size,allocated_bytes=s.st_blocks*512,sha256=sha(p),inode=s.st_ino,device=s.st_dev,uid=s.st_uid,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns))
 return out
before=shutil.disk_usage(r)
q=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qstat','-xf','655301.hopper-m-02'],capture_output=True,text=True);assert q.returncode==0 and re.search(r'job_state\s*=\s*F\b',q.stdout) and re.search(r'Exit_status\s*=\s*0\b',q.stdout)
auditp=r/'evidence/PLACE4096-4096_TRAINING_CPU.json';audit=json.loads(auditp.read_text());assert audit['status']=='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE'
protected=r/'learning/PLACE4096/main/checkpoint-4096';full=inventory(protected)
assert [{k:x[k] for k in ('name','bytes','sha256')} for x in full]==[dict(name=x['path'],bytes=x['bytes'],sha256=x['sha256']) for x in audit['entries']]
closed=r/'runtime/ALLOCATION_CLOSED_655301.hopper-m-02.json';assert json.loads(closed.read_text())['status']=='FIXED_SEGMENT_COMPLETE'
receipt=r/'evidence/PLACE4096/checkpoint-2048.archive_receipt.json';readback=r/'evidence/PLACE4096/checkpoint-2048.full_readback.json'
ar=json.loads(receipt.read_text());hf=json.loads(readback.read_text());assert ar['revision']==hf['revision']=='43ee74595d710c3f581ef61c19104620b0bdfdf0' and hf['status']=='PASS_FULL_FILE_READBACK' and hf['complete_trainer'] and len(hf['files'])==14 and ar['entries']==hf['files'] and ar['tree_sha256']==hf['tree_sha256']
targets=[];groups={}
for step in (2048,3072):
 folder=r/f'learning/PLACE4096/main/checkpoint-{step}';inv=inventory(folder);assert len(inv)==14
 assert json.loads((folder/'trainer_state.json').read_text())['global_step']==step
 assert json.loads((folder/'PI0282_RECOVERY.json').read_text())['cursor']==step*8
 if step==2048:assert [dict(path=x['name'],bytes=x['bytes'],sha256=x['sha256']) for x in inv]==hf['files']
 payload=[x for x in inv if x['name'] in ('model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','optimizer.pt')];assert len(payload)==3
 groups[str(step)]=dict(folder=str(folder),all_entries=inv,retire=payload,retained_metadata=[x for x in inv if x not in payload],archive_binding='HF_FULL14_FIXED_REVISION_PASS' if step==2048 else 'SUPERSEDED_ROLLING_RESUME_NO_ELIGIBLE_QUALITY_POINT_FULL4096_PRESERVED')
 targets.extend(payload)
for point,v in CORE['points'].items():
 folder=r/'delivery'/point;ex=folder/'EXPORT.json';assert sha(ex)==v['EXPORT.json']['sha256']
 e=json.loads(ex.read_text());delta=folder/'exact_changed_tensors.safetensors'
 if not delta.exists():continue
 ds=delta.lstat();assert stat.S_ISREG(ds.st_mode) and ds.st_uid==uid and ds.st_nlink==1 and delta.resolve().parent==folder.resolve()
 assert sha(delta)==e['exact_delta_sha256'] and ds.st_size==e['exact_delta_bytes']
 assert v['TRAINING_CPU.json']['value']['export_sha256']==sha(ex) and v['MATERIALIZATION.json']['value']['export_receipt_sha256']==sha(ex)
 assert any(x['path']=='exact_changed_tensors.safetensors' and x['sha256']==e['exact_delta_sha256'] and x['bytes']==ds.st_size for x in v['DIRECT_READBACK.json']['value']['files'])
 targets.append(dict(path=str(delta),name=delta.name,bytes=ds.st_size,allocated_bytes=ds.st_blocks*512,sha256=e['exact_delta_sha256'],inode=ds.st_ino,device=ds.st_dev,uid=ds.st_uid,nlink=ds.st_nlink,mtime_ns=ds.st_mtime_ns))
consumers=[];unknown=[];own=[];infrastructure=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=uid:continue
  tokens=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x];own.append(dict(pid=int(proc.name),argv=tokens))
  status=(proc/'status').read_text();name=status.splitlines()[0].split(':',1)[1].strip()
  if (name=='(sd-pam)' and tokens==['(sd-pam)']) or (name=='sshd-session' and len(tokens)==1 and tokens[0].strip()=='sshd-session: e1192548@notty'):
   infrastructure.append(dict(pid=int(proc.name),name=name,argv=tokens,reason='POSITIVELY_CLASSIFIED_PAM_OR_SSH_SESSION_MONITOR_NOT_MODEL_APPLICATION'));continue
  for x in targets:
   if any(x['path'] in a or Path(x['path']).parent.name in a for a in tokens):consumers.append(dict(pid=int(proc.name),kind='ARGV',path=x['path']))
  for fd in (proc/'fd').iterdir():
   try:p=str(fd.resolve(strict=True))
   except FileNotFoundError:continue
   for x in targets:
    if p==x['path']:consumers.append(dict(pid=int(proc.name),kind='FD',path=p))
 except FileNotFoundError:continue
 except (PermissionError,OSError) as error:unknown.append(dict(pid=int(proc.name),error=str(error)))
assert not consumers and not unknown,(consumers,unknown)
quota=subprocess.run(['/bin/bash','-lc','hpc quota'],capture_output=True,text=True)
out=dict(status='EXACT_PREVIEW_NO_MUTATIONS',epoch=time.time(),root=str(r),targets=targets,groups=groups,protected4096=dict(root=str(protected),entries=full,audit_sha256=sha(auditp)),scheduler=dict(job='655301.hopper-m-02',terminal=True,exit_status=0,raw_sha256=hashlib.sha256(q.stdout.encode()).hexdigest()),own_processes=own,consumers=consumers,unknown=unknown,logical_bytes=sum(x['bytes'] for x in targets),allocated_bytes=sum(x['allocated_bytes'] for x in targets),filesystem_free_before=before.free,quota=dict(returncode=quota.returncode,stdout=quota.stdout,stderr=quota.stderr) if quota else 'COMMAND_UNAVAILABLE',CORE_proof_epoch=CORE['epoch'])
out['excluded_infrastructure']=infrastructure
preview=r/'archive/DOSE_CAPACITY_RECLAIM_PREVIEW.json';assert not preview.exists();preview.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
'''

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    core=json.loads((here/'DOSE_CAPACITY_RECLAIM_CORE_PROOF.json').read_text())
    value=json.loads(remote('hop','CORE='+repr(core)+'\n'+PROGRAM))
    (here/'DOSE_CAPACITY_RECLAIM_PREVIEW.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],files=len(value['targets']),logical_GiB=value['logical_bytes']/1024**3,allocated_GiB=value['allocated_bytes']/1024**3,consumers=value['consumers'],unknown=value['unknown']),indent=2))
