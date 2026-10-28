"""Owner PC helper: exact closed PI0286 slots to durable NFS, then free their SSD copy."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def offload(slots):
    assert slots and len(slots)==len(set(slots)) and all(Path(s).name==s and s not in ('.','..') for s in slots)
    result=json.loads(remote('core',f'''import json,os,hashlib,shutil,subprocess,time
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286');ssd=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0286/source/episodes');slots={slots!r}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def inventory(root):
 out={{}}
 for p in root.rglob('*'):
  assert not p.is_symlink(),('Nested symlink not admitted',p)
  if p.is_file():
   s=p.stat();assert s.st_uid==os.getuid() and s.st_nlink==1
   out[str(p.relative_to(root))]={{'size':s.st_size,'sha256':sha(p)}}
 return out
def closed():
 active=[]
 for proc in Path('/proc').glob('[0-9]*'):
  try:
   if proc.stat().st_uid!=os.getuid():continue
   b=(proc/'cmdline').read_bytes()
   if str(r/'code/').encode() in b:active.append((int(proc.name),'scientific_process'))
   for fd in (proc/'fd').iterdir():
    try:t=Path(os.readlink(fd))
    except OSError:continue
    if any(t.is_relative_to(ssd/s) for s in slots):active.append((int(proc.name),'slot_fd'))
  except FileNotFoundError:continue
  except PermissionError:
   comm=(proc/'comm').read_text().strip()
   assert comm in ('sd-pam','(sd-pam)','sshd-session','systemd') or (comm=='sshd' and b.startswith(b'sshd: ') and b'@notty' in b),('Unidentified own consumer',proc.name,comm)
 assert not active,active
closed()
# Require actual correction assembly closure before relocating any learner prefix.
bank=r/'correction/CORRECTION_BANK.json';b=json.loads(bank.read_text())
assert b['status'] in ('PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC','CORRECTION_NOT_ADMITTED')
collection=r/'source/evidence/CORRECTIONS32.json';c=json.loads(collection.read_text())
assert b['collection_sha256']==sha(collection) and c['requested']==len(c['slots'])==32
audit_hashes={{}}
for panel in (r/'source/evidence').glob('*.json'):
 cpu=r/'audit'/(panel.stem+'_CPU.json')
 if not cpu.exists():continue
 v=json.loads(panel.read_text());a=json.loads(cpu.read_text())
 if a.get('status')!='PASS_INDEPENDENT_CPU_CONTACT_SCORE_CLOCK_INSTALLATION_AND_PAIRED_INITIAL_INPUTS':continue
 assert a['source_sha256']==sha(panel)
 for row in a['rows']:audit_hashes[row['source_result_sha256']]={{'panel_sha256':sha(panel),'CPU_sha256':sha(cpu)}}
for row in b['CPU_readbacks']:
 assert row['status']=='PASS_INDEPENDENT_CPU_CORRECTION_CONTACT_CLOCK_STATE_GRIPPER_SCORE'
 audit_hashes[row['source_result_sha256']]={{'correction_bank_sha256':sha(bank)}}
out=[];free_before=shutil.disk_usage(ssd).free
for slot in slots:
 source=ssd/slot;assert source.is_dir() and not source.is_symlink()
 assert source.resolve()==ssd.resolve()/slot and source.resolve().is_relative_to(ssd.resolve())
 attempts=sorted(source.glob('attempt-*'));assert attempts
 valid=[]
 for attempt in attempts:
  result=attempt/'RESULT.json';assert result.is_file(),'Preserve open engineering prefix'
  v=json.loads(result.read_text())
  if v['status'].startswith('VALID'):
   h=sha(result);assert h in audit_hashes,('Independent CPU closure missing',slot)
   for name,value in v['files'].items():assert sha(attempt/name)==value
   valid.append({{'result_sha256':h,'CPU':audit_hashes[h]}})
 assert len(valid)==1,'Only exactly one completed valid physical episode per slot'
 original=inventory(source);dest=r/'durable/episodes'/slot
 assert dest.resolve().is_relative_to(r.resolve()) and dest!=source
 dest.mkdir(parents=True,exist_ok=True)
 subprocess.run(['rsync','-a',str(source)+'/',str(dest)+'/'],check=True)
 assert inventory(dest)==original and inventory(source)==original
 closed()
 receipt=dict(status='PASS_CLOSED_SLOT_DURABLE_NFS_FULL_MEMBER_READBACK_READY_EXACT_SSD_RETIREMENT',slot=slot,original_SSD_root=str(source),durable_root=str(dest),members=original,valid=valid,correction_bank_sha256=sha(bank),model_queries=0,physics=0,epoch=time.time())
 proof=r/'evidence/offload'/(slot+'.json');proof.parent.mkdir(parents=True,exist_ok=True);assert not proof.exists();proof.write_text(json.dumps(receipt,indent=2)+chr(10))
 # Exact checked own directory only. All failed attempts and small proofs remain in the verified NFS slot.
 assert source.resolve()==ssd.resolve()/slot and not source.is_symlink() and inventory(source)==original
 shutil.rmtree(source);source.symlink_to(dest,target_is_directory=True)
 assert source.resolve()==dest.resolve() and inventory(dest)==original
 receipt['status']='PASS_CLOSED_SLOT_DURABLE_READBACK_AND_EXACT_SSD_OFFLOAD_LOGICAL_PATH_PRESERVED';proof.write_text(json.dumps(receipt,indent=2)+chr(10));out.append(receipt)
print(json.dumps(dict(status='PASS_OWNER_EXPLICIT_CLOSED_SLOTS_OFFLOADED',slots=out,free_before_bytes=free_before,free_after_bytes=shutil.disk_usage(ssd).free,model_queries=0,physics=0)))
'''))
    path=Path('reports/PI0286/preparation/OFFLOAD_CLOSED_'+str(len(slots))+'.json');path.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--slots',nargs='+',required=True);a=p.parse_args()
    v=offload(a.slots);print(json.dumps({k:x for k,x in v.items() if k!='slots'}))
