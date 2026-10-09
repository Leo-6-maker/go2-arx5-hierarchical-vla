"""Exact owner-authorized PI0284 six-shard retirement; preserve every small proof."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def retire():
    preview=json.loads(Path('reports/PI0286/preparation/SSD_RECLAIM_EXTRA_PREVIEW.json').read_text())
    rows=preview['rows']
    assert [r['point'] for r in rows]==['CTRL2048','LR5X1024','LR5X2048']
    result=json.loads(remote('core',f'''import hashlib,json,os,shutil,time
from pathlib import Path
rows={rows!r};needles=[r['path'].encode() for r in rows]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def consumers():
 found=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:
   if p.stat().st_uid!=os.getuid():continue
   argv=(p/'cmdline').read_bytes()
   if any(n in argv for n in needles):found.append((p.name,'argv'))
   if any(n in (p/'maps').read_bytes() for n in needles):found.append((p.name,'maps'))
   for fd in (p/'fd').iterdir():
    try:t=os.readlink(fd).encode()
    except OSError:continue
    if any(n in t for n in needles):found.append((p.name,'fd'))
  except FileNotFoundError:continue
  except PermissionError:
   comm=(p/'comm').read_text().strip()
   assert comm in ('sd-pam','(sd-pam)','sshd-session','systemd') or (comm=='sshd' and argv.startswith(b'sshd: ') and b'@notty' in argv),('Unidentified own consumer',p.name,comm)
 assert not found,found
def restat(f):
 p=Path(f['path']);assert p.is_file() and not p.is_symlink()
 assert p.resolve().is_relative_to(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0284/models').resolve())
 s=p.stat()
 for key,attr in [('size','st_size'),('ino','st_ino'),('dev','st_dev'),('uid','st_uid'),('nlink','st_nlink')]:assert f[key]==getattr(s,attr),(p,key)
 assert s.st_uid==os.getuid() and s.st_nlink==1
 return s
consumers();before=shutil.disk_usage('/var/cache').free;files=[];small={{}}
for row in rows:
 root=Path(row['path']);assert root.name==row['point'] and not root.is_symlink()
 cpu=Path(row['CPU_path']);assert sha(cpu)==row['CPU_sha256'];a=json.loads(cpu.read_text())
 assert a['status']=='PASS_INDEPENDENT_CPU_NATIVE_TASK_AND_ACTUAL_INSTALLATION_RECOMPUTE' and a['persistent_count']==0 and len(a['rows'])==4 and a['selected']==a['deployment']=='NONE'
 material=root/'MATERIALIZATION.json';assert sha(material)==row['materialization_sha256'];m=json.loads(material.read_text())
 assert m['status']=='PASS_EXACT_TRANSFER_MASTER_READBACK_NATIVE_BF16_VIEW' and m['export_receipt_sha256']==row['export_receipt_sha256']
 large={{f['path'] for f in row['files']}}
 small[str(root)]={{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and str(p) not in large}}
 for f in row['files']:
  s=restat(f);h=sha(f['path']);assert h==f['expected_sha256']==m['files'][Path(f['path']).name]
  files.append(dict(f,actual_sha256=h,current_allocated_bytes=s.st_blocks*512))
consumers()
for f in files:restat(f)
for f in files:Path(f['path']).unlink();assert not Path(f['path']).exists()
for root,members in small.items():
 for name,h in members.items():assert sha(Path(root)/name)==h
after=shutil.disk_usage('/var/cache').free
out=dict(status='DELETED_VERIFIED_EXACT_SIX_OWNER_AUTHORIZED_FAILED_PI0284_VIEW_SHARDS',files=files,logical_bytes=sum(f['size'] for f in files),allocated_bytes=sum(f['current_allocated_bytes'] for f in files),free_before_bytes=before,free_after_bytes=after,observed_shared_free_delta_bytes=after-before,small_proof_hashes_preserved=small,PI0285_DOSE_parent_touched=False,old_HF_history_touched=False,personal_quota='UNKNOWN_NOT_DF',model_queries=0,physics=0,epoch=time.time())
for row in rows:(Path(row['path'])/'PI0286_RETIREMENT_EXTRA.json').write_text(json.dumps(out,indent=2)+chr(10))
print(json.dumps(out))
'''))
    Path('reports/PI0286/preparation/SSD_RECLAIM_EXTRA_EXECUTED.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    v=retire();print(json.dumps({k:x for k,x in v.items() if k not in ('files','small_proof_hashes_preserved')}))
