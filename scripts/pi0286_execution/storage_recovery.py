"""Owner-approved exact four-shard retirement and stopped PI0286 storage migration."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def retire():
    report=json.loads(Path('reports/PI0286/preparation/SSD_RECLAIM_EXACT_CANDIDATES.json').read_text())
    approved=('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0285/models/PLACE4096-4096',
              '/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0284/models/CTRL1024')
    selected=[r for r in report['roots'] if r['path'] in approved]
    assert len(selected)==2 and sum(len(r['large_files']) for r in selected)==4
    result=json.loads(remote('core',f'''import json,hashlib,os,time,shutil
from pathlib import Path
rows={selected!r};targets=[x for r in rows for x in r['large_files']]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def stat(x):
 p=Path(x['path']);assert p.is_file() and not p.is_symlink();s=p.stat()
 for k,a in [('size','st_size'),('nlink','st_nlink'),('uid','st_uid'),('ino','st_ino'),('dev','st_dev')]:assert x[k]==getattr(s,a),(p,k)
 assert s.st_uid==os.getuid() and s.st_nlink==1
 return s
def consumers():
 found=[];needles=[str(Path(r['path'])).encode() for r in rows]
 for proc in Path('/proc').glob('[0-9]*'):
  try:
   if proc.stat().st_uid!=os.getuid():continue
   argv=(proc/'cmdline').read_bytes()
   if any(n in argv for n in needles):found.append((int(proc.name),'argv'))
   maps=(proc/'maps').read_bytes()
   if any(n in maps for n in needles):found.append((int(proc.name),'maps'))
   for fd in (proc/'fd').iterdir():
    try:t=os.readlink(fd).encode()
    except OSError:continue
    if any(n in t for n in needles):found.append((int(proc.name),'fd'))
  except FileNotFoundError:continue
  except PermissionError:
   # Positively identified session monitors have no application model authority.
   try:comm=(proc/'comm').read_text().strip()
   except OSError:comm='UNKNOWN'
   if comm=='sshd':
    assert argv.startswith(b'sshd: ') and b'@notty' in argv,('Unidentified SSH process',proc.name)
   else:assert comm in ('sd-pam','(sd-pam)','sshd-session','systemd'),('Unreadable own application consumer',proc.name,comm)
 assert not found,found
before=shutil.disk_usage('/var/cache').free;verified=[]
consumers()
for r in rows:
 m=Path(r['path'])/'MATERIALIZATION.json';v=json.loads(m.read_text())
 assert v['status']=='PASS_EXACT_TRANSFER_MASTER_READBACK_NATIVE_BF16_VIEW'
 for x in r['large_files']:
  s=stat(x);h=sha(x['path']);assert h==v['files'][Path(x['path']).name]
  verified.append(dict(x,sha256=h,allocated_bytes=s.st_blocks*512,materialization_sha256=sha(m)))
consumers()
for x in verified:stat(x)
for x in verified:Path(x['path']).unlink();assert not Path(x['path']).exists()
after=shutil.disk_usage('/var/cache').free
out=dict(status='DELETED_VERIFIED_EXACT_FOUR_OWNER_APPROVED_FAILED_MODEL_SHARDS',files=verified,logical_bytes=sum(x['size'] for x in verified),allocated_bytes=sum(x['allocated_bytes'] for x in verified),free_before_bytes=before,free_after_bytes=after,observed_shared_free_delta_bytes=after-before,personal_quota='UNKNOWN_NOT_DF',small_metadata_retained=True,protected_parent_unchanged=True,model_queries=0,physics=0,epoch=time.time())
for r in rows:(Path(r['path'])/'PI0286_RETIREMENT.json').write_text(json.dumps(out,indent=2)+chr(10))
print(json.dumps(out))
'''))
    Path('reports/PI0286/preparation/SSD_RECLAIM_EXECUTED.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def durable():
    """Copy and hash all closed source files to NFS; retire no source payload."""
    result=json.loads(remote('core', '''import json,hashlib,subprocess,time
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286');source=r/'source';dest=r/'durable/source'
active=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:b=p.read_bytes()
 except OSError:continue
 if str(r/'code/').encode() in b:active.append(int(p.parent.name))
assert not active,('Durable full readback requires a closed scientific boundary',active)
assert source.resolve()==Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0286/source')
def inventory(root):
 out={}
 for p in root.rglob('*'):
  if not p.is_file():continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  out[str(p.relative_to(root))]={'size':p.stat().st_size,'sha256':h.hexdigest()}
 return out
before=inventory(source);dest.mkdir(parents=True,exist_ok=True)
# Dereference the one quality-ledger alias so the NFS readback is independently durable.
subprocess.run(['rsync','-aL',str(source)+'/',str(dest)+'/'],check=True)
after=inventory(dest);assert all(after.get(k)==v for k,v in before.items()) and inventory(source)==before
v=dict(status='PASS_STOPPED_SOURCE_DURABLE_NFS_EXACT_READBACK_NO_DELETION',files=before,source_root=str(source.resolve()),durable_root=str(dest),model_queries=0,physics=0,epoch=time.time())
(r/'evidence/DURABLE_SOURCE_READBACK.json').write_text(json.dumps(v,indent=2)+chr(10));print(json.dumps(v))
'''))
    Path('reports/PI0286/preparation/DURABLE_SOURCE_READBACK.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def migrate():
    result=json.loads(remote('core', '''import json,hashlib,os,time,shutil,subprocess
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286');local=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0286')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def inventory(root):
 out={}
 for p in root.rglob('*'):
  if p.is_symlink():out[str(p.relative_to(root))]={'symlink':os.readlink(p)}
  elif p.is_file():out[str(p.relative_to(root))]={'size':p.stat().st_size,'sha256':sha(p)}
 return out
# Fully stopped boundary; no code, scene, model or physics changes.
active=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:b=p.read_bytes()
 except OSError:continue
 if str(r/'code/').encode() in b:active.append(int(p.parent.name))
assert not active,active
stop=json.loads((r/'runtime/SCIENTIFIC_STOP.json').read_text())
assert stop['status']=='ENGINEERING_STORAGE_PAUSE_NOT_SCIENTIFIC_QUALITY_VERDICT' and not stop['quality_based_retry'] and not stop['science_contract_change']
assert not (r/'source').is_symlink() and not (r/'design/RECORDING_TIER.json').exists()
archive=r/'source-NFS-engineering-archive';assert not archive.exists()
source=r/'source';original=inventory(source);runtime=r/'runtime';noise=runtime/'noise';budget=runtime/'quality_budget.jsonl'
assert budget.is_file() and not budget.is_symlink() and noise.is_dir() and not noise.is_symlink()
noise_inventory=inventory(noise);budget_hash=sha(budget);seal_hash=sha(r/'design/CODE_SEAL.json')
required=sum(v.get('size',0) for v in original.values())+sum(v.get('size',0) for v in noise_inventory.values())+budget.stat().st_size
assert shutil.disk_usage('/var/cache').free>required+10*(1<<30),'Need at least10GiB working reserve after copy'
local.mkdir(parents=True,exist_ok=True);dest=local/'source';local_noise=local/'runtime/noise';local_budget=local/'runtime/quality_budget.jsonl'
assert not dest.exists() and not local_noise.exists() and not local_budget.exists()
local_budget.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['rsync','-a',str(source)+'/',str(dest)+'/'],check=True)
subprocess.run(['rsync','-a',str(noise)+'/',str(local_noise)+'/'],check=True)
local_budget.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(budget,local_budget)
assert inventory(dest)==original and inventory(local_noise)==noise_inventory and sha(local_budget)==budget_hash
# Preserve durable originals with exact hashes; install same logical path aliases.
source.rename(archive);source.symlink_to(dest,target_is_directory=True)
old_noise=runtime/'noise-NFS-engineering-archive';assert not old_noise.exists();noise.rename(old_noise);noise.symlink_to(local_noise,target_is_directory=True)
old_budget=runtime/'quality_budget-NFS-engineering-archive.jsonl';assert not old_budget.exists();budget.rename(old_budget);budget.symlink_to(local_budget)
alias=dest/'evidence/quality_budget.jsonl';assert not alias.exists();alias.symlink_to(local_budget)
assert inventory(archive)==original and sha(budget)==budget_hash and sha(alias)==budget_hash
tier=dict(status='PASS_STOPPED_PI0286_SAME_CONDITION_CUMULATIVE_LEDGER_EXACT_STORAGE_READBACK',record_root=str(dest),large_output_roots={},reason='OWNER_ENGINEERING_STORAGE_PAUSE_AFTER_NFS_LATENCY_MEASUREMENT',original_record_root=str(archive),original_source_inventory_sha256=hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest(),cumulative_quality_budget_sha256=budget_hash,noise_inventory_sha256=hashlib.sha256(json.dumps(noise_inventory,sort_keys=True).encode()).hexdigest(),code_seal_sha256=seal_hash,algorithm_or_frame_or_fsync_changes=0,quality_retry=0,model_queries=0,physics=0)
(r/'design/RECORDING_TIER.json').write_text(json.dumps(tier,indent=2)+chr(10))
assert sha(r/'design/CODE_SEAL.json')==seal_hash
out=dict(status='PASS_EXACT_STOPPED_SOURCE_NOISE_AND_GLOBAL_QUALITY_LEDGER_STORAGE_MIGRATION',tier=tier,source_files=len(original),source_bytes=sum(v.get('size',0) for v in original.values()),quality_ledger_aliases_same_inode=budget.stat().st_ino==alias.stat().st_ino,source_inventory=original,noise_inventory=noise_inventory,free_after_bytes=shutil.disk_usage(local).free,original_NFS_source_retained=True,model_queries=0,physics=0,epoch=time.time())
(r/'evidence/STORAGE_MIGRATION.json').write_text(json.dumps(out,indent=2)+chr(10));print(json.dumps(out))
'''))
    Path('reports/PI0286/preparation/STORAGE_MIGRATION.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    action=sys.argv[1];v=retire() if action=='retire' else migrate() if action=='migrate' else durable() if action=='durable' else None
    assert v is not None
    print(json.dumps({k:x for k,x in v.items() if k not in ('source_inventory','noise_inventory','files')}))
