"""Retire exactly three HF-readback-closed C0 payload files; retain evidence."""
import json,sys
from pathlib import Path
sys.path.insert(0,'scripts/pi0276_execution')
from coordinator import remote
core=json.loads(remote('core',r'''
import json
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
c=json.loads((r/'runtime/C0_REPLAY1024-1024_endpoint_EVALUATION_CLOSED.json').read_text())
assert c['status']=='ENDPOINT_CLOSED_VALID_TRAIN_FAIL'
p=json.loads((r/'qualification/C0_REPLAY1024-1024_ROUTE.json').read_text());assert p['TRAIN']['T1']==2
assert not (r/'qualification/CONFIRM_C_CANDIDATE.json').exists()
print(json.dumps(dict(closed=c,route=p)))
'''))
script=r'''
import hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286');core=CORE
cp=r/'learning/C0_REPLAY1024/main/checkpoint-1024';assert cp.resolve().is_relative_to(r.resolve())
q=subprocess.check_output(['/cm/shared/apps/pbspro/current/bin/qstat','-xf','656152.hopper-m-02'],text=True)
assert 'job_state = F' in q and 'Exit_status = 0' in q
assert json.loads((r/'archive/c0_full_trainer/WORKER_DONE.json').read_text())['returncode']==0
readback=r/'evidence/C0_REPLAY1024/checkpoint-1024.full_readback.json'
assert hashlib.sha256(readback.read_bytes()).hexdigest()=='daba4bae104b639a672a6b15721cd76f0e00aa546cde3b8dcd5f6ad25eb40169'
v=json.loads(readback.read_text());assert v['status']=='PASS_FULL_FILE_READBACK' and v['complete_trainer']
assert v['revision']=='b0b8c65de8d5bbdf0946ccdda9ecb23348cf16c9'
a=json.loads((r/'evidence/C0_REPLAY1024-1024_TRAINING_CPU.json').read_text())
assert a['entries']==v['files'] and len(v['files'])==14
h=hashlib.sha256()
for e in v['files']:h.update(f"PI0286/C0_REPLAY1024/checkpoint-1024/{e['path']}\0{e['bytes']}\0{e['sha256']}\n".encode())
assert h.hexdigest()==v['tree_sha256']
card=json.loads((r/'design/C1_CORRECT1024_CARD.json').read_text());assert Path(card['parent']).resolve()!=cp.resolve()
names=['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','optimizer.pt'];entries={e['path']:e for e in v['files']}
items=[]
for name in names:
 p=cp/name;s=p.lstat();assert not p.is_symlink() and p.resolve().parent==cp.resolve() and s.st_uid==os.getuid() and s.st_nlink==1 and s.st_size==entries[name]['bytes']
 items.append(dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,sha256=entries[name]['sha256']))
receipt=dict(status='PREVIEW_EXACT_HF_ONLY_RETIREMENT',epoch=time.time(),items=items,core_endpoint_closure=core,HF={k:v[k] for k in ['repo_id','prefix','revision','tree_sha256']},full_readback_sha256=hashlib.sha256(readback.read_bytes()).hexdigest(),classification='HF_ONLY_FULL_TRAINER_LOCAL_METADATA_ONLY_NOT_RESUME',retained='All small configs, hashes, receipts, training journals, screen and Core view retained; C1 original parent unchanged',selected='NONE',deployment='NONE')
path=r/'evidence/storage/C0_FULL_HF_ONLY_RETIREMENT.json';assert not path.exists();path.write_text(json.dumps(receipt,indent=2)+'\n')
before=os.statvfs(cp)
for item in items:
 p=Path(item['path']);s=p.lstat();assert (s.st_dev,s.st_ino,s.st_size,s.st_nlink)==(item['device'],item['inode'],item['bytes'],1);p.unlink()
after=os.statvfs(cp);receipt.update(status='DELETED_AND_VERIFIED_PINNED_HF_ONLY',deleted_logical_bytes=sum(x['bytes'] for x in items),deleted_allocated_bytes=sum(x['allocated_bytes'] for x in items),shared_filesystem_free_change_bytes=after.f_bavail*after.f_frsize-before.f_bavail*before.f_frsize,quota_headroom='UNKNOWN_NO_NEW_PERSONAL_QUOTA_RECEIPT',all_exact_paths_absent=all(not Path(x['path']).exists() for x in items))
path.write_text(json.dumps(receipt,indent=2)+'\n');(cp/'LOCAL_PAYLOAD_RETIRED_NOT_RESUME.json').write_text(json.dumps(dict(status=receipt['status'],HF=receipt['HF'],receipt=str(path)))+'\n')
print(json.dumps(receipt))
'''.replace('CORE',repr(core))
value=remote('hop',script)
p=Path('reports/PI0286/execution/C0_FULL_HF_ONLY_RETIREMENT.json');p.write_text(value,encoding='utf-8')
print(value)
