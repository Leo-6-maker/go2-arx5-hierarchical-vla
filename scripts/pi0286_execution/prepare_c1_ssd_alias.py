"""Owner-authorized fixed C1 model-only SSD aliases, only at closed C0 boundary."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pi0276_execution'))
from coordinator import remote

CODE = r'''
import hashlib,json,os,time
from pathlib import Path
R=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
S=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0286')
P='C1_CORRECT1024-1024'
closed=R/'runtime/C0_REPLAY1024-1024_endpoint_EVALUATION_CLOSED.json'
assert closed.is_file(),'C0 finite endpoint must be closed before aliases'
closure=json.loads(closed.read_text())
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=(p/'cmdline').read_bytes().split(b'\0')
  if any(str(R/'code'/n).encode() in argv for n in ('evaluate.py','policy.py','model_worker.py','transport.py')):active.append(int(p.name))
 except OSError:pass
assert not active,('ACTIVE_OWN_ENDPOINT',active)
seal=hashlib.sha256((R/'design/CODE_SEAL.json').read_bytes()).hexdigest()
view=R.parent/'PI0283/code/model_view.py'
texts={name:(R/'code'/name).read_text() for name in ('transport.py','model_worker.py','policy.py')}
assert 'folder.replace(final)' in texts['transport.py']
assert "output.mkdir(parents=True,exist_ok=True)" in view.read_text()
assert "sha(Path(binding['model_root'])/name)==h" in texts['model_worker.py']
assert 'source.resolve().is_relative_to' in texts['model_worker.py']
assert not (R/'transfers'/P).exists() and not (R/'transfers'/P).is_symlink()
aliases=[(R/'transfers'/(P+'.partial'),S/'transfers'/P),(R/'models'/P,S/'models'/P)]
receipt=R/'evidence/evaluation/C1_SSD_MODEL_ONLY_ALIAS.json'
if receipt.exists():
 old=json.loads(receipt.read_text())
 for logical,target in aliases:assert logical.is_symlink() and logical.resolve()==target.resolve()
 print(json.dumps(old))
else:
 for logical,target in aliases:
  assert not logical.exists() and not logical.is_symlink(),str(logical)
  assert not target.exists() and not target.is_symlink(),str(target)
  assert target.is_relative_to(S) and logical.is_relative_to(R)
 free=os.statvfs(S).f_bavail*os.statvfs(S).f_frsize
 estimated_peak=5670085448+6288169448
 assert free>estimated_peak+4*1024**3,('ACTUAL_SSD_CAPACITY',free)
 for logical,target in aliases:
  target.mkdir(parents=True);logical.parent.mkdir(parents=True,exist_ok=True);logical.symlink_to(target,target_is_directory=True)
  assert logical.resolve()==target.resolve() and not list(logical.iterdir())
 assert hashlib.sha256((R/'design/CODE_SEAL.json').read_bytes()).hexdigest()==seal
 result=dict(status='PASS_CLOSED_BOUNDARY_EMPTY_FIXED_C1_MODEL_ONLY_SSD_ALIASES',point=P,epoch=time.time(),actual_free_before_B=free,actual_free_after_B=os.statvfs(S).f_bavail*os.statvfs(S).f_frsize,estimated_delta_plus_BF16_peak_B=estimated_peak,full_master_disk_temporary=False,paths=[dict(logical=str(a),actual=str(b),empty=True) for a,b in aliases],C0_closure_sha256=hashlib.sha256(closed.read_bytes()).hexdigest(),C0_closure_status=closure.get('status'),Core_seal_sha256=seal,code_sha256={n:hashlib.sha256(t.encode()).hexdigest() for n,t in texts.items()},model_view_sha256=hashlib.sha256(view.read_bytes()).hexdigest(),same_logical_model_point=True,actual_C1_transfer=False,model_queries=0,physics=0,optimizer_transfer=False,global_100GiB_cap_owner_retained=True)
 receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''

if __name__ == '__main__':
    result=json.loads(remote('core',CODE))
    out=Path(__file__).resolve().parents[2]/'reports/PI0286/preparation/C1_SSD_MODEL_ONLY_ALIAS.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
