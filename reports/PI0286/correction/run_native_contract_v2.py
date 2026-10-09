"""Owner-authorized one-line Core repair, archived seal, versioned metadata derivation."""
import base64,json,sys
from pathlib import Path
here=Path(__file__).resolve().parent;repo=here.parents[2]
sys.path.insert(0,str(repo/'scripts/pi0276_execution'))
from coordinator import remote
payloads={name:base64.b64encode((here/name).read_bytes()).decode() for name in ['derive_native_contract_v2.py','native_contract_boundary.py']}
expected=(repo/'scripts/pi0286_execution/recovery.py').read_text()
script=f'''import base64,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
locks=[]
for name in ('SOURCE_GPU0.lock','EVALUATION_OWNER.lock'):
 f=(r/'runtime'/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
try:
 active=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   text=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
   if any(str(r/'code'/name) in text for name in ('evaluate.py','recovery.py','model_worker.py','policy.py','teacher.py')):active.append(dict(pid=p.name,cmdline=text))
  except (FileNotFoundError,PermissionError):pass
 assert not active,active
 path=r/'code/recovery.py';sealpath=r/'design/CODE_SEAL.json'
 old=path.read_bytes();oldhash=sha(path);assert oldhash=='62c6cd64e9a0ce6980243881f4948122d8ba3170fba52e535dbc0e89a546128d',oldhash
 sealbytes=sealpath.read_bytes();seal=json.loads(sealbytes);assert seal['files']['recovery.py']==oldhash
 needle=b"valid=valid and err['native_preclip_out_of_range_count']==0"
 lines=old.splitlines(keepends=True);assert sum(x.strip()==needle for x in lines)==1
 new=b''.join(x for x in lines if x.strip()!=needle)
 assert new.decode().replace('\\r\\n','\\n')=={expected!r}
 archive=r/'correction/implementation_erratum';archive.mkdir(exist_ok=True)
 for name,b in [('recovery_PRE_NATIVE_CONTRACT_V2.py',old),('CODE_SEAL_PRE_NATIVE_CONTRACT_V2.json',sealbytes)]:
  q=archive/name
  if q.exists():assert q.read_bytes()==b
  else:q.write_bytes(b)
 tmp=path.with_suffix('.native_contract_v2.tmp');tmp.write_bytes(new);tmp.replace(path)
 newhash=sha(path);seal['files']['recovery.py']=newhash
 tmp=sealpath.with_suffix('.native_contract_v2.tmp');tmp.write_text(json.dumps(seal,indent=2)+'\\n');tmp.replace(sealpath)
 receipt=dict(status='OWNER_AUTHORIZED_IMPLEMENTATION_ADDED_DIAGNOSTIC_GATE_REMOVAL_SINGLE_FILE',old_recovery_sha256=oldhash,new_recovery_sha256=newhash,
 old_code_seal_sha256=hashlib.sha256(sealbytes).hexdigest(),new_code_seal_sha256=sha(sealpath),
 removed_exact_condition=needle.decode(),removed_lines=1,other_seal_members_changed=False,
 owner_authority='root: original release contract correction; no new scientific branch',active_source_processes=active,
 CPU_previous_tool_closed=True,model_calls=0,physics_calls=0,optimizer_calls=0,epoch=time.time())
 q=archive/'SINGLE_FILE_AND_SEAL_UPDATE.json';assert not q.exists();q.write_text(json.dumps(receipt,indent=2)+'\\n')
 for name,encoded in {payloads!r}.items():
  q=r/'correction/tools'/name;b=base64.b64decode(encoded)
  if q.exists():assert q.read_bytes()==b
  else:q.write_bytes(b)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
 result=subprocess.run(['/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python','-u',str(r/'correction/tools/derive_native_contract_v2.py')],env=env,text=True,capture_output=True,timeout=240)
 print(json.dumps(receipt));print(result.stdout)
 if result.returncode:raise RuntimeError(result.stderr[-8000:])
finally:
 for f in locks:fcntl.flock(f,fcntl.LOCK_UN);f.close()
'''
print(remote('core',script))
