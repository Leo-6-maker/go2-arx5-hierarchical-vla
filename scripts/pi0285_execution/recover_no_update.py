"""Recover the confirmed terminal zero-update input-audit attempt only."""
import base64,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def recover():
    files={n:base64.b64encode((Path(__file__).parent/n).read_bytes()).decode() for n in ('train.py','submit.py')}
    program=r'''import base64,fcntl,hashlib,json,re,subprocess,shlex
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0285');job='655259.hopper-m-02'
with (r/'runtime/TRAINING_ALLOCATION.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt=r/'runtime'/('NO_UPDATE_RECOVERY_'+job+'.json')
 if receipt.exists():print(receipt.read_text());raise SystemExit(0)
 q=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qstat','-fx',job],capture_output=True,text=True)
 assert q.returncode==0 and re.search(r'job_state\s*=\s*F\b',q.stdout) and re.search(r'Exit_status\s*=\s*1\b',q.stdout)
 closed=json.loads((r/'runtime'/('ALLOCATION_CLOSED_'+job+'.json')).read_text());assert closed['status']=='ENGINEERING_ATTEMPT_RETAINED'
 completed=r/'learning/completed_updates.jsonl';assert not completed.exists() or not completed.read_text().strip()
 folder=r/'learning/PLACE4096/continuous';assert not list(folder.glob('checkpoint-*'))
 forwards=[json.loads(x) for x in (folder/'native_training_forwards.jsonl').read_text().splitlines()];assert len(forwards)==1
 assert not (folder/'consumption.jsonl').read_text().strip()
 log=r/'runtime/PLACE4096-continuous-0-8.log';assert 'Object of type BatchFeature is not JSON serializable' in log.read_text()
 archive=r/'runtime/recovery'/job;archive.mkdir(parents=True,exist_ok=False)
 (archive/'qstat.txt').write_text(q.stdout)
 paths=[folder,r/'runtime/PLACE4096-continuous-0-8.start.json',r/'runtime/PLACE4096-continuous-0-8.done.json',log,r/'design/CODE_SEAL.json']
 inventory=[]
 for p in paths:
  assert p.exists() and p.resolve().is_relative_to(r.resolve()) and not p.is_symlink()
  if p.is_dir():members=[x for x in p.rglob('*') if x.is_file()]
  else:members=[p]
  inventory.extend(dict(path=str(x.relative_to(r)),bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in members)
  if p==r/'design/CODE_SEAL.json':(archive/'CODE_SEAL.json').write_bytes(p.read_bytes())
  else:p.rename(archive/p.name)
 seal=json.loads((r/'design/CODE_SEAL.json').read_text())
 for n,b in FILES.items():
  p=r/'code'/n;p.write_bytes(base64.b64decode(b));seal['files'][n]=hashlib.sha256(p.read_bytes()).hexdigest()
 seal['engineering_recovery']=dict(PBS_JOBID=job,reason='MAPPING_COMPATIBLE_PASSIVE_INPUT_AUDIT',native_dependencies_unchanged=True)
 (r/'design/CODE_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
 check="from contract import guard;guard();from data import Dataset,collate;from train import audit_tensor_tree;from transformers.feature_extraction_utils import BatchFeature;import torch,json;from pathlib import Path;r=Path("+repr(str(r))+" );d=Dataset(r/'data',r/'data/PLACE4096.jsonl');b=collate([d[i] for i in range(4)],r/'data');b.pop('_pi0282_visits');assert isinstance(b,BatchFeature);v=audit_tensor_tree(b);json.dumps(v);t=torch.tensor([1.,2.]);a=audit_tensor_tree({'a':[BatchFeature({'x':t})]});from learning import tensor_tree;assert a['a'][0]['x']==tensor_tree(t);assert torch.equal(t,torch.tensor([1.,2.]));print('PASS_ACTUAL_FOUR_INPUT_BATCH_MAPPING_AND_NESTED_TENSOR_HASHES_ZERO_MODEL_QUERIES')"
 shell='source '+str(r/'code/environment.sh')+'; export CUDA_VISIBLE_DEVICES=""; "$venv/bin/python" -c '+shlex.quote(check)
 result=subprocess.run(['/bin/bash','-lc',shell],capture_output=True,text=True)
 (archive/'CPU_check.log').write_text(result.stdout+result.stderr)
 assert result.returncode==0,(result.stdout,result.stderr[-3000:])
 value=dict(status='PASS_TERMINAL_ZERO_UPDATES_AUDIT_REPAIR',PBS_JOBID=job,completed_optimizer_updates=0,raw_native_forward_backward_microbatches=1,raw_chunks=4,rolled_back_without_optimizer=True,preserved_files=inventory,CPU_check=result.stdout,model_optimizer_physics_queries_added_by_repair=0,code_files={n:seal['files'][n] for n in FILES},scientific_contract_changed=False,selected='NONE',deployment='NONE')
 receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''.replace('FILES',repr(files))
    result=json.loads(remote('hop',program));p=Path('reports/PI0285/execution/NO_UPDATE_RECOVERY_655259.json');p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':recover()
