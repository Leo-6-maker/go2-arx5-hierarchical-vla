"""Read-only exact-path inventory for the pristine model and delivered WBC."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote

if __name__=='__main__':
    print(remote('core','''import json,subprocess
from pathlib import Path
paths=['/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/assets/GR00T-N1.7-3B','/home/liuyu-nfs/Projects/isaac-gr00t','/home/liuyu-nfs/Projects/GR00T-WholeBodyControl','/home/liuyu-nfs/stage13c_staging/native_compat']
for name in paths:
 p=Path(name);print(json.dumps(dict(path=name,exists=p.exists(),resolved=str(p.resolve()),entries=[dict(name=x.name,bytes=x.stat().st_size,link=x.is_symlink()) for x in sorted(p.iterdir())] if p.is_dir() else [])))
 if (p/'.git').exists():
  for args in [['git','-C',name,'rev-parse','HEAD'],['git','-C',name,'status','--short']]:
   r=subprocess.run(args,capture_output=True,text=True);print(json.dumps(dict(command=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
base=Path(paths[0])
for n in ['config.json','processor_config.json','model.safetensors.index.json']:
 p=base/n
 if p.is_file():print('FILE '+str(p)+'\\n'+p.read_text())
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
'''))
