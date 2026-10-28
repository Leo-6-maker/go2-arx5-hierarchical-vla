"""Stage only the isolated CPU explanation tool and return its small receipt."""
import base64
import json
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parents[2]/'scripts/pi0276_execution'))
from coordinator import remote
payload=base64.b64encode((here/'preclip_failed15_cpu.py').read_bytes()).decode()
script=f'''import base64,json,os,subprocess
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
p=r/'correction/tools/preclip_failed15_cpu.py'
b=base64.b64decode({payload!r})
p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():assert p.read_bytes()==b
else:p.write_bytes(b)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
result=subprocess.run(['/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python','-u',str(p)],env=env,text=True,capture_output=True,timeout=240)
print(result.stdout)
if result.returncode:raise RuntimeError(result.stderr[-8000:])
'''
print(remote('core',script))
