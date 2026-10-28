"""Fixed preparation paths and zero-query CPU boundary."""
import os
import sys
from pathlib import Path
import json
import hashlib
import csv

os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    NUMEXPR_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1')
sys.dont_write_bytecode=True
if hasattr(os,'sched_getaffinity'):
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5')
PREP=BASE/'PI0268/preparation';OUT=PREP/'output';SOURCE=PREP/'source_metadata'
R1=BASE/'PI0267/R1';RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
DATA=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
sys.path[:0]=[str(DATA/'qa'),str(DATA/'R3'),'/home/liuyu-nfs/Projects/isaac-gr00t']


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True,default=lambda v:v.tolist())+'\n')
def csv_write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]


def restrict():
    import resource
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30))
    def guard(event,args):
        if event in ('socket.connect','subprocess.Popen','os.system'):
            raise RuntimeError('PI0268_NO_NETWORK_OR_SUBPROCESS_IN_ANALYSIS')
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=os.fsdecode(args[0])
            if p.endswith(('.safetensors','.pt','.pth','.bin')):
                raise RuntimeError('PI0268_NO_WEIGHT_OR_OPTIMIZER_LOAD')
    sys.addaudithook(guard)


ZERO=dict(model_queries=0,gradient_queries=0,optimizer_updates=0,renderer_calls=0,physics_steps=0,WBC_calls=0,PBS_submissions=0)
