"""Same native loader/query, frozen checkpoint and original physical-noise namespace."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
from budget import publication,process
from native_worker import query

def run(root,point):
    root=Path(root);publication(root);card=json.loads((root/'design/RUNTIME_CARD.json').read_text());sys.path.append(card['parent_code'])
    for path,h in card['dependencies'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    from inference import load,array_hash
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=process(root,'start')
    try:
        model,codec,binding=load(root,point)
        print(json.dumps(dict(ready=True,point=point)),flush=True)
        for line in sys.stdin:
            request=json.loads(line);request['K']=4
            if 'noise' not in request:
                key=json.dumps(dict(condition=request['condition'],query_index=request['query_index']),sort_keys=True)
                path=root/'source/noise'/(hashlib.sha256(key.encode()).hexdigest()+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():
                    prior=Path(card['parent_root'])/'source/noise'/path.name
                    if prior.exists():packet=np.load(prior);assert packet.shape==(1,40,132) and packet.dtype==np.uint16
                    else:
                        seed=int.from_bytes(hashlib.sha256(('PI0282-28203-'+key).encode()).digest()[:8],'little')%(2**63-1)
                        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                            packet=torch.randn((1,40,132),dtype=torch.bfloat16,device='cuda:0').cpu().view(torch.uint16).numpy()
                    temporary=path.with_suffix('.tmp')
                    with temporary.open('wb') as f:np.save(f,packet);f.flush();os.fsync(f.fileno())
                    temporary.replace(path)
                request['noise']=str(path)
            print(json.dumps(query(root,model,codec,request)),flush=True)
        for n,p in model.named_parameters():assert array_hash(p.detach().cpu().contiguous().view(torch.uint8).numpy())==binding['effective_bf16_tensor_sha256'][n],n
    finally:process(root,'end',started)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--point',required=True);a=p.parse_args();run(a.root,a.point)
