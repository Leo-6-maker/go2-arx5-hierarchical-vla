"""Freeze two label-independent global S4 noise packets before model scores."""
import hashlib
import json

import numpy as np
import torch

from pi0260_r2 import ROOT,sha


def main():
    out=ROOT/'R3/noise';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,seed in (('B0',26031),('B1',26032)):
        f32=torch.randn((1,40,132),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
        bits=f32.to(torch.bfloat16).view(torch.uint16).numpy()
        path=out/(name+'.npz');assert not path.exists()
        np.savez_compressed(path,float32=f32.numpy(),bfloat16_bits=bits)
        rows.append(dict(name=name,seed=seed,shape=[1,40,132],generator='torch CPU float32 standard normal',
                         input_float32_sha256=hashlib.sha256(f32.numpy().tobytes()).hexdigest(),
                         cast_bfloat16_bits_sha256=hashlib.sha256(bits.tobytes()).hexdigest(),
                         file_sha256=sha(path),shared_across='all conditions, labels and models'))
    p=out/'manifest.json';assert not p.exists()
    p.write_text(json.dumps(dict(status='R3_GLOBAL_B0_B1_FROZEN',rows=rows,model_queries=0),indent=2)+'\n')
    print(json.dumps(dict(status='R3_GLOBAL_B0_B1_FROZEN',manifest_sha256=sha(p))))


if __name__=='__main__':main()
