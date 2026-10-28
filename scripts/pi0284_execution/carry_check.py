"""One charged native prediction per real canary, or the retained CPU comparator."""
import argparse,json,sys
from pathlib import Path

def run(root,arm,phase):
    root=Path(root);card=json.loads((root/'design/RUNTIME_CARD.json').read_text())
    sys.path.append(card['parent_code'])
    if phase=='compare':
        from canary import compare
        compare(root,arm);return
    from budget import process
    from inference import load,array_hash
    from native_worker import query
    from score import gate
    import numpy as np
    import torch
    point=arm+'-CANARY-'+phase
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=process(root,'start')
    try:
        model,codec,binding=load(root,point)
        folder=root/'learning'/arm/phase
        request=dict(K=4,input=str(root/'data/CANARY_INPUT.npz'),noise=str(root/'data/CANARY_NOISE.npy'),output=str(folder/'CANARY_OUTPUT.npz'))
        answer=query(root,model,codec,request)
        with np.load(request['output']) as v:pred=v['decoded'].copy();state=v['actual_state'].copy()
        result={k:v for k,v in answer.items() if k!='output'};result['gate']=gate(pred,state)
        (folder/'CANARY_PREDICTIONS.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
        for n,p in model.named_parameters():assert array_hash(p.detach().cpu().contiguous().view(torch.uint8).numpy())==binding['effective_bf16_tensor_sha256'][n],n
    finally:process(root,'end',started)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',choices=('CTRL2048','LR5X2048'),required=True);p.add_argument('--phase',choices=('continuous','split','compare'),required=True);a=p.parse_args();run(a.root,a.arm,a.phase)
