"""Original DROID policy methods/processor/history, with the already bound child model."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from budget import publication,charge,process,time_allowed
from prepare_core import sha,write

def original_policy(model,base,tag):
    from transformers import AutoProcessor
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.policy import BasePolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    from native_model import local_processor
    # Reuse stock methods with the verified resident child; do not load a second model.
    policy=Gr00tPolicy.__new__(Gr00tPolicy);BasePolicy.__init__(policy,strict=True)
    policy.model=model
    with local_processor():policy.processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    policy.processor.eval();policy.embodiment_tag=EmbodimentTag.resolve(tag)
    policy.modality_configs={k:v for k,v in policy.processor.get_modality_configs()[tag].items() if k!='rl_info'}
    policy.collate_fn=policy.processor.collator;policy.language_key=policy.modality_configs['language'].modality_keys[0]
    assert policy.modality_configs['video'].delta_indices==[-15,0]
    return policy

def run(root,point):
    root=Path(root);publication(root);runtime=json.loads((root/'design/RUNTIME_CARD.json').read_text());sys.path.append(runtime['parent_code'])
    for path,h in runtime['dependencies'].items():assert sha(path)==h,path
    from inference import load,array_hash
    card=json.loads((root/'design/RETENTION32.json').read_text());folder=root/'retention'/point;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'RESULT.json').exists(),'Reuse closed retention panel'
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=process(root,'start')
    try:
        model,_,binding=load(root,point);policy=original_policy(model,card['original_model_root'],card['source_tag']);records=[]
        for row in card['rows']:
            output=folder/(row['id']+'.npz')
            if output.exists():records.append(dict(id=row['id'],path=str(output),sha256=sha(output)));continue
            time_allowed(root,2);assert sha(row['input_path'])==row['input_sha256']
            with np.load(row['input_path']) as f:
                state=f['state'].copy();observation=dict(video={k:f[k][None].copy() for k in card['video_key_binding']},state={k:state[a:b][None,None].copy() for k,(a,b) in card['state_slices'].items()},language={policy.language_key:[[row['language']]]})
            packet=np.load(row['noise_path']);assert sha(row['noise_path'])==row['noise_sha256'] and packet.shape==(1,40,132) and packet.dtype==np.uint16
            noise=torch.from_numpy(packet.copy()).view(torch.bfloat16).cuda();raw=[];heads=[];draws=[]
            charge(root,'query',str(output))
            def head(*args):charge(root,'head',str(output));heads.append(1)
            hook=model.action_head.model.register_forward_pre_hook(head)
            native=model.get_action;randn=torch.randn
            def capture(*a,**kw):
                assert 'action' not in kw.get('inputs',{})
                v=native(*a,**kw);raw.append(v['action_pred'].detach().float().cpu().numpy());return v
            def fixed(*a,**kw):
                assert tuple(kw.get('size',a[0] if a else ()))==tuple(noise.shape) and kw['dtype']==noise.dtype and torch.device(kw['device'])==noise.device
                draws.append(1);return noise.clone()
            model.get_action=capture;torch.randn=fixed
            try:
                with torch.inference_mode():decoded,_=policy.get_action(observation)
                torch.cuda.synchronize()
            finally:model.get_action=native;torch.randn=randn;hook.remove()
            assert len(raw)==len(draws)==1 and len(heads)==4
            np.savez_compressed(output,raw=raw[0],noise_uint16=packet,input_sha256=np.asarray(row['input_sha256']),**{'decoded_'+k:v for k,v in decoded.items()})
            records.append(dict(id=row['id'],path=str(output),sha256=sha(output)));write(folder/'PROGRESS.json',dict(records=records))
        for n,p in model.named_parameters():assert array_hash(p.detach().cpu().contiguous().view(torch.uint8).numpy())==binding['effective_bf16_tensor_sha256'][n],n
        write(folder/'RESULT.json',dict(status='COMPLETE_ORIGINAL_DROID32_DESCRIPTIVE_RETENTION_NOT_TASK_QUALIFICATION',point=point,records=records,queries=32,heads=128,processor_identity=card['processor_identity'],selected='NONE',deployment='NONE'))
    finally:process(root,'end',started)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--point',choices=('CTRL2048','LR5X2048'),required=True);a=p.parse_args();run(a.root,a.point)
