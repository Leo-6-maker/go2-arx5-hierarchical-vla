"""Actual fixed evaluation packets and inherited language assignments; CPU only."""
import hashlib
import json
import sys
import numpy as np
import torch
from pi0266_common import ROOT,REF,read,sha,write
sys.path.insert(0,str(REF/'src'))
from pi0258_kinematic import instructions


def main():
    target=ROOT/'design/EVAL_NOISE_CARD.json';assert not target.exists(),'actual packets already frozen: reuse them'
    assert not (ROOT/'evidence/TRAINING_STARTED.json').exists()
    torch.set_num_threads(1)
    card=read(ROOT/'design/SCENE_CARD.json');rows=[];texts=[]
    for family in card['families']:
        if family['split'] not in ('FEEDBACK_TRAIN','NEW_SIM_EVAL'):continue
        for task in ('M1','M3'):
            for command in (0,1):
                variant=(family['rank']+command)%2
                text=instructions(task,command,family['factors']['colors'])[variant]
                texts.append(dict(family=family['family_id'],split=family['split'],task=task,command=command,template='T'+str(variant),instruction=text))
        if family['split']!='NEW_SIM_EVAL':continue
        for task in ('M1','M3'):
            for name in ('N0','N1'):
                for index in range(100):
                    key=f"PI0266:EVAL:26641:{family['family_id']}:{task}:{name}:{index}"
                    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')
                    generator=torch.Generator(device='cpu').manual_seed(seed)
                    packet=torch.randn((1,40,132),dtype=torch.float32,generator=generator)
                    bits=packet.to(torch.bfloat16).view(torch.uint16)
                    rel=f"noise/{family['family_id']}-{task}-{name}-{index:03d}.npz";path=ROOT/rel
                    path.parent.mkdir(exist_ok=True);assert not path.exists()
                    np.savez_compressed(path,epsilon_fp32=packet.numpy(),epsilon_bf16_uint16=bits.numpy())
                    assert torch.equal(packet,torch.randn((1,40,132),dtype=torch.float32,generator=torch.Generator(device='cpu').manual_seed(seed)))
                    rows.append(dict(key=key,seed=seed,file=rel,sha256=sha(path),shape=[1,40,132],command_and_model_independent=True))
    assert len(rows)==3200 and len({r['key'] for r in rows})==3200
    write(ROOT/'design/LANGUAGE_CARD.json',dict(status='FIXED_INHERITED_T0_T1_M3_NEW_FAMILY_LANGUAGE_ASSIGNMENTS',rows=texts,
        source_sha256=sha(ROOT/'code/pi0258_kinematic.py'),assignment='Inherited (family rank + command)%2; TRAIN M1 is later replaced only by the shared T0-T7 eight-pass schedule.',
        hidden_phase_tokens=False,new_EVAL_images_or_trajectories_read=False))
    write(target,dict(status='PASS_ACTUAL_3200_EVAL_RANDOM_PACKETS_SEALED',rows=rows,source_SCENE_CARD_sha256=sha(ROOT/'design/SCENE_CARD.json'),
        generator='Torch CPU FP32 standard normal plus actual BF16 cast bits',namespace='PI0266:EVAL:26641:<family>:<task>:<N0|N1>:<replan-index>',
        new_EVAL_images_or_trajectories_read=False,GR00T_queries=0,optimizer_updates=0))
    print(json.dumps(dict(status='PASS_FIXED_ACTUAL_3200_EVAL_PACKETS',packets=len(rows),metadata_only_families=True,model_queries=0)),flush=True)


if __name__=='__main__':main()
