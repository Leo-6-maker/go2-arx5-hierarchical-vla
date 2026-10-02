"""Finite stdin inference worker. No policy query before both final Trainer seals."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
DATA=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(BASE/'PI0250/runtime/hf_home'),
                  NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false')
sys.path[:0]=[str(DATA/'R3'),str(DATA/'runtime'),str(DATA/'qa'),str(BASE/'PI0250/runtime/reference-317a/scripts'),
              str(BASE/'PI0250/runtime/reference-317a/src'),'/home/liuyu-nfs/Projects/isaac-gr00t',str(BASE/'PI0266/code')]
from pi0260_r3_inputs import build,batch,identity
from pi0255_motion3 import sha,write


def read(p):return json.loads(Path(p).read_text())
def append(p,value):
    with p.open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def count(p):return sum(1 for _ in p.open()) if p.exists() else 0


def admitted():
    audit=read(R1/'evidence/MAIN_TRAINING_AUDIT.json')
    assert audit['status']=='PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY' and audit['optimizer_updates']=={'N':4096,'R':4096}
    assert len(audit['complete_checkpoints'])==4
    assert {(r['arm'],r['step']) for r in audit['complete_checkpoints']}=={(a,s) for a in ('N','R') for s in (2048,4096)}
    for checkpoint in audit['complete_checkpoints']:
        assert checkpoint['complete_trainer'] and len(checkpoint['entries'])>=16
        assert {'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','model.safetensors.index.json'}<=set(r['path'] for r in checkpoint['entries'])


def main(point):
    admitted()
    import torch
    from transformers import AutoModel
    from gr00t.data.embodiment_tags import EmbodimentTag
    import pi0242_train_infer_bridge as bridge
    import stage21_same_scene_groot_inference as stage21
    from stage40_b0_model_control import _bind_local_cosmos_processor
    card=read(R1/'design/EVALUATION_CARD.json');binding=read(R1/'model_bindings'/(point+'.json'))
    assert binding['view_type']=='model-only' and binding['point']==point
    for name,digest in binding['model_files_sha256'].items():assert sha(Path(binding['view'])/name)==digest
    cosmos=BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot'
    assert _bind_local_cosmos_processor(Path('/home/liuyu-nfs/Projects/isaac-gr00t'),cosmos)['resolved_snapshot']==str(cosmos.resolve())
    proc=build(read(R1/'design/statistics.json'),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    assert sha(R1/'design/statistics.json')==card['statistics_sha256']
    model=AutoModel.from_pretrained(binding['view'],local_files_only=True,trust_remote_code=True).eval().to('cuda:0',dtype=torch.bfloat16)
    assert model.action_head.num_inference_timesteps==4
    attempts=R1/'evidence/model_attempts.jsonl';headsfile=R1/'evidence/model_heads.jsonl';heads=count(headsfile)
    print(json.dumps(dict(ready=True,point=point)),flush=True)
    for line in sys.stdin:
        request=json.loads(line);input_path=Path(request['input']);output=Path(request['output'])
        raw_root=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1').resolve()
        assert input_path.resolve().is_relative_to(raw_root) and output.resolve().is_relative_to(raw_root)
        with np.load(input_path,allow_pickle=False) as z:
            packet=dict(body_rgb=z['body_rgb'].copy(),wrist_rgb=z['wrist_rgb'].copy(),state_v1=z['state_v1'].copy(),language=request['language'])
        assert set(packet)=={'body_rgb','wrist_rgb','state_v1','language'}
        policy=batch(proc,packet)['inputs'];assert 'action' not in policy and 'action_mask' not in policy
        noise_path=Path(request['noise_file']);noise_name=request['noise_name']
        assert noise_path.resolve().is_relative_to((raw_root/'noise').resolve())
        registry=request.get('noise_registry','EVALUATION_CARD.json')
        assert registry in ('EVALUATION_CARD.json','FIXED_OFFLINE_CARD.json','SECONDARY_CARD.json')
        noise_card=card if registry=='EVALUATION_CARD.json' else read(R1/'design'/registry)
        entry=next(r for r in noise_card['noise'] if r['file']==str(noise_path));assert sha(noise_path)==entry['sha256']
        row=next(r for r in entry['keys'] if r['name']==noise_name)
        with np.load(noise_path) as z:noise=z[noise_name].copy()
        assert hashlib.sha256(noise.tobytes()).hexdigest()==row['noise_sha256']
        input_identity=identity(policy);journal=output.with_suffix('.receipt.json')
        if request.get('processor_contract')=='OLD':
            old_card=read(R1/'design/FIXED_OFFLINE_CARD.json')
            assert old_card['old_statistics_sha256']==card['statistics_sha256']
            assert input_identity==request['expected_condition_sha256']
        if output.exists():
            if journal.exists():prior=read(journal)
            else:
                with np.load(output,allow_pickle=False) as z:prior=json.loads(str(z['receipt_json']))
                prior['output_sha256']=sha(output);write(journal,prior)
            assert prior['point']==point and prior['binding']==binding and prior['input_sha256']==input_identity
            assert prior['noise_sha256']==row['noise_sha256'] and prior['output_sha256']==sha(output)
            print(json.dumps(dict(status='VALID_REUSED',output=str(output))),flush=True);continue
        ordinal=count(attempts)+1;assert ordinal<=24576
        append(attempts,dict(ordinal=ordinal,point=point,request=request,input_sha256=input_identity,noise_sha256=row['noise_sha256']))
        def hook(module,inputs):
            nonlocal heads
            heads+=1;assert heads<=98304
            append(headsfile,dict(ordinal=heads,attempt=ordinal,point=point))
        handle=model.action_head.model.register_forward_pre_hook(hook)
        try:
            with torch.inference_mode():stock=bridge._stock_query(model,policy,stage21,noise)
        finally:handle.remove()
        assert len(stock['trace'])==4 and np.array_equal(stock['noise'],noise)
        raw=np.asarray(stock['raw'],np.float32)
        # Keep invalid numerical predictions as real quality failures, without another model call.
        decode_failure=None
        if np.isfinite(raw).all():
            try:decoded=bridge._decode_physical(proc,EmbodimentTag.NEW_EMBODIMENT,raw,packet['state_v1'])
            except bridge.PI0242Blocker as error:
                assert str(error)=='P1 physical decode contract failed'
                decoded=np.full((1,40,10),np.nan,dtype=np.float32);decode_failure=str(error)
        else:decoded=np.full((1,40,10),np.nan,dtype=np.float32)
        trace_x=np.stack([x['x'] for x in stock['trace']]);trace_v=np.stack([x['v'] for x in stock['trace']]);trace_t=np.stack([x['t'] for x in stock['trace']])
        actual_noise=torch.from_numpy(noise).to(torch.bfloat16).float().numpy()
        assert np.array_equal(trace_x[0],actual_noise),'ACTUAL_INITIAL_NOISE_DTYPE_IDENTITY'
        assert decoded.shape==(1,40,10)
        finite=bool(np.isfinite(raw).all() and np.isfinite(decoded).all())
        receipt=dict(status='VALID',point=point,binding=binding,input_sha256=input_identity,noise_sha256=row['noise_sha256'],
            attempt=ordinal,heads=4,native_S4=True,model_latency_s=float(stock['elapsed_s']),initial_noise_dtype='torch.bfloat16',
            source_noise_dtype='numpy.float32',finite_output=finite,decode_failure=decode_failure,no_teacher_or_privileged_condition=True)
        output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_suffix('.partial.npz')
        np.savez_compressed(temp,raw=raw,decoded=decoded,state=packet['state_v1'],noise=noise,actual_initial_noise=actual_noise,
            trace_x=trace_x,trace_v=trace_v,trace_t=trace_t,receipt_json=np.array(json.dumps(receipt,sort_keys=True)))
        with temp.open('rb') as f:os.fsync(f.fileno())
        os.replace(temp,output)
        receipt['output_sha256']=sha(output);write(journal,receipt)
        print(json.dumps(dict(status='VALID',output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('donor','N4096','R4096'),required=True);main(p.parse_args().point)
