"""Finite stdin inference worker. PI0273 final-only quality with shared actual training admission."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0273';R1=ROOT
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


from quality_gate import admitted


def main(point):
    admitted()
    import torch
    from transformers import AutoModel
    from gr00t.data.embodiment_tags import EmbodimentTag
    import pi0242_train_infer_bridge as bridge
    import stage21_same_scene_groot_inference as stage21
    from stage40_b0_model_control import _bind_local_cosmos_processor
    card=read(R1/'design/RUNTIME_EVALUATION_CARD.json');binding=read(R1/'model_bindings'/(point+'.json'))
    assert binding['view_type']=='model-only' and binding['point']==point
    for name,digest in binding['model_files_sha256'].items():assert sha(Path(binding['view'])/name)==digest
    cosmos=BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot'
    assert _bind_local_cosmos_processor(Path('/home/liuyu-nfs/Projects/isaac-gr00t'),cosmos)['resolved_snapshot']==str(cosmos.resolve())
    arm='ROT'
    statistics=R1/'contracts'/arm/'source/dataset/statistics.json'
    assert sha(statistics)==card['contracts'][arm]==binding['statistics_sha256']
    proc=build(read(statistics),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    model=AutoModel.from_pretrained(binding['view'],local_files_only=True,trust_remote_code=True).eval().to('cuda:0',dtype=torch.bfloat16)
    assert model.action_head.num_inference_timesteps==4
    mode_path=R1/'design'/f"{point.removesuffix('5120')}_READOUT_R1.json"
    mode=read(mode_path);assert mode['mode']=='S4' and mode['query_dtype_by_phase']['quality']=='torch.bfloat16'
    assert sha(mode_path)==binding['readout_contract_sha256']
    attempts=R1/'evidence/model_attempts.jsonl';headsfile=R1/'evidence/model_heads.jsonl';heads=count(headsfile)
    print(json.dumps(dict(ready=True,point=point)),flush=True)
    for line in sys.stdin:
        request=json.loads(line);input_path=Path(request['input']);output=Path(request['output'])
        raw_root=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0273').resolve()
        assert (input_path.resolve().is_relative_to(raw_root) or input_path.resolve().is_relative_to(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')) or input_path.resolve().is_relative_to(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0271')) or input_path.resolve().is_relative_to(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0268/R1/factors')) or input_path.resolve().is_relative_to((BASE/'PI0268/R1/design/fit_fixtures').resolve())) and output.resolve().is_relative_to(raw_root)
        with np.load(input_path,allow_pickle=False) as z:
            packet=dict(body_rgb=z['body_rgb'].copy(),wrist_rgb=z['wrist_rgb'].copy(),state_v1=z['state_v1'].copy(),language=request['language'])
        assert set(packet)=={'body_rgb','wrist_rgb','state_v1','language'}
        policy=batch(proc,packet)['inputs'];assert 'action' not in policy and 'action_mask' not in policy
        noise_path=Path(request['noise_file']);noise_name=request['noise_name']
        assert any(noise_path.resolve().is_relative_to(p) for p in (Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1/noise'),Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0269/noise'),raw_root.parent/'PI0270/noise-newfit',raw_root.parent/'PI0271/noise-support'))
        registry=request.get('noise_registry','RUNTIME_EVALUATION_CARD.json')
        assert registry in ('RUNTIME_EVALUATION_CARD.json','RUNTIME_OFFLINE_CARD.json','RUNTIME_COMPONENT_CARD.json','NEWFIT_QUERY_CARD.json','LATE_QUERY_CARD.json')
        assert point in ('BROAD5120','TARGET5120')
        noise_card=card if registry=='RUNTIME_EVALUATION_CARD.json' else read(R1/'design'/registry)
        entry=next(r for r in noise_card['noise'] if r['file']==str(noise_path));assert sha(noise_path)==entry['sha256']
        row=next(r for r in entry['keys'] if r['name']==noise_name)
        with np.load(noise_path) as z:noise=z[noise_name].copy()
        assert noise.shape==(1,40,132) and noise.dtype==np.float32 and np.isfinite(noise).all()
        assert hashlib.sha256(noise.tobytes()).hexdigest()==row['noise_sha256']
        input_identity=identity(policy);journal=output.with_suffix('.receipt.json')
        if 'expected_condition_sha256' in request:
            assert input_identity==request['expected_condition_sha256'],'FROZEN_CONDITION_IDENTITY'
        if output.exists():
            if journal.exists():prior=read(journal)
            else:
                with np.load(output,allow_pickle=False) as z:prior=json.loads(str(z['receipt_json']))
                prior['output_sha256']=sha(output);write(journal,prior)
            assert prior['point']==point and prior['binding']==binding and prior['input_sha256']==input_identity
            assert prior['noise_sha256']==row['noise_sha256'] and prior['output_sha256']==sha(output)
            print(json.dumps(dict(status='VALID_REUSED',output=str(output))),flush=True);continue
        ordinal=count(attempts)+1;assert ordinal<=18432
        if any(output.resolve().is_relative_to(raw_root/lane) for lane in ('virtual','component')):
            virtual_attempts=R1/'evidence/virtual_model_attempts.jsonl'
            virtual_ordinal=count(virtual_attempts)+1
            append(virtual_attempts,dict(ordinal=virtual_ordinal,global_attempt=ordinal,point=point,request=request))
        append(attempts,dict(ordinal=ordinal,point=point,request=request,input_sha256=input_identity,noise_sha256=row['noise_sha256']))
        def hook(module,inputs):
            nonlocal heads
            heads+=1;assert heads<=73728
            append(headsfile,dict(ordinal=heads,attempt=ordinal,point=point))
        before=heads;query_dtypes=[];output_dtypes=[]
        handle=model.action_head.model.register_forward_pre_hook(hook)
        query_hook=model.action_head.action_encoder.register_forward_pre_hook(lambda _m,args:query_dtypes.append(str(args[0].dtype)))
        output_hook=model.action_head.action_decoder.register_forward_hook(lambda _m,_args,value:output_dtypes.append(str(value.dtype)))
        try:
            with torch.inference_mode():stock=bridge._stock_query(model,policy,stage21,noise)
        finally:handle.remove();query_hook.remove();output_hook.remove()
        assert heads-before==4 and query_dtypes==['torch.bfloat16']*4 and len(output_dtypes)==4
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
            attempt=ordinal,heads=4,native_S4=True,execution_phase='quality',query_dtype='torch.bfloat16',observed_query_dtypes=query_dtypes,observed_readout_dtypes=output_dtypes,frozen_request=request,model_latency_s=float(stock['elapsed_s']),initial_noise_dtype='torch.bfloat16',
            source_noise_dtype='numpy.float32',finite_output=finite,decode_failure=decode_failure,no_teacher_or_privileged_condition=True)
        output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_suffix('.partial.npz')
        np.savez_compressed(temp,raw=raw,decoded=decoded,state=packet['state_v1'],noise=noise,actual_initial_noise=actual_noise,
            trace_x=trace_x,trace_v=trace_v,trace_t=trace_t,receipt_json=np.array(json.dumps(receipt,sort_keys=True)))
        with temp.open('rb') as f:os.fsync(f.fileno())
        os.replace(temp,output)
        receipt['output_sha256']=sha(output);write(journal,receipt)
        print(json.dumps(dict(status='VALID',output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('BROAD5120','TARGET5120'),required=True);main(p.parse_args().point)
