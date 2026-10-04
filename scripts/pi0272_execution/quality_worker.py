"""Frozen PI0272 requests only; one phase-bound readout and first-valid output."""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0272'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0272')
sys.path.insert(0,str(BASE/'PI0270/code'))
# Import the qualified worker under a distinct name, not this module's filename.
import importlib.util
spec=importlib.util.spec_from_file_location('pi0270_quality_worker',BASE/'PI0270/code/quality_worker.py')
mature=importlib.util.module_from_spec(spec);spec.loader.exec_module(mature)
read,sha,write,append,count=mature.read,mature.sha,mature.write,mature.append,mature.count
sys.path.insert(0,str(Path(__file__).resolve().parent))


def admitted():
    resolution=read(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json')
    assert resolution['amendment_release']=='8a2ffbefe0c57536c0a7e340da0616c7987c74f1'
    assert resolution['status']=='RESOLVED_BY_EXPLICIT_SCIENTIFIC_AUTHORITY'
    audit=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json')
    assert audit['status']=='PASS_PI0272_BOTH_ACTUAL_1024_MATCHED_TRAINING'
    assert {(r['arm'],r['step']) for r in audit['complete_checkpoints']}=={(a,s) for a in ('S4FIT','DIRECT') for s in (512,1024)}
    assert read(ROOT/'design/EVALUATION_CARD.json')['expected_queries']==2148
    for row in read(ROOT/'design/QUALITY_EXECUTION_FILESET_R1.json')['files']:
        assert sha(row['path'])==row['sha256']


def main(point):
    admitted()
    import torch
    from transformers import AutoModel
    from gr00t.data.embodiment_tags import EmbodimentTag
    import pi0242_train_infer_bridge as bridge
    import stage21_same_scene_groot_inference as stage21
    from stage40_b0_model_control import _bind_local_cosmos_processor
    from readout import infer
    card=read(ROOT/'design/EVALUATION_CARD.json')
    rows={q['ordinal']:q for q in card['rows'] if q['point']==point}
    binding=read(ROOT/'model_bindings'/f'{point}.json')
    assert binding['view_type']=='model-only' and binding['point']==point
    for name,digest in binding['model_files_sha256'].items():assert sha(Path(binding['view'])/name)==digest
    mode=read(ROOT/'design'/('S4FIT_READOUT_R1.json' if point=='S4FIT1024' else 'DIRECT_READOUT_R1.json'))
    assert mode['query_dtype_by_phase']==dict(training='torch.float32',quality='torch.bfloat16')
    assert binding['readout_contract_sha256']==sha(ROOT/'design'/('S4FIT_READOUT_R1.json' if point=='S4FIT1024' else 'DIRECT_READOUT_R1.json'))
    cosmos=BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot'
    assert _bind_local_cosmos_processor(Path('/home/liuyu-nfs/Projects/isaac-gr00t'),cosmos)['resolved_snapshot']==str(cosmos.resolve())
    stats=ROOT/'contracts/ROT/source/dataset/statistics.json';assert sha(stats)==binding['statistics_sha256']==mode['statistics_sha256']
    proc=mature.build(read(stats),cosmos)
    model=AutoModel.from_pretrained(binding['view'],local_files_only=True,trust_remote_code=True).eval().to('cuda:0',dtype=torch.bfloat16)
    assert model.action_head.num_inference_timesteps==4
    attempts=ROOT/'evidence/model_attempts.jsonl';headsfile=ROOT/'evidence/model_heads.jsonl';heads=count(headsfile)
    print(json.dumps(dict(ready=True,point=point)),flush=True)
    for line in sys.stdin:
        request=json.loads(line);q=rows[request['ordinal']];assert request==q
        output=Path(q['output']);assert output.resolve().is_relative_to(RAW.resolve())
        assert sha(q['input'])==q['input_sha256']
        with np.load(q['input'],allow_pickle=False) as z:
            packet={k:z[k].copy() for k in ('body_rgb','wrist_rgb','state_v1')}
        packet['language']=q['language'];policy=mature.batch(proc,packet)['inputs']
        assert 'action' not in policy and 'action_mask' not in policy
        identity=mature.identity(policy);assert identity==q['expected_condition_sha256']
        journal=output.with_suffix('.receipt.json')
        if output.exists():
            if journal.exists():prior=read(journal)
            else:
                with np.load(output,allow_pickle=False) as z:prior=json.loads(str(z['receipt_json']))
                prior['output_sha256']=sha(output);write(journal,prior)
            assert prior['binding']==binding and prior['frozen_request']==q and prior['output_sha256']==sha(output)
            print(json.dumps(dict(status='VALID_REUSED',output=str(output))),flush=True);continue
        noise=None
        if q['readout']=='S4FIT':
            source=q['noise'];assert sha(source['file'])==source['sha256']
            with np.load(source['file'],allow_pickle=False) as z:noise=z['float32'].copy()
            digest=source.get('noise_sha256') or source['keys'][0]['noise_sha256']
            assert noise.shape==(1,40,132) and noise.dtype==np.float32 and np.isfinite(noise).all()
            assert hashlib.sha256(noise.tobytes()).hexdigest()==digest
        else:assert q['readout']=='DIRECT' and q['noise'] is None
        ordinal=count(attempts)+1;assert ordinal<=2560
        append(attempts,dict(ordinal=ordinal,point=point,request=q,input_sha256=identity))
        before=heads
        def hook(_module,_inputs):
            nonlocal heads
            heads+=1;assert heads<=7168
            append(headsfile,dict(ordinal=heads,attempt=ordinal,point=point))
        query_dtypes=[];output_dtypes=[]
        handle=model.action_head.model.register_forward_pre_hook(hook);arrays={}
        query_hook=model.action_head.action_encoder.register_forward_pre_hook(lambda _m,args:query_dtypes.append(str(args[0].dtype)))
        output_hook=model.action_head.action_decoder.register_forward_hook(lambda _m,_args,value:output_dtypes.append(str(value.dtype)))
        try:
            start=time.perf_counter()
            if q['readout']=='S4FIT':
                with torch.inference_mode():stock=bridge._stock_query(model,policy,stage21,noise)
                assert len(stock['trace'])==4 and np.array_equal(stock['noise'],noise)
                raw=np.asarray(stock['raw'],np.float32)
                arrays.update(noise=noise,actual_initial_noise=torch.from_numpy(noise).to(torch.bfloat16).float().numpy(),
                    trace_x=np.stack([x['x'] for x in stock['trace']]),trace_v=np.stack([x['v'] for x in stock['trace']]),
                    trace_t=np.stack([x['t'] for x in stock['trace']]))
                assert np.array_equal(arrays['trace_x'][0],arrays['actual_initial_noise'])
                actual_dtype=output_dtypes[-1];rng_unchanged=None
            else:
                cpu=torch.random.get_rng_state();cuda=torch.cuda.get_rng_state_all()
                with torch.inference_mode():
                    prepared=stage21._rec_to_dtype(copy.deepcopy(policy),torch.bfloat16)
                    bb,action=model.prepare_input(prepared);features=model.backbone(bb)
                    generated=infer(model.action_head,features,dict(state=action.state,embodiment_id=action.embodiment_id),execution_phase='quality')
                    actual_dtype=str(generated.dtype);raw=generated.detach().float().cpu().numpy()
                rng_unchanged=torch.equal(cpu,torch.random.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
                assert rng_unchanged
                arrays.update(zero_query=np.zeros((1,40,132),np.float32))
            elapsed=time.perf_counter()-start
        finally:handle.remove();query_hook.remove();output_hook.remove()
        assert heads-before==(4 if q['readout']=='S4FIT' else 1)
        assert query_dtypes==['torch.bfloat16']*(heads-before) and len(output_dtypes)==heads-before
        decode_failure=None
        if np.isfinite(raw).all():
            try:decoded=bridge._decode_physical(proc,EmbodimentTag.NEW_EMBODIMENT,raw,packet['state_v1'])
            except bridge.PI0242Blocker as error:
                assert str(error)=='P1 physical decode contract failed'
                decoded=np.full((1,40,10),np.nan,np.float32);decode_failure=str(error)
        else:decoded=np.full((1,40,10),np.nan,np.float32)
        receipt=dict(status='VALID',point=point,binding=binding,frozen_request=q,input_sha256=identity,attempt=ordinal,
            heads=heads-before,execution_phase='quality',query_dtype='torch.bfloat16',readout_output_dtype=actual_dtype,
            observed_query_dtypes=query_dtypes,observed_readout_dtypes=output_dtypes,
            native_S4=q['readout']=='S4FIT',direct_generation_RNG_unchanged=rng_unchanged,model_latency_s=elapsed,
            readout_contract_sha256=binding['readout_contract_sha256'],finite_output=bool(np.isfinite(raw).all() and np.isfinite(decoded).all()),
            decode_failure=decode_failure,no_teacher_or_privileged_condition=True)
        output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_suffix('.partial.npz')
        np.savez_compressed(temp,raw=raw,decoded=decoded,state=packet['state_v1'],receipt_json=np.array(json.dumps(receipt,sort_keys=True)),**arrays)
        with temp.open('rb') as f:os.fsync(f.fileno())
        os.replace(temp,output);receipt['output_sha256']=sha(output);write(journal,receipt)
        print(json.dumps(dict(status='VALID',output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('S4FIT1024','DIRECT0','DIRECT1024'),required=True)
    main(p.parse_args().point)
