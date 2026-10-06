"""PI0277 fixed-policy S4 evaluation; inherited math, new roots and bounded ledger."""
import argparse,json,os,sys,hashlib
from pathlib import Path
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');COMMON=BASE/'PI0277';ROOT=COMMON/'R2';RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0277/R2')
DATA=RAW.parents[1]/'PI0260/R2'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(BASE/'PI0250/runtime/hf_home'),
    NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false')
sys.path[:0]=[str(DATA/'R3'),str(DATA/'runtime'),str(DATA/'qa'),str(BASE/'PI0250/runtime/reference-317a/scripts'),
    str(BASE/'PI0250/runtime/reference-317a/src'),'/home/liuyu-nfs/Projects/isaac-gr00t']
import numpy as np
from pi0260_r3_inputs import build,batch,identity
from evaluation_card import sha,write

def append(p,row):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def main(point):
    import torch
    from transformers import AutoModel
    from gr00t.data.embodiment_tags import EmbodimentTag
    import pi0242_train_infer_bridge as bridge
    import stage21_same_scene_groot_inference as stage21
    admission=json.loads((ROOT/'evidence'/f'POLICY_ADMISSION_{point}.json').read_text())
    assert admission['status']=='PASS_PI0277_FIXED_POLICY_IDENTITY'
    assert admission['evaluation_card_sha256']==sha(ROOT/'design/EVALUATION_CARD.json')
    for r in json.loads((ROOT/'design/EVALUATION_FILESET.json').read_text())['files']:assert sha(r['path'])==r['sha256']
    binding=json.loads((ROOT/'model_bindings'/f'{point}.json').read_text())
    noise_card=json.loads((ROOT/'design/EVALUATION_NOISE.json').read_text())
    noises={(r['file'],r['name']):r for r in noise_card['rows']}
    for name,h in binding['model_files_sha256'].items():assert sha(Path(binding['view'])/name)==h
    stats=ROOT/'contracts/TASK/source/dataset/statistics.json';assert sha(stats)==binding['statistics_sha256']
    from stage40_b0_model_control import _bind_local_cosmos_processor
    cosmos=BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot'
    assert _bind_local_cosmos_processor(Path('/home/liuyu-nfs/Projects/isaac-gr00t'),cosmos)['resolved_snapshot']==str(cosmos.resolve())
    proc=build(json.loads(stats.read_text()),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    model=AutoModel.from_pretrained(binding['view'],local_files_only=True,trust_remote_code=True).eval().to('cuda:0',dtype=torch.bfloat16)
    assert model.action_head.num_inference_timesteps==4
    journal=COMMON/'evidence/model_attempts.jsonl';headsfile=COMMON/'evidence/model_heads.jsonl'
    attempts=sum(1 for _ in journal.open()) if journal.exists() else 0
    heads=sum(1 for _ in headsfile.open()) if headsfile.exists() else 0
    print(json.dumps(dict(ready=True,point=point)),flush=True)
    for line in sys.stdin:
        request=json.loads(line);inp=Path(request['input']);out=Path(request['output']);noise_path=Path(request['noise_file'])
        assert all(p.resolve().is_relative_to(RAW.resolve()) for p in (inp,out))
        assert noise_path.resolve().is_relative_to((RAW.parent/'noise').resolve()) or noise_path.resolve().is_relative_to((RAW.parent.with_name('PI0276')/'noise').resolve())
        frozen=noises[str(noise_path),request['noise_name']]
        assert request['noise_sha256']==frozen['sha256']
        if point=='TASK8192':assert frozen['lane']=='FIT', 'Reuse all old valid donor offline/physical outputs'
        with np.load(inp,allow_pickle=False) as z:
            assert set(z.files)=={'body_rgb','wrist_rgb','state_v1'}
            packet={k:z[k].copy() for k in z.files}
        packet['language']='Stack the red cube on the green cube and release it.'
        policy=batch(proc,packet)['inputs'];assert 'action' not in policy and 'action_mask' not in policy
        key=identity(policy)
        if frozen['lane']=='FIT':
            expected=next(r for r in json.loads((ROOT/'design/FIT_CARD.json').read_text())['rows'] if r['uid']==frozen['condition'])
            assert key==expected['condition_sha256'], 'ACTUAL_FIT_INPUT_IDENTITY'
        with np.load(noise_path) as z:noise=z[request['noise_name']].copy()
        assert noise.shape==(1,40,132) and noise.dtype==np.float32 and np.isfinite(noise).all()
        digest=hashlib.sha256(noise.tobytes()).hexdigest();assert digest==request['noise_sha256']
        if out.exists():
            with np.load(out) as z:receipt=json.loads(str(z['receipt_json']))
            assert receipt['input_sha256']==key and receipt['noise_sha256']==digest and receipt['binding']==binding
            receipt['output_sha256']=sha(out)
            sidecar=out.with_suffix('.receipt.json')
            if sidecar.exists():assert json.loads(sidecar.read_text())==receipt
            else:write(sidecar,receipt)
            print(json.dumps(dict(status='VALID_REUSED',output=str(out))),flush=True);continue
        attempts+=1;assert attempts<=14336
        append(journal,dict(ordinal=attempts,point=point,request=request,input_sha256=key,noise_sha256=digest))
        def count_head(module,args):
            nonlocal heads
            heads+=1;assert heads<=57344
            append(headsfile,dict(ordinal=heads,attempt=attempts,point=point))
        before=heads;query_dtypes=[];output_dtypes=[]
        hook=model.action_head.model.register_forward_pre_hook(count_head)
        query_hook=model.action_head.action_encoder.register_forward_pre_hook(lambda _m,args:query_dtypes.append(str(args[0].dtype)))
        output_hook=model.action_head.action_decoder.register_forward_hook(lambda _m,_args,value:output_dtypes.append(str(value.dtype)))
        try:
            with torch.inference_mode():stock=bridge._stock_query(model,policy,stage21,noise)
        finally:hook.remove();query_hook.remove();output_hook.remove()
        assert heads-before==4 and query_dtypes==['torch.bfloat16']*4 and len(output_dtypes)==4
        raw=np.asarray(stock['raw'],np.float32);error=None
        if np.isfinite(raw).all():
            try:decoded=bridge._decode_physical(proc,EmbodimentTag.NEW_EMBODIMENT,raw,packet['state_v1'])
            except bridge.PI0242Blocker as e:decoded=np.full((1,40,10),np.nan,np.float32);error=str(e)
        else:decoded=np.full((1,40,10),np.nan,np.float32)
        assert len(stock['trace'])==4 and np.array_equal(stock['noise'],noise)
        tx=np.stack([r['x'] for r in stock['trace']]);tv=np.stack([r['v'] for r in stock['trace']]);tt=np.stack([r['t'] for r in stock['trace']])
        actual_noise=torch.from_numpy(noise).to(torch.bfloat16).float().numpy();assert np.array_equal(tx[0],actual_noise)
        receipt=dict(status='VALID',point=point,binding=binding,input_sha256=key,noise_sha256=digest,attempt=attempts,
            heads=4,decode_failure=error,finite_output=bool(np.isfinite(raw).all() and np.isfinite(decoded).all()),
            model_latency_s=float(stock['elapsed_s']),query_dtype='BF16',observed_query_dtypes=query_dtypes,observed_readout_dtypes=output_dtypes,
            execution_phase='quality',physical_decode_dtype='FP32',no_teacher_or_privileged_inputs=True)
        out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix('.partial.npz')
        np.savez_compressed(temp,raw=raw,decoded=decoded,state=packet['state_v1'],noise=noise,actual_initial_noise=actual_noise,
            trace_x=tx,trace_v=tv,trace_t=tt,receipt_json=np.array(json.dumps(receipt,sort_keys=True)))
        with temp.open('rb') as f:os.fsync(f.fileno())
        os.replace(temp,out);receipt['output_sha256']=sha(out);write(out.with_suffix('.receipt.json'),receipt)
        print(json.dumps(dict(status='VALID',output=str(out))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=tuple(f'{a}{n}' for a in ('GRIP','FOCUS') for n in (512,1024,2048,4096)),required=True);main(p.parse_args().point)
