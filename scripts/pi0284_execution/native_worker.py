"""Versioned inference adapter; reuse the proven parent loader, native sampler and codec."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
import torch
from budget import publication,charge,process,time_allowed

def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def query(root,model,codec,request):
    k=request['K'];assert k in (4,16) and model.action_head.action_horizon==40
    time_allowed(root,2)
    source=Path(request['input']);target=Path(request['output']);noise_path=Path(request['noise'])
    assert not target.exists(),'Reuse valid output'
    assert target.resolve().is_relative_to(root.resolve())
    for p in (source,noise_path):assert p.resolve().is_relative_to(root.resolve()) or p.resolve().is_relative_to(Path('/home/liuyu-nfs/Projects/go2-arx5/PI0283').resolve())
    with np.load(source) as f:state=f['actual_state'].copy();base=f['base_raw'].copy();wrist=f['wrist_raw'].copy()
    from gr00t.policy.gr00t_policy import _rec_to_dtype
    inputs=_rec_to_dtype(dict(codec.observation(base,wrist,state)),dtype=torch.bfloat16)
    assert 'action' not in inputs
    packet=np.load(noise_path);assert packet.shape==(1,40,132) and packet.dtype==np.uint16
    noise=torch.from_numpy(packet.copy()).view(torch.bfloat16).cuda()
    charge(root,'query',str(target));heads=[];states=[];times=[];draws=[]
    def head(*args):charge(root,'head',str(target));heads.append(1)
    def capture(module,args):
        states.append(args[0].detach().cpu().view(torch.uint16).numpy().copy())
        times.append(args[1].detach().cpu().numpy().copy())
    hook=model.action_head.model.register_forward_pre_hook(head)
    trace=model.action_head.action_encoder.register_forward_pre_hook(capture)
    original=torch.randn
    def fixed(*args,**kwargs):
        assert tuple(kwargs.get('size',args[0] if args else ()))==(1,40,132)
        assert kwargs['dtype']==torch.bfloat16 and torch.device(kwargs['device'])==torch.device('cuda:0')
        draws.append(1);return noise.clone()
    began=time.perf_counter();model.action_head.num_inference_timesteps=k
    try:
        torch.randn=fixed
        with torch.inference_mode():
            returned=model.get_action(inputs=inputs);result=returned['action_pred'].detach()
        torch.cuda.synchronize()
    finally:torch.randn=original;hook.remove();trace.remove()
    assert len(draws)==1 and len(heads)==len(states)==k and result.dtype==torch.bfloat16
    raw=result.float().cpu().numpy();assert raw.shape==(1,40,132)
    states.append(result.cpu().view(torch.uint16).numpy().copy())
    error=''
    try:decoded=codec.decode_actions(raw[0],state)
    except ValueError as exc:error=str(exc);decoded=np.full((40,10),np.nan,dtype=np.float32)
    target.parent.mkdir(parents=True,exist_ok=True)
    condition={n:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=ah(v.detach().cpu().contiguous().view(torch.uint8).numpy())) for n,v in inputs.items() if isinstance(v,torch.Tensor)}
    features={n:ah(returned[n].detach().cpu().contiguous().view(torch.uint8).numpy()) for n in ('backbone_features','state_features')}
    np.savez_compressed(target,raw=raw,decoded=decoded,noise_uint16=packet,actual_state=state,
        raw_states_uint16=np.stack(states),native_time_buckets=np.stack(times),K=k,
        decode_error=np.asarray(error),source_sha256=np.asarray(hashlib.sha256(source.read_bytes()).hexdigest()),
        identity_json=np.asarray(json.dumps(codec.identity,sort_keys=True)),condition_fields_json=np.asarray(json.dumps(condition,sort_keys=True)),feature_hashes_json=np.asarray(json.dumps(features,sort_keys=True)),latency_s=time.perf_counter()-began)
    return dict(output=str(target),K=k,heads=len(heads),raw_sha256=ah(raw),noise_sha256=ah(packet),decoded_sha256=ah(decoded))

def main(root):
    publication(root);card=json.loads((root/'design/RUNTIME_CARD.json').read_text())
    for path,h in card['dependencies'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    sys.path.insert(0,card['parent_code'])
    from inference import load,array_hash
    assert Path(sys.modules['inference'].__file__).resolve()==Path(card['parent_code'])/'inference.py'
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=process(root,'start')
    try:
        model,codec,binding=load(Path(card['parent_root']),'EVENT1024')
        print(json.dumps(dict(ready=True,parent='EVENT1024',native_load=card['parent_code'],actual_modes={n:m.training for n,m in [('model',model),('backbone',model.backbone),('head',model.action_head)]})),flush=True)
        for line in sys.stdin:print(json.dumps(query(root,model,codec,json.loads(line))),flush=True)
        for name,p in model.named_parameters():assert array_hash(p.detach().cpu().contiguous().view(torch.uint8).numpy())==binding['effective_bf16_tensor_sha256'][name],name
    finally:process(root,'end',start)

def prepare_noise(root):
    publication(root);done=root/'design/NOISE_PACKETS.json'
    assert not done.exists(),'Reuse sealed native packets'
    card=json.loads((root/'design/PRE_SCORE_INPUTS.json').read_text());pairs=json.loads((root/'design/NATURAL_PAIRS.json').read_text())['pairs'];members=[]
    start=process(root,'start')
    try:
        for row in card['rows']:
            for packet in (1,2,3):members.append((row['uid'],packet,Path(row['input']).parent/f'noise-{packet}.npy'))
        for i,pair in enumerate(pairs):
            for packet in (0,1):members.append((f'NATURAL_PAIR_{i}',packet,root/'natural'/f'pair-{i}-noise-{packet}.npy'))
        saved=[]
        for uid,packet,path in members:
            assert not path.exists()
            seed=int.from_bytes(hashlib.sha256(f'28401:{uid}:{packet}'.encode()).digest()[:8],'little')%(2**63-1)
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                value=torch.randn(size=(1,40,132),dtype=torch.bfloat16,device='cuda:0').cpu().view(torch.uint16).numpy()
            np.save(path,value,allow_pickle=False)
            saved.append(dict(uid=uid,packet=packet,seed=seed,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),array_sha256=ah(value)))
        done.write_text(json.dumps(dict(status='ALL_ACTUAL_NATIVE_BF16_FULL_NOISE_FIXED_BEFORE_NEW_SCORES',members=saved,diagnostic_rows=48,native_dimensions=[1,40,132]),indent=2)+'\n')
        print(json.dumps(dict(status='PASS_PRE_SCORE_NATIVE_NOISE',new_neural_queries=0,members=len(saved))))
    finally:process(root,'end',start)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--prepare-noise',action='store_true');a=p.parse_args()
    prepare_noise(Path(a.root)) if a.prepare_noise else main(Path(a.root))
