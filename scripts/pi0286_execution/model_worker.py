"""Resident original BF16 K4 inference with actual per-episode language."""
import argparse,json,os,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
import numpy as np
from contract import ROOT,INSTRUCTIONS,guard,sha,write
from budget import charge,process,time_allowed


def query(model,codec,request):
    import hashlib,torch
    from gr00t.policy.gr00t_policy import _rec_to_dtype
    time_allowed(ROOT,2)
    source,target=Path(request['input']),Path(request['output'])
    permitted=[ROOT.resolve()]
    tier=ROOT/'design/RECORDING_TIER.json'
    if tier.exists():
        bound=json.loads(tier.read_text());permitted.append(Path(bound['record_root']).resolve())
        permitted.extend(Path(p).resolve() for p in bound.get('large_output_roots',{}).values())
    assert any(source.resolve().is_relative_to(r) for r in permitted+[(ROOT/'data').resolve()]),'Unsealed source input'
    assert any(target.resolve().is_relative_to(r) for r in permitted),'Output must remain in owned PI0286 storage'
    assert not target.exists(),'Reuse first saved output, including valid failures'
    with np.load(source) as f:state=f['actual_state'].copy();base=f['base_raw'].copy();wrist=f['wrist_raw'].copy();instruction=str(f['instruction'])
    assert instruction==INSTRUCTIONS['T1'] and state.shape==(10,)
    inputs=_rec_to_dtype(dict(codec.observation(base,wrist,state,instruction)),dtype=torch.bfloat16);assert 'action' not in inputs
    key=json.dumps(dict(condition=request['condition'],query_index=request['query_index']),sort_keys=True)
    noise_path=ROOT/'runtime/noise'/(hashlib.sha256(key.encode()).hexdigest()+'.npy');noise_path.parent.mkdir(exist_ok=True)
    seed=int.from_bytes(hashlib.sha256(('PI0286:'+key).encode()).digest()[:8],'little')%(2**63-1)
    packet=np.load(noise_path) if noise_path.exists() else None
    if packet is not None:assert packet.shape==(1,40,132) and packet.dtype==np.uint16
    expected=torch.from_numpy(packet.copy()).view(torch.bfloat16).cuda() if packet is not None else None
    charge(ROOT,'query',str(target));heads=[];states=[];times=[];draws=[]
    def head(*args):charge(ROOT,'head',str(target));heads.append(1)
    def capture(module,args):states.append(args[0].detach().cpu().view(torch.uint16).numpy().copy());times.append(args[1].detach().cpu().numpy().copy())
    hook=model.action_head.model.register_forward_pre_hook(head);trace=model.action_head.action_encoder.register_forward_pre_hook(capture)
    original=torch.randn
    def noise(*args,**kwargs):
        nonlocal expected
        assert tuple(kwargs.get('size',args[0] if args else ()))==(1,40,132) and kwargs['dtype']==torch.bfloat16 and torch.device(kwargs['device'])==torch.device('cuda:0')
        if expected is None:
            expected=original(*args,**kwargs);temp=noise_path.with_suffix('.tmp')
            with temp.open('wb') as f:np.save(f,expected.detach().cpu().view(torch.uint16).numpy());f.flush();os.fsync(f.fileno())
            temp.replace(noise_path)
        draws.append(1);return expected.clone()
    began=time.perf_counter();model.action_head.num_inference_timesteps=4
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.randn=noise
            with torch.inference_mode():returned=model.get_action(inputs=inputs);result=returned['action_pred'].detach()
            torch.cuda.synchronize()
    finally:torch.randn=original;hook.remove();trace.remove()
    assert len(draws)==1 and len(heads)==len(states)==4 and result.dtype==torch.bfloat16 and result.shape==(1,40,132)
    raw=result.float().cpu().numpy();states.append(result.cpu().view(torch.uint16).numpy().copy());error=''
    try:decoded=codec.decode_actions(raw[0],state)
    except ValueError as exc:error=str(exc);decoded=np.full((40,10),np.nan,dtype=np.float32)
    from learning import tensor_tree
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,raw=raw,decoded=decoded,noise_uint16=expected.cpu().view(torch.uint16).numpy(),raw_states_uint16=np.stack(states),native_time_buckets=np.stack(times),K=4,actual_state=state,instruction=np.asarray(instruction),source_sha256=np.asarray(sha(source)),decode_error=np.asarray(error),identity_json=np.asarray(json.dumps(codec.identity,sort_keys=True)),condition_fields_json=np.asarray(json.dumps(tensor_tree(inputs),sort_keys=True)),latency_s=time.perf_counter()-began)
    return dict(output=str(target),heads=4,K=4,sha256=sha(target))


def run(point,canary=None):
    import fcntl,torch
    guard();sys.path.append(str(ROOT.parent/'PI0283/code'))
    lock=(ROOT/'runtime/MODEL_LEASE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from native_model import load_model
    from language import LanguageCodec
    from scope import digest
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    binding=json.loads((ROOT/'design/MODEL_POINTS.json').read_text())[point]
    for name,h in binding['files'].items():assert sha(Path(binding['model_root'])/name)==h,name
    started=process(ROOT,'start')
    try:
        model,info=load_model(binding['model_root']);assert all(not info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval();model.requires_grad_(False);model.to('cuda:0');assert model.action_head.action_horizon==40
        expected=binding['effective_bf16_tensor_sha256'];assert set(dict(model.named_parameters()))==set(expected)
        for n,p in model.named_parameters():assert digest(p)==expected[n],n
        codec=LanguageCodec(ROOT/'data');print(json.dumps(dict(ready=True,point=point,per_episode_language=True,K=4)),flush=True)
        if canary:
            from gate import original_gate
            target=Path(canary)/'CANARY_OUTPUT.npz';request=dict(input=str(ROOT/'data/CANARY_INPUT.npz'),output=str(target),condition='PI0286_FIXED_RESUME_CANARY',query_index=0)
            if not target.exists():query(model,codec,request)
            with np.load(target) as f:
                raw=f['raw'].copy();pred=f['decoded'].copy();state=f['actual_state'].copy();packet=f['noise_uint16'].copy()
            try:gate=original_gate(pred,state,0.,'PI0286_CANARY')
            except (ValueError,AssertionError) as exc:gate=dict(decision='REJECT',error=str(exc))
            import hashlib
            ah=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            write(Path(canary)/'CANARY_PREDICTIONS.json',dict(raw_sha256=ah(raw),noise_sha256=ah(packet),decoded_sha256=ah(pred),gate=gate))
        else:
            for line in sys.stdin:print(json.dumps(query(model,codec,json.loads(line))),flush=True)
        for n,p in model.named_parameters():assert digest(p)==expected[n],n
    finally:process(ROOT,'end',started);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',required=True);p.add_argument('--canary');a=p.parse_args();run(a.point,a.canary)
