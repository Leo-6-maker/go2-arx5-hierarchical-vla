"""Resident native BF16 S4 queries, fixed full noise and unchanged EEF10 decode."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from codec import load_codec,sha

def array_hash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def charge(root,kind,identity):
    import fcntl
    path=root/'source/evidence/model_budget.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
        count=sum(json.loads(line)['kind']==kind for line in f if line.strip())
        assert count<(5120 if kind=='policy_attempt' else 20480)
        f.seek(0,2);f.write(json.dumps(dict(kind=kind,id=identity,utc_ns=time.time_ns()))+'\n');f.flush();os.fsync(f.fileno())

def load(root,point):
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from safetensors import safe_open
    manifest=json.loads((root/'design/IMPLEMENTATION.json').read_text())
    assert manifest['release']=='8fdaab73b8b6686409317d9562833235c2839acc' and manifest['published_commit']
    for name,digest in manifest['files'].items():assert sha(root/'code'/name)==digest,name
    binding=json.loads((root/'design/MODEL_POINTS.json').read_text())[point]
    for name,digest in binding['files'].items():assert sha(Path(binding['model_root'])/name)==digest,name
    model,info=Gr00tN1d7.from_pretrained(binding['model_root'],torch_dtype=torch.bfloat16,local_files_only=True,output_loading_info=True)
    assert all(not info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),info
    model.eval();model.requires_grad_(False);model.to('cuda:0')
    assert model.action_head.num_inference_timesteps==4 and model.action_head.action_horizon==40
    expected=binding['effective_bf16_tensor_sha256'];parameters=dict(model.named_parameters())
    assert set(parameters)==set(expected)
    for name,p in parameters.items():assert array_hash(p.detach().cpu().contiguous().view(torch.uint8).numpy())==expected[name],name
    return model,load_codec(root/'data'),binding

def query(root,model,codec,request):
    from gr00t.policy.gr00t_policy import _rec_to_dtype
    source,target=Path(request['input']),Path(request['output'])
    assert source.resolve().is_relative_to(root.resolve()) and target.resolve().is_relative_to(root.resolve())
    assert not target.exists(),'Reuse first valid saved output'
    with np.load(source) as data:
        state=data['actual_state'].copy();base=data['base_raw'].copy();wrist=data['wrist_raw'].copy()
    assert state.shape==(10,)
    inputs=_rec_to_dtype(dict(codec.observation(base,wrist,state)),dtype=torch.bfloat16)
    assert 'action' not in inputs,'No future action, expert prefix or RTC input'
    active=str(target);charge(root,'policy_attempt',active)
    heads=[]
    def head(*args):charge(root,'action_head',active);heads.append(1)
    hook=model.action_head.model.register_forward_pre_hook(head)
    key=json.dumps(dict(condition=request['condition'],query_index=request['query_index']),sort_keys=True)
    noise_path=root/'source/noise'/(hashlib.sha256(key.encode()).hexdigest()+'.npy')
    noise_path.parent.mkdir(parents=True,exist_ok=True)
    expected=None
    if noise_path.exists():
        array=np.load(noise_path);assert array.shape==(1,40,132) and array.dtype==np.uint16
        expected=torch.from_numpy(array.copy()).view(torch.bfloat16).cuda()
    seed=int.from_bytes(hashlib.sha256(('PI0282-28203-'+key).encode()).digest()[:8],'little')%(2**63-1)
    draws=[];original=torch.randn
    def native_noise(*args,**kwargs):
        nonlocal expected
        shape=kwargs.get('size',args[0] if args else None)
        assert tuple(shape)==(1,40,132) and kwargs['dtype']==torch.bfloat16
        assert torch.device(kwargs['device'])==torch.device('cuda:0')
        if expected is None:
            expected=original(*args,**kwargs)
            # The first policy saves its actual native draw before the Euler steps.
            tmp=noise_path.with_suffix('.tmp')
            with tmp.open('wb') as f:np.save(f,expected.detach().cpu().view(torch.uint16).numpy());f.flush();os.fsync(f.fileno())
            tmp.replace(noise_path)
        draws.append(1);return expected.clone()
    started=time.perf_counter()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.randn=native_noise
        try:
            with torch.inference_mode():raw=model.get_action(inputs=inputs)['action_pred'].detach().float().cpu().numpy()
            torch.cuda.synchronize()
        finally:torch.randn=original;hook.remove()
    assert len(draws)==1 and len(heads)==4 and raw.shape==(1,40,132)
    decoded=codec.decode_actions(raw[0],state)
    noise=expected.detach().cpu().view(torch.uint16).numpy()
    np.savez_compressed(target,raw=raw,decoded=decoded,noise_uint16=noise,actual_state=state,
        source_sha256=np.asarray(sha(source)),identity_json=np.asarray(json.dumps(codec.identity,sort_keys=True)),
        normalized_OOB_fraction=np.mean((raw[0,:,:10]<-1)|(raw[0,:,:10]>1)),latency_s=time.perf_counter()-started)
    return dict(output=str(target),raw_sha256=array_hash(raw),noise_sha256=array_hash(noise),decoded_sha256=array_hash(decoded))

def main(args):
    root=Path(args.root);torch.set_num_threads(4);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert os.environ.get('CUDA_VISIBLE_DEVICES') or os.environ.get('PBS_JOBID')
    model,codec,binding=load(root,args.point)
    print(json.dumps(dict(ready=True,point=args.point)),flush=True)
    if args.canary:
        from cpu_inputs import FrozenDataset
        from go2arx5_vla.contracts import TrajectoryChunkV1
        from go2arx5_vla.safety import TrajectorySafetyGateV1
        from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,matrix_to_quat
        row=FrozenDataset(root/'data',cache=False);base,wrist,state,_=row.raw(row.rows[0])
        folder=root/'learning'/args.arm/args.phase;source=folder/'CANARY_INPUT.npz';target=folder/'CANARY_OUTPUT.npz'
        np.savez_compressed(source,base_raw=base,wrist_raw=wrist,actual_state=state)
        request=dict(input=str(source),output=str(target),condition='CANARY-'+args.arm,query_index=0)
        answer=query(root,model,codec,request)
        with np.load(target) as data:commands=data['decoded']
        try:
            chunk=TrajectoryChunkV1(chunk_id='PI0282_CANARY',source='PI0282_EEF10_NATIVE',observation_timestamp=0.,point_timestamps=np.arange(40)*.05,
                eef_position=commands[:,:3],eef_orientation=matrix_to_quat(rot6d_rows_to_matrix(commands[:,3:9])),source_frame='PI0276_MANISKILL_WORLD',
                tcp_semantic_id='PI0276_PANDA_HAND_TCP',validity_start=0.,validity_end=1.95,gripper=commands[:,-1])
            gate=TrajectorySafetyGateV1().evaluate(chunk,now=0.,achieved_position=state[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[None,3:9]))[0]).as_dict()
        except ValueError as exc:gate=dict(decision='REJECT',invalid_rotation=str(exc))
        result={k:v for k,v in answer.items() if k!='output'};result['gate']=gate
        (folder/'CANARY_PREDICTIONS.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    else:
        for line in sys.stdin:print(json.dumps(query(root,model,codec,json.loads(line))),flush=True)
    expected=binding['effective_bf16_tensor_sha256']
    for name,p in model.named_parameters():assert array_hash(p.detach().cpu().contiguous().view(torch.uint8).numpy())==expected[name],name

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--point',required=True)
    parser.add_argument('--canary',action='store_true');parser.add_argument('--arm');parser.add_argument('--phase')
    args=parser.parse_args();assert not args.canary or args.arm and args.phase;main(args)
