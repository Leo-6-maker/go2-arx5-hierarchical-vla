"""Passive native RNG capture, ordered consumption and complete Trainer saves."""
import contextlib,hashlib,json,os,sys,time
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from scope import digest


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def seed_for(update,micro):
    return int.from_bytes(hashlib.sha256(f'28203:{update}:{micro}'.encode()).digest()[:8],'little')%(2**63-1)


class NativeNoise(TorchDispatchMode):
    def __init__(self):super().__init__();self.noise=None;self.time=None
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        value=func(*args,**(kwargs or {}))
        if str(func)=='aten.randn.default' and list(value.shape)[-2:]==[40,132]:
            assert self.noise is None,'MULTIPLE_NATIVE_NOISE_DRAWS';self.noise=value.detach().clone()
        return value
    def profile(self,frame,event,value):
        if event=='return' and frame.f_code.co_name=='sample_time' and frame.f_code.co_filename.endswith('/gr00t/model/gr00t_n1d7/gr00t_n1d7.py'):
            assert self.time is None;self.time=value.detach().clone()


@contextlib.contextmanager
def native_rng(update,micro):
    previous=sys.getprofile();observer=NativeNoise()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        seed=seed_for(update,micro);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        sys.setprofile(observer.profile)
        try:
            with observer:yield observer
        finally:sys.setprofile(previous)


class Journal:
    def __init__(self,path):
        self.path=Path(path);self.f=self.path.open('a',buffering=1<<20);self.count=0
    def append(self,value):
        self.f.write(json.dumps(value,sort_keys=True)+'\n');self.count+=1
        if self.count%16==0:self.f.flush()
    def close(self):
        self.f.flush();os.fsync(self.f.fileno());self.f.close()
        return dict(path=str(self.path),new_lines=self.count,sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            crash_loss_boundary='At most15 not-yet-flushed journal entries; final success receipt requires fsync. Completed optimizer ledger fsyncs every completed update.')


def tensor_tree(value):
    if torch.is_tensor(value):return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=digest(value))
    if isinstance(value,dict):return {str(k):tensor_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [tensor_tree(v) for v in value]
    return value


def completed_kind(rows,arm,phase,step):
    assert len(rows)<4224
    duplicate=any(r['arm']==arm and r['phase']==phase and r['step']==step for r in rows)
    kind='technical' if duplicate else ('memory' if phase=='memory' else ('formal' if phase=='main' else 'canary'))
    caps=dict(formal=4096,memory=32,technical=64,canary=32)
    assert sum(r['work_kind']==kind for r in rows)<caps[kind],('COMPLETED_UPDATE_CAP',kind)
    return kind


def check_next_update(root,arm,phase,step):
    p=Path(root)/'learning/completed_updates.jsonl'
    rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    completed_kind(rows,arm,phase,step)


def append_completed(root,arm,phase,step,consumed):
    # Completed optimizer work is charged immediately, including discarded
    # canaries and work subsequently rolled back to a recovery checkpoint.
    import fcntl
    root=Path(root);p=root/'learning/completed_updates.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);rows=[json.loads(x) for x in f if x.strip()]
        kind=completed_kind(rows,arm,phase,step)
        f.seek(0,2);f.write(json.dumps(dict(arm=arm,phase=phase,work_kind=kind,step=step,consumed_chunks=consumed,wall=time.time()))+'\n');f.flush();os.fsync(f.fileno())


def save_receipt(checkpoint,cursor,initial_scope):
    checkpoint=Path(checkpoint)
    required=['config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin']
    assert all((checkpoint/n).exists() for n in required), 'INCOMPLETE_TRAINER'
    assert list(checkpoint.glob('*.safetensors')),'MISSING_FULL_MODEL'
    payload=dict(cursor=cursor,scope=initial_scope,required_files=required)
    (checkpoint/'PI0282_RECOVERY.json').write_text(json.dumps(payload,indent=2)+'\n')


def export_point(checkpoint,original_base,output,parameter_names):
    """CPU serialization only: whole changed tensors, never residual addition."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    checkpoint=Path(checkpoint);original_base=Path(original_base);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    original_index=json.loads((original_base/'model.safetensors.index.json').read_text())['weight_map']
    rows={};changed={};full={};bf16={};file_hashes={}
    for shard in sorted(checkpoint.glob('*.safetensors')):
        file_hashes[shard.name]=file_digest(shard)
        with safe_open(shard,framework='pt',device='cpu') as source:
            for key in source.keys():
                value=source.get_tensor(key)
                full[key]=dict(dtype=str(value.dtype),shape=list(value.shape),sha256=digest(value))
                if key in parameter_names:bf16[key]=digest(value.to(torch.bfloat16))
                with safe_open(original_base/original_index[key],framework='pt',device='cpu') as parent:
                    old=parent.get_tensor(key)
                # dtype changes must also reconstruct exactly; unchanged packed
                # non10 rows remain in the native complete tensor container.
                differs=value.dtype!=old.dtype or not torch.equal(value,old)
                if differs:
                    assert key.startswith(('action_head.state_encoder.','action_head.action_encoder.','action_head.action_decoder.','action_head.model.')),key
                    changed[key]=value.contiguous();rows[key]=full[key]
    assert set(bf16)==set(parameter_names),'MISSING_EFFECTIVE_PARAMETER_HASH'
    # At most the authorized native projector/DiT payload is collected in RAM.
    target=output/'exact_changed_tensors.safetensors'
    assert not target.exists(),'Adopt immutable existing export'
    save_file(changed,str(target))
    reconstructed={}
    with safe_open(target,framework='pt',device='cpu') as delta:
        for key in full:
            if key in changed:value=delta.get_tensor(key)
            else:
                with safe_open(original_base/original_index[key],framework='pt',device='cpu') as parent:value=parent.get_tensor(key)
            reconstructed[key]=dict(dtype=str(value.dtype),shape=list(value.shape),sha256=digest(value))
    assert reconstructed==full,'EXACT_FULL_STATE_RECONSTRUCTION_MISMATCH'
    def tree_hash(tree):return hashlib.sha256(json.dumps(tree,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for name in ('config.json','processor_config.json','statistics.json','embodiment_id.json'):
        assert (checkpoint/name).exists();file_hashes[name]=file_digest(checkpoint/name)
    receipt=dict(status='PASS_CPU_EXACT_CHANGED_TENSOR_EXPORT_AND_FULL_STATE_HASH_PARITY',
       original_base=str(original_base),checkpoint=str(checkpoint),model_files=file_hashes,
       exact_delta_file=target.name,exact_delta_bytes=target.stat().st_size,exact_delta_sha256=file_digest(target),
       changed_tensors=rows,full_FP32_master_state=full,full_state_sha256=tree_hash(full),reconstructed_full_state_sha256=tree_hash(reconstructed),
       effective_bf16_tensor_sha256=bf16,model_queries=0,optimizer_transfer=False,residual_addition=False,transfer_quantization=False,
       quality_dtype_cast='Native BF16 inference hash is separately recorded; FP32 master transfer is exact.')
    (output/'EXPORT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
