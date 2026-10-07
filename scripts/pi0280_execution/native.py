"""Official frozen DROID source reproduction with actual paired noise and no learning."""
import argparse,copy,hashlib,importlib.metadata,json,os,shutil,sys,time,urllib.request
from pathlib import Path
from bind_pristine import ROOT,BASE,SOURCE,CODE_SHA,REVISION,RELEASE,digest

TAG='oxe_droid_relative_eef_relative_joint'
DATA=ROOT/'assets/droid_sample'
OUT=ROOT/'native'

def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2)+'\n')

def freeze(p,v):
    if p.exists():assert json.loads(p.read_text())==v,str(p)
    else:write(p,v)

def prepare():
    import numpy as np,pandas as pd
    from gr00t.utils.video_utils import get_all_frames
    assert json.loads((ROOT/'evidence/PRISTINE_FILES.json').read_text())['status']=='PRISTINE_LOCAL_FILES_VERIFIED'
    OUT.mkdir(parents=True,exist_ok=True)
    materialized=[]
    names=['meta/'+n for n in ['info.json','modality.json','episodes.jsonl','tasks.jsonl']]
    for episode in (1,2):
        names.append(f'data/chunk-000/episode_{episode:06d}.parquet')
        names.extend(f'videos/chunk-000/{view}/episode_{episode:06d}.mp4' for view in ['observation.images.exterior_1_left','observation.images.wrist_left'])
    for name in names:
        source=ROOT/'official/demo_data/droid_sample'/name;target=DATA/name;target.parent.mkdir(parents=True,exist_ok=True)
        content=source.read_bytes()
        if content.startswith(b'version https://git-lfs.github.com/spec/v1'):
            pointer=dict(line.split(' ',1) for line in content.decode().splitlines());expected=pointer['oid'].split(':')[1];size=int(pointer['size']);assert size<1<<30
            if not target.exists():
                url=f'https://media.githubusercontent.com/media/NVIDIA/Isaac-GR00T/{CODE_SHA}/demo_data/droid_sample/{name}'
                temporary=target.with_suffix(target.suffix+'.partial')
                with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as stream:shutil.copyfileobj(response,stream)
                assert temporary.stat().st_size==size and digest(temporary)==expected,name
                temporary.replace(target)
            assert target.stat().st_size==size and digest(target)==expected,name
        else:
            expected=hashlib.sha256(content).hexdigest()
            if not target.exists():target.write_bytes(content)
            assert digest(target)==expected,name
        materialized.append(dict(path=name,bytes=target.stat().st_size,sha256=expected,pinned_git_source=CODE_SHA))
    freeze(ROOT/'evidence/DROID_MATERIALIZATION.json',dict(files=materialized,total_bytes=sum(r['bytes'] for r in materialized),training_data_production=0,purpose='existing official source evaluation only'))
    info=json.loads((DATA/'meta/info.json').read_text());assert info['fps']==15
    episodes={r['episode_index']:r for r in map(json.loads,(DATA/'meta/episodes.jsonl').read_text().splitlines())}
    tasks={r['task_index']:r['task'] for r in map(json.loads,(DATA/'meta/tasks.jsonl').read_text().splitlines())}
    cfg=json.loads((BASE/'processor_config.json').read_text())['processor_kwargs']['modality_configs'][TAG]
    assert cfg['video']['delta_indices']==[-15,0] and cfg['action']['delta_indices']==list(range(40))
    video_map={'exterior_image_1_left':'observation.images.exterior_1_left','wrist_image_left':'observation.images.wrist_left'}
    rows=[];files=[]
    for episode in (1,2):
        path=DATA/f'data/chunk-000/episode_{episode:06d}.parquet';df=pd.read_parquet(path)
        assert len(df)==episodes[episode]['length']
        anchors=np.linspace(15,len(df)-40,16,dtype=int).tolist();assert len(set(anchors))==16
        files.append(dict(path=str(path),sha256=digest(path)))
        frames={}
        for key,name in video_map.items():
            p=DATA/f'videos/chunk-000/{name}/episode_{episode:06d}.mp4';files.append(dict(path=str(p),sha256=digest(p)))
            values,timestamps=get_all_frames(str(p),decoder_kwargs={'num_ffmpeg_threads':4})
            assert len(values)==len(df) and np.allclose(timestamps,np.arange(len(df))/15,atol=1e-5),(name,len(values),len(df))
            frames[key]=values
        states=np.stack(df['observation.state']).astype(np.float32);actions=np.stack(df['action']).astype(np.float32)
        assert states.shape==(len(df),17) and actions.shape==(len(df),17)
        for anchor in anchors:
            identity=f'episode{episode}-frame{anchor:04d}';p=OUT/'inputs'/f'{identity}.npz';p.parent.mkdir(exist_ok=True)
            arrays={k:np.stack([v[anchor-15],v[anchor]]) for k,v in frames.items()}
            arrays.update(state=states[anchor],descriptive_future_action=actions[anchor:anchor+40],actual_history_frame_indices=np.array([anchor-15,anchor]),actual_history_timestamps_s=np.array([anchor-15,anchor])/15)
            if not p.exists():np.savez_compressed(p,**arrays)
            else:
                with np.load(p) as saved:assert set(saved.files)==set(arrays) and all(np.array_equal(saved[k],v) for k,v in arrays.items())
            task=tasks[int(df.iloc[anchor]['task_index'])]
            rows.append(dict(id=identity,episode=episode,frame=anchor,language=task,input_path=str(p),input_sha256=digest(p)))
    card=dict(release=RELEASE,official_code=CODE_SHA,model_revision=REVISION,source_tag=TAG,fps=15,action_horizon=40,action_period_s=1/15,history_frame_deltas=[-15,0],video_key_binding=video_map,state_slices=dict(eef_9d=[0,9],gripper_position=[9,10],joint_position=[10,17]),selection='16 equally spaced valid anchors per real episode, inclusive [15, length-40], fixed before outputs',sources=files,rows=rows,actual_queries_planned=64,training_updates=0)
    freeze(ROOT/'design/NATIVE_CARD.json',card)
    print('NATIVE_CARD_FROZEN32; REAL_HISTORY; ZERO_MODEL_QUERIES',flush=True)

def array_hash(a):
    return hashlib.sha256(a.tobytes()).hexdigest()

def reproduce():
    import numpy as np,torch
    from safetensors import safe_open
    from transformers import AutoModel
    import gr00t.policy.gr00t_policy as official
    card=json.loads((ROOT/'design/NATIVE_CARD.json').read_text())
    assert card['release']==RELEASE and not (OUT/'RESULT.json').exists(),'Do not repeat completed source panel'
    assert str(Path(official.__file__).resolve()).startswith(str(ROOT/'official')+'/')
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    # Record Hugging Face's actual loading information while retaining the stock constructor.
    original_load=AutoModel.from_pretrained;loading=[]
    def audited_load(*args,**kwargs):
        value,detail=original_load(*args,**kwargs,output_loading_info=True)
        loading.append(detail);return value
    AutoModel.from_pretrained=audited_load
    try:policy=official.Gr00tPolicy(TAG,str(BASE),device='cuda:0',strict=True)
    finally:AutoModel.from_pretrained=original_load
    assert len(loading)==1 and not any(loading[0].get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),loading
    policy.model.requires_grad_(False);policy.model.eval()
    assert all(not p.requires_grad for p in policy.model.parameters())
    weights={};params=dict(policy.model.named_parameters());index=json.loads((BASE/'model.safetensors.index.json').read_text())['weight_map']
    for shard in sorted(set(index.values())):
        with safe_open(BASE/shard,framework='pt',device='cpu') as handle:
            for key in handle.keys():
                if key not in params:continue
                actual=params[key].detach().cpu();expected=handle.get_tensor(key).to(actual.dtype)
                assert torch.equal(actual,expected),key
                weights[key]=array_hash(actual.contiguous().view(torch.uint8).numpy())
    assert set(weights)==set(params),(set(params)-set(weights))
    write(ROOT/'evidence/WEIGHT_LOADING.json',dict(status='PRISTINE_EFFECTIVE_WEIGHTS_VERIFIED',loading=loading,parameters=len(params),parameter_hashes=weights,trainable_parameters=0,official_policy_file=official.__file__))
    binding=dict(release=RELEASE,official_source=CODE_SHA,model_revision=REVISION,model_root=str(BASE),policy_file=official.__file__,policy_sha256=digest(Path(official.__file__)),native_card_sha256=digest(ROOT/'design/NATIVE_CARD.json'),wrapper_sha256=digest(Path(__file__)),precision='stock Gr00tPolicy BF16 model and floating inputs; stock FP32 decode',native_steps=policy.model.action_head.num_inference_timesteps,action_horizon=policy.model.action_head.action_horizon,device=torch.cuda.get_device_name(),dependencies={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','huggingface-hub','safetensors']},new_training_updates=0,selected_checkpoint='NONE',deployment='NONE')
    assert binding['native_steps']==4 and binding['action_horizon']==40
    freeze(ROOT/'design/BASELINE_BINDING.json',binding)
    ledger=ROOT/'evidence/model_budget.jsonl';ledger.parent.mkdir(parents=True,exist_ok=True)
    previous=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    counts={k:sum(x['kind']==k for x in previous) for k in ('policy_attempt','action_head')}
    def charge(kind,identity):
        assert counts[kind]<(1024 if kind=='policy_attempt' else 4096)
        with ledger.open('a') as f:f.write(json.dumps(dict(kind=kind,id=identity,utc_ns=time.time_ns()))+'\n');f.flush();os.fsync(f.fileno())
        counts[kind]+=1
    active={'id':None};hook=policy.model.action_head.model.register_forward_pre_hook(lambda *args:charge('action_head',active['id']))
    original_predict=policy.model.get_action;raw=[]
    def capture_prediction(*args,**kwargs):
        assert 'action' not in kwargs.get('inputs',{}),'No future labels or RTC input'
        value=original_predict(*args,**kwargs);raw.append(value['action_pred'].detach().cpu().float().numpy());return value
    policy.model.get_action=capture_prediction
    report=[]
    for row in card['rows']:
        result_dir=OUT/'outputs'/row['id'];result_dir.mkdir(parents=True,exist_ok=True)
        if (result_dir/'PAIR.json').exists():report.append(json.loads((result_dir/'PAIR.json').read_text()));continue
        assert digest(Path(row['input_path']))==row['input_sha256']
        with np.load(row['input_path']) as data:
            state=data['state'];observation=dict(video={k:data[k][None] for k in card['video_key_binding']},state={k:state[a:b][None,None] for k,(a,b) in card['state_slices'].items()},language={policy.language_key:[[row['language']]]})
            truth=data['descriptive_future_action'].copy()
        outputs=[];noise=None;timings=[];raw=[]
        noise_path=result_dir/'actual_initial_noise.npy'
        if noise_path.exists():noise=torch.from_numpy(np.load(noise_path)).view(torch.bfloat16).cuda()
        for lane in ('stock','thin_wrapper'):
            active['id']=row['id']+'/'+lane
            if (result_dir/f'{lane}.npz').exists():
                with np.load(result_dir/f'{lane}.npz') as saved:
                    outputs.append({k[len('decoded_'):]:saved[k].copy() for k in saved.files if k.startswith('decoded_')});raw.append(saved['raw'].copy())
                    noise=torch.from_numpy(saved['noise_uint16'].copy()).view(torch.bfloat16).cuda();timings.append(float(saved['latency_s']))
                continue
            charge('policy_attempt',active['id']);original_randn=torch.randn;noise_calls=[]
            def actual_noise(*args,**kwargs):
                nonlocal noise
                if noise is None:
                    noise=original_randn(*args,**kwargs)
                    assert noise.dtype==torch.bfloat16
                    np.save(noise_path,noise.detach().cpu().view(torch.uint16).numpy())
                assert tuple(kwargs['size'])==tuple(noise.shape) and kwargs['dtype']==noise.dtype and torch.device(kwargs['device'])==noise.device
                noise_calls.append(1);return noise.clone()
            torch.randn=actual_noise;started=time.perf_counter()
            try:
                with torch.inference_mode():decoded,_=policy.get_action(copy.deepcopy(observation)) if lane=='stock' else serve(policy,observation)
                torch.cuda.synchronize()
            finally:torch.randn=original_randn
            elapsed=time.perf_counter()-started;assert len(noise_calls)==1 and all(np.isfinite(x).all() for x in decoded.values())
            np.savez_compressed(result_dir/f'{lane}.npz',raw=raw[-1],noise_uint16=noise.detach().cpu().view(torch.uint16).numpy(),latency_s=np.array(elapsed),**{'decoded_'+k:v for k,v in decoded.items()})
            outputs.append(decoded);timings.append(elapsed)
        assert np.array_equal(raw[0],raw[1]) and all(np.array_equal(outputs[0][k],outputs[1][k]) for k in outputs[0]),row['id']
        decoded=np.concatenate([outputs[0][k][0] for k in ['eef_9d','gripper_position','joint_position']],axis=-1)
        pair=dict(id=row['id'],status='PASS_EXACT_SAME_RUNTIME_PARITY',noise_sha256=array_hash(noise.detach().cpu().view(torch.uint16).numpy()),native_dtype=str(noise.dtype),raw_shape=list(raw[0].shape),decoded_shapes={k:list(v.shape) for k,v in outputs[0].items()},latencies_s=timings,descriptive_mse_by_key={k:float(np.mean((decoded[:,a:b]-truth[:,a:b])**2)) for k,(a,b) in card['state_slices'].items()},task_completion='NOT_MEASURED_OFFICIAL_OPEN_LOOP')
        write(result_dir/'PAIR.json',pair);report.append(pair);print(row['id'],pair['status'],flush=True)
    hook.remove()
    for key,p in params.items():assert array_hash(p.detach().cpu().contiguous().view(torch.uint8).numpy())==weights[key],key
    write(OUT/'RESULT.json',dict(status='NATIVE_SOURCE_INFERENCE_REPRODUCED',rows=report,completed_pairs=len(report),actual_counts=counts,effective_weights_unchanged=True,new_training_updates=0,task_success_claim=False))
    print('NATIVE_SOURCE_INFERENCE_REPRODUCED',counts,flush=True)

def serve(policy,observation):
    """Only transport isolation; stock validation, preprocessing, model and decode."""
    return policy.get_action(copy.deepcopy(observation))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','reproduce']);args=parser.parse_args()
    prepare() if args.phase=='prepare' else reproduce()
