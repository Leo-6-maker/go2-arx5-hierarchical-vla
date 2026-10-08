"""Saved native consumption/modes and full Trainer readback; no extra model call."""
import argparse,json,math,os,sys,hashlib
from collections import Counter
from pathlib import Path
from contract import ROOT,sha,write


def audit(arm,step):
    import torch
    from safetensors import safe_open
    sys.path.append(str(ROOT.parent/'PI0283/code'))
    from scope import is_projector,digest
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(4)
    card=json.loads((ROOT/'design'/(arm+'_CARD.json')).read_text());folder=ROOT/'learning'/arm/'main'
    plan=[json.loads(x) for x in Path(card['plan_path']).read_text().splitlines()]
    assert step<=card['updates'] and step in (1024,2048,4096)
    def prefix(name):
        p=folder/name;lines=p.read_bytes().splitlines(keepends=True);assert len(lines)>=step*2
        blob=b''.join(lines[:step*2]);return [json.loads(x) for x in lines[:step*2]],dict(path=str(p),bytes=len(blob),lines=step*2,sha256=hashlib.sha256(blob).hexdigest())
    consumed,cproof=prefix('consumption.jsonl');forwards,fproof=prefix('native_training_forwards.jsonl');modes,mproof=prefix('native_loss_modes_LR.jsonl')
    detailed=0;visits=[]
    for i,(r,f,m) in enumerate(zip(consumed,forwards,modes)):
        assert r['step']==f['step']==m['phase_step']==i//2+1 and r['micro']==f['micro']==i%2
        assert r['cursor']==f['cursor']==m['cursor']==i*4 and r['visits']==f['visits']==plan[i*4:i*4+4]
        assert math.isfinite(r['loss']) and math.isclose(r['loss'],m['native_loss']/2,rel_tol=0,abs_tol=1e-12)
        assert m['native_reduction']['active_mask_sum']==1600
        assert len(r['native_noise_sha256'])==len(r['native_time_sha256'])==64 and len(r['native_time'])==4
        assert all(math.isfinite(x) for x in r['native_time'])
        proof=r['gradient']
        if 'row_module_gradient_norms' in proof:
            assert all(math.isfinite(x) and x>0 for x in proof['row_module_gradient_norms'].values())
            assert math.isfinite(proof['DiT_gradient_norm']) and proof['DiT_gradient_norm']>0;detailed+=1
        visits.extend(r['visits'])
    assert Counter(v['phase'] for v in visits)==Counter(v['phase'] for v in plan[:step*8])
    assert Counter(v['task'] for v in visits)==Counter(v['task'] for v in plan[:step*8])
    peaks=[max(m['actual_LRs'][i] for m in modes) for i in range(2)];assert peaks==card['peak_LRs']
    scope=json.loads((folder/'SCOPE.json').read_text());projectors=list(scope['source_rows']);assert len(projectors)==14
    cp=folder/f'checkpoint-{step}';export=ROOT/'delivery'/f'{arm}-{step}'/'EXPORT.json';saved=json.loads(export.read_text())
    recovery=json.loads((cp/'PI0282_RECOVERY.json').read_text());assert recovery['cursor']==step*8
    assert json.loads((cp/'trainer_state.json').read_text())['global_step']==step
    assert all((cp/n).is_file() for n in recovery['required_files'])
    for n,h in saved['model_files'].items():assert sha(cp/n)==h,n
    for n in ('processor_config.json','statistics.json','embodiment_id.json'):assert sha(cp/n)==sha(ROOT/'data'/n)
    base=Path(saved['original_base']);index=json.loads((base/'model.safetensors.index.json').read_text())['weight_map'];seen=set();changed_rows=[];changed_dit=0;frozen=0
    for shard in sorted(cp.glob('*.safetensors')):
        with safe_open(shard,framework='pt',device='cpu') as f:
            for name in f.keys():
                value=f.get_tensor(name);assert dict(dtype=str(value.dtype),shape=list(value.shape),sha256=digest(value))==saved['full_FP32_master_state'][name]
                with safe_open(base/index[name],framework='pt',device='cpu') as original:old=original.get_tensor(name).to(value.dtype)
                if is_projector(name):
                    assert value.shape[0]==32 and value.dtype==torch.float32
                    assert torch.equal(value[:10],old[:10]) and torch.equal(value[11:],old[11:])
                    if not torch.equal(value[10],old[10]):changed_rows.append(name)
                elif name.startswith('action_head.model.'):assert value.dtype==torch.float32;changed_dit+=not torch.equal(value,old)
                else:assert torch.equal(value,old),name;frozen+=1
                seen.add(name)
    assert seen==set(saved['full_FP32_master_state']) and changed_rows and changed_dit
    opt=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
    assert len(opt['param_groups'])==2;pg,dg=opt['param_groups'];assert pg['weight_decay']==0. and dg['weight_decay']==1e-5 and len(pg['params'])==14
    assert all(int(s['step'])==card['parent_global_update']+step for s in opt['state'].values())
    for name,idx in zip(projectors,pg['params']):
        for key in ('exp_avg','exp_avg_sq'):
            x=opt['state'][idx][key];assert x.shape[0]==32 and not torch.count_nonzero(x[:10]) and not torch.count_nonzero(x[11:]),(name,key)
    entries=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(cp.iterdir()) if p.is_file()]
    value=dict(status='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE',point=f'{arm}-{step}',checkpoint=str(cp),entries=entries,plan_sha256=sha(card['plan_path']),export_sha256=sha(export),journal_prefixes=dict(consumption=cproof,native_forward=fproof,native_modes=mproof),formal_microbatches=len(consumed),chunks=len(visits),task_counts=dict(Counter(v['task'] for v in visits)),phase_counts=dict(Counter(v['phase'] for v in visits)),detailed_gradient_microbatches=detailed,actual_LR_peaks=peaks,actual_module_modes_first=modes[0]['modules'],native_loss_first16_mean=sum(m['native_loss'] for m in modes[:32])/32,native_loss_last16_mean=sum(m['native_loss'] for m in modes[-32:])/32,changed_row10_tensors=changed_rows,changed_DiT_tensors=changed_dit,frozen_parameter_tensors=frozen,other31_rows_exact=True,other31_Adam_moments_zero=True,new_model_queries=0,new_optimizer_updates=0,new_physics=0,selected='NONE',deployment='NONE')
    target=ROOT/'evidence'/f'{arm}-{step}_TRAINING_CPU.json'
    if target.exists():assert json.loads(target.read_text())==json.loads(json.dumps(value))
    else:write(target,value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('entries','actual_module_modes_first','changed_row10_tensors')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args();audit(a.arm,a.step)
