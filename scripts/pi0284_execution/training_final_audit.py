"""Read-only full native Trainer/consumption/scope audit; no neural forward."""
import argparse,json,math,os
from collections import Counter
from pathlib import Path
import torch
from safetensors import safe_open
from learning import file_digest
from scope import is_projector,digest


def audit(root,stage):
    root=Path(root);torch.set_num_threads(4)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert stage in ('CTRL2048','LR5X2048')
    card=json.loads((root/f'design/{stage}_CARD.json').read_text());total=card['updates']
    folder=root/f'learning/{stage}/main'
    journal=json.loads((folder/'JOURNAL.json').read_text())
    rows=[json.loads(x) for x in (folder/'consumption.jsonl').read_text().splitlines()]
    forwards=[json.loads(x) for x in (folder/'native_training_forwards.jsonl').read_text().splitlines()]
    plan=[json.loads(x) for x in (Path(card['plan_path'])).read_text().splitlines()]
    assert len(rows)==len(forwards)==total*2 and len(plan)==total*8
    for key in ('consumption','native_training_forwards'):assert file_digest(journal[key]['path'])==journal[key]['sha256']
    visits=[];detailed=0
    for i,(r,f) in enumerate(zip(rows,forwards)):
        assert r['step']==f['step']==i//2+1 and r['micro']==f['micro']==i%2
        assert r['cursor']==f['cursor']==i*4 and r['visits']==f['visits']==plan[i*4:i*4+4]
        assert math.isfinite(r['loss']) and len(r['native_noise_sha256'])==len(r['native_time_sha256'])==64
        assert len(r['native_time'])==4 and all(math.isfinite(x) for x in r['native_time'])
        proof=r['gradient']
        if 'row_module_gradient_norms' in proof:
            assert all(math.isfinite(x) and x>0 for x in proof['row_module_gradient_norms'].values())
            assert math.isfinite(proof['DiT_gradient_norm']) and proof['DiT_gradient_norm']>0;detailed+=1
        visits+=r['visits']
    assert Counter(v['phase'] for v in visits)==dict.fromkeys(range(4),total*2)
    scope=json.loads((folder/'SCOPE.json').read_text());projectors=list(scope['source_rows']);assert len(projectors)==14
    print('PASS_ORDER_NATIVE_INPUT_NOISE_TIME_AND_PHASE_COUNTS',flush=True)
    checkpoints=[]
    for step in (1024,2048):
        print('READ_ONLY_CHECKPOINT',stage,step,flush=True)
        cp=folder/f'checkpoint-{step}';export=root/f'delivery/{stage}{step}/EXPORT.json'
        record=json.loads(export.read_text());recovery=json.loads((cp/'PI0282_RECOVERY.json').read_text())
        assert recovery['cursor']==step*8 and json.loads((cp/'trainer_state.json').read_text())['global_step']==step
        assert all((cp/n).is_file() for n in recovery['required_files'])
        for n,h in record['model_files'].items():assert file_digest(cp/n)==h,n
        for n in ('processor_config.json','statistics.json','embodiment_id.json'):assert file_digest(cp/n)==file_digest(root/'data'/n)
        base=Path(record['original_base']);index=json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
        seen=set();changed_projectors=[];changed_dit=0;frozen=0
        for shard in sorted(cp.glob('*.safetensors')):
            with safe_open(shard,framework='pt',device='cpu') as f:
                for n in f.keys():
                    value=f.get_tensor(n);meta=record['full_FP32_master_state'][n]
                    assert dict(dtype=str(value.dtype),shape=list(value.shape),sha256=digest(value))==meta
                    with safe_open(base/index[n],framework='pt',device='cpu') as original:old=original.get_tensor(n).to(value.dtype)
                    if is_projector(n):
                        assert value.shape[0]==32 and value.dtype==torch.float32
                        assert torch.equal(value[:10],old[:10]) and torch.equal(value[11:],old[11:]),('NON10_CHANGED',n)
                        if not torch.equal(value[10],old[10]):changed_projectors.append(n)
                    elif n.startswith('action_head.model.'):
                        assert value.dtype==torch.float32;changed_dit+=not torch.equal(value,old)
                    else:assert torch.equal(value,old),('FROZEN_PRETRAIN_CHANGED',n);frozen+=1
                    seen.add(n);del value,old
        assert seen==set(record['full_FP32_master_state']) and changed_projectors and changed_dit
        opt=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
        assert len(opt['param_groups'])==2
        pgroup,dgroup=opt['param_groups'];assert pgroup['weight_decay']==0. and dgroup['weight_decay']==1e-5
        assert len(pgroup['params'])==14 and opt['state']
        for state in opt['state'].values():assert int(state['step'])==card['parent_global_update']+step
        for name,idx in zip(projectors,pgroup['params']):
            for key in ('exp_avg','exp_avg_sq'):
                x=opt['state'][idx][key];assert x.shape[0]==32 and not torch.count_nonzero(x[:10]) and not torch.count_nonzero(x[11:]),(name,key)
        optimizer_parameters=len(opt['state']);del opt,x
        entries=[dict(path=p.name,bytes=p.stat().st_size,sha256=file_digest(p)) for p in sorted(cp.iterdir()) if p.is_file()]
        checkpoints.append(dict(step=step,path=str(cp),complete_Trainer=True,entries=entries,frozen_parameter_tensors=frozen,
            row10_changed_tensors=changed_projectors,DiT_changed_tensors=changed_dit,optimizer_parameters=optimizer_parameters,
            other31_rows_exact_original=True,other31_Adam_moments_zero=True,full_master_state_sha256=record['full_state_sha256']))
    updates=[json.loads(x) for x in (root/'learning/completed_updates.jsonl').read_text().splitlines()]
    counts=Counter(r['work_kind'] for r in updates);assert counts.get('canary',0)==32 and counts.get('formal',0)<=4096 and len(updates)<=4208
    result=dict(status='PASS_ACTUAL_NATIVE_CONTINUATION_FULL_TRAINERS_ORDER_SCOPE_AND_MOMENTS_CPU',stage=stage,parent_global_update=card['parent_global_update'],checkpoints=checkpoints,
        completed_updates=len(updates),kind_counts=dict(counts),formal_microbatches=len(rows),formal_chunks=len(visits),
        phase_counts=dict(Counter(v['phase'] for v in visits)),detailed_gradient_microbatches=detailed,
        actual_per_step_mode='OBSERVED_IN_NATIVE_FORWARD_HOOK',native_input_noise_time_hashes_present=True,
        new_model_queries=0,new_optimizer_updates=0,new_physics_steps=0,selected='NONE',deployment='NONE')
    modes=[json.loads(x) for x in (folder/'native_loss_modes_LR.jsonl').read_text().splitlines()]
    assert len(modes)==len(rows)
    for r,m in zip(rows,modes):
        assert m['phase_step']==r['step'] and m['cursor']==r['cursor']
        assert math.isclose(r['loss'],m['native_loss']/2,rel_tol=0,abs_tol=1e-12)
        assert m['native_reduction']['active_mask_sum']==1600
    result['actual_LR_peaks']=[max(m['actual_LRs'][i] for m in modes) for i in range(2)]
    assert result['actual_LR_peaks']==card['peak_LRs']
    result['actual_module_modes_first']=modes[0]['modules']
    result['native_loss_first16_mean']=sum(m['native_loss'] for m in modes[:32])/32
    result['native_loss_last16_mean']=sum(m['native_loss'] for m in modes[-32:])/32
    result['unique_uids']=len({v['uid'] for v in visits})
    result['repeat_draws']=len(visits)-result['unique_uids']
    out=root/f'runtime/{stage}_TRAINING_FINAL_CPU_AUDIT.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out)
    print(json.dumps({k:v for k,v in result.items() if k!='checkpoints'}));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',required=True,choices=('CTRL2048','LR5X2048'));a=p.parse_args();audit(a.root,a.stage)
