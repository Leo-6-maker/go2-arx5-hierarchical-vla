"""Actual Hopper processors, all first-2048 R inputs; no model construction."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4); torch.set_num_interop_threads(1)
from train import ROOT,paired,read,sha,Pairs,seed_for
from pi0260_r3_inputs import build,identity
import pi0260_r2_codec as codec

def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30)); os.nice(19)
    if hasattr(os,'sched_getaffinity'): os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    call=torch.nn.Module.__call__
    def guard(self,*a,**kw):
        if type(self).__module__.startswith('torchvision.transforms'): return call(self,*a,**kw)
        raise RuntimeError('PI0269_CPU_NO_NEURAL_QUERY')
    torch.nn.Module.__call__=guard
    def forbidden(*a,**kw): raise RuntimeError('PI0269_CPU_NO_OPTIMIZER')
    torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    card=read(ROOT/'design/APPROVED_RUN_CARD.json')
    assert sha(ROOT/'PAIRING_PLAN.json')==card['source_plan_sha256']
    stats=ROOT/'contracts/ROT/source/dataset/statistics.json';assert sha(stats)==card['statistics_sha256']
    processor=build(read(stats),paired.COSMOS)
    folder=ROOT/'evidence/input-audit/common';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'sample_audit.jsonl').exists()
    dataset=Pairs(processor,folder,'COMMON',0);keys=set();ranges={};count=0
    old=[json.loads(s) for s in (ROOT.with_name('PI0268')/'ROT/main/training/sample_audit.jsonl').read_text().splitlines()]
    for k,value in enumerate(dataset):
        inputs=value['inputs'];unit=k//2+1;micro=k%2
        meta=value['_pi0264_meta'];assert (meta['unit'],meta['micro'])==(unit,micro)
        condition={name:v for name,v in inputs.items() if name not in ('action','action_mask')}
        key=identity(condition);keys.add(key)
        if k<len(old):
            assert key==old[k]['condition_sha256']
            assert identity({n:inputs[n] for n in ('action','action_mask')})==old[k]['labels_masks_sha256']
        for name in ('state','action'):
            x=inputs[name].detach().cpu().numpy();assert x.dtype==np.float32 and np.isfinite(x).all()
            assert x.shape[-1]==132 and not x[...,10:].any()
            assert abs(x[...,3:9]).max()<=1+2e-6
            maximum=ranges.setdefault(name,dict(abs_max=0.,rotation_abs_max=0.))
            maximum['abs_max']=max(maximum['abs_max'],float(abs(x).max()))
            maximum['rotation_abs_max']=max(maximum['rotation_abs_max'],float(abs(x[...,3:9]).max()))
        assert inputs['action'].shape==(1,40,132)
        mask=inputs['action_mask'];assert torch.all(mask[...,:10]==1) and not torch.any(mask[...,10:])
        assert dataset.plan['units'][unit-1]['seed']==seed_for(unit)
        count+=1
        if count%512==0:print('COMMON_CPU_INPUTS',count,flush=True)
    assert count==8192
    consumed=[json.loads(s) for s in (folder/'sample_audit.jsonl').read_text().splitlines()]
    plan=dataset.plan;terminal=0;strata={}
    for row in consumed:
        unit=plan['units'][row['unit']-1]
        source_rows=dataset.family(unit['family'])[2];raw=source_rows[row['index']]
        terminal+=int(raw['terminal']);key=str((row['task'],row['slot'],raw['quartile'],raw['variant'],raw['terminal']))
        strata[key]=strata.get(key,0)+1
    actual={(r['family'],r['index'],r['language']) for r in consumed}
    fit=read(ROOT/'design/FIT112_FIRST2048.json')['rows']
    assert len(fit)==112 and all((r['family'],r['index'],r['language']) in actual for r in fit)
    result=dict(status='PASS_PI0269_HOPPER_ACTUAL_PROCESSOR_BOUNDARY',microbatches=count,distinct_conditions=len(keys),
        ranges=ranges,statistics_sha256=sha(stats),sample_audit_sha256=sha(folder/'sample_audit.jsonl'),
        previous_ROT_first2048_actual_bit_identity=True,terminal_microbatches=terminal,strata=strata,FIT112_full_plan_seen=True,
        model_queries=0,optimizer_updates=0,script_sha256=sha(__file__))
    target=ROOT/'evidence/HOPPER_CPU_ADMISSION.json'; assert not target.exists()
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result),flush=True)

if __name__=='__main__': main()
