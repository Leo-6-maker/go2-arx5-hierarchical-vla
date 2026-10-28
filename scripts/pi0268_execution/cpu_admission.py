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
        raise RuntimeError('PI0268_CPU_NO_NEURAL_QUERY')
    torch.nn.Module.__call__=guard
    def forbidden(*a,**kw): raise RuntimeError('PI0268_CPU_NO_OPTIMIZER')
    torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    card=read(ROOT/'design/APPROVED_RUN_CARD.json')
    assert sha(ROOT/'PAIRING_PLAN.json')==card['source_plan_sha256']
    expected=[json.loads(s) for s in (ROOT.with_name('PI0267')/'R/main/training/sample_audit.jsonl').read_text().splitlines()][:4096]
    assert len(expected)==4096
    old_packets=[]; conditions={}; results={}
    for arm in ('OLD','ROT'):
        stats=ROOT/'contracts'/arm/'source/dataset/statistics.json'; assert sha(stats)==card['contracts'][arm]
        processor=build(read(stats),paired.COSMOS)
        folder=ROOT/'evidence/input-audit'/arm; folder.mkdir(parents=True,exist_ok=True)
        assert not (folder/'sample_audit.jsonl').exists(), 'do not overwrite prior CPU audit'
        dataset=Pairs(processor,folder,arm,0); keys=set(); ranges={}; count=0
        for k,value in enumerate(dataset):
            inputs=value['inputs']; exp=expected[k]; unit=k//2+1; micro=k%2
            assert value['_pi0264_meta']['unit']==unit and value['_pi0264_meta']['micro']==micro
            assert exp['unit']==unit and exp['micro']==micro
            cond={key:v for key,v in inputs.items() if key not in ('action','action_mask')}
            label={key:inputs[key] for key in ('action','action_mask')}
            key=identity(cond); keys.add(key)
            unchanged={key:v for key,v in inputs.items() if key not in ('state','action')}
            partial={key:inputs[key][...,[0,1,2,9]] for key in ('state','action')}
            signature=(identity(unchanged),identity(partial))
            if arm=='OLD':
                assert key==exp['condition_sha256'],('OLD_CONDITION_IDENTITY',unit,micro)
                assert identity(label)==exp['labels_masks_sha256'],('OLD_LABEL_IDENTITY',unit,micro)
                old_packets.append(signature)
            else: assert signature==old_packets[k],('UNINTENDED_INPUT_CHANGE',unit,micro)
            for name in ('state','action'):
                x=inputs[name].detach().cpu().numpy(); assert x.dtype==np.float32 and np.isfinite(x).all()
                assert x.shape[-1]==132 and not x[...,10:].any()
                if name=='action': assert x.shape[-2]==40
                if arm=='ROT': assert abs(x[...,3:9]).max()<=1+2e-6
                active=x[...,:10]; r=ranges.setdefault(name,dict(min=float('inf'),max=-float('inf'),rotation_abs_max=0.))
                r['min']=min(r['min'],float(active.min()));r['max']=max(r['max'],float(active.max()));r['rotation_abs_max']=max(r['rotation_abs_max'],float(abs(x[...,3:9]).max()))
            mask=inputs['action_mask']; assert torch.all(mask[...,:10]==1) and not torch.any(mask[...,10:])
            assert dataset.plan['units'][unit-1]['seed']==seed_for(unit)
            count+=1
            if count%512==0: print(arm,'CPU_INPUTS',count,flush=True)
        assert count==4096
        results[arm]=dict(microbatches=count,distinct_conditions=len(keys),ranges=ranges,statistics_sha256=sha(stats),sample_audit_sha256=sha(folder/'sample_audit.jsonl'))
        dataset.family.cache_clear(); del dataset,processor
    assert len(old_packets)==4096
    donor=read(ROOT/'DONOR_LOCAL_READBACK.json')
    result=dict(status='PASS_PI0268_HOPPER_ACTUAL_PROCESSOR_BOUNDARY',contracts=results,OLD_bit_identity=True,
        unchanged_pixels_tokens_raw_labels_XYZ_gripper_masks=True,seed_algorithm='original big endian',
        donor_revision=donor['revision'],model_queries=0,optimizer_updates=0,script_sha256=sha(__file__))
    target=ROOT/'evidence/HOPPER_CPU_ADMISSION.json'; assert not target.exists()
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result),flush=True)

if __name__=='__main__': main()
