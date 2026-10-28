"""Finite CPU and H200 qualification, exact comparisons, then matched training."""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path
import cpu_inputs as data

def child(root,script,*args):
    command=[sys.executable,'-u',str(Path(__file__).with_name(script)),*map(str,args)]
    print('START',script,*args,flush=True)
    subprocess.run(command,check=True)

def cursor_check(root):
    import torch
    from gripper import check
    assert os.environ.get('PBS_JOBID') and not torch.cuda.is_initialized()
    ledger=data.ItemLedger(root/'runtime/cpu_items.jsonl',torch.multiprocessing.get_context('spawn'))
    ds=data.Inputs(root,'FAST_CACHE2',ledger,'ordered_cursor_interruption')
    checked=[]
    for start in (0,8,256,4096,8190,8192):
        # First queue is deliberately discarded after one delivered microbatch.
        loader=data.loader(ds,start=start,pin_memory=False);iterator=iter(loader)
        first=None if start==8192 else next(iterator)
        del iterator,loader
        loader=data.loader(ds,start=start,stop=min(8192,start+2),pin_memory=False);actual=list(loader);del loader
        assert [v['_pi0276_visit']['micro_index'] for v in actual]==list(range(start,min(8192,start+2)))
        if first is not None:
            ds.initialize();assert ds.helper.identity(first)==ds.helper.identity(actual[0])
        checked.append(dict(next_unconsumed_micro=start,delivered=len(actual),discarded_queue_recreated=True))
    assert not torch.cuda.is_initialized()
    data.write(root/'evidence/CURSOR_CHECK.json',dict(status='PASS_ACTUAL_ORDERED_PREFETCH_INTERRUPTION',checks=checked,cumulative_items=ledger.counter.value))
    data.write(root/'evidence/GRIPPER_CPU_CHECK.json',check())

def cpu(root):
    import torch
    assert os.environ.get('PBS_JOBID') and not torch.cuda.is_initialized()
    ledger=data.ItemLedger(root/'runtime/cpu_items.jsonl',torch.multiprocessing.get_context('spawn'))
    started=time.perf_counter();data.build_cache(root,ledger)
    data.write(root/'evidence/CACHE_COST.json',dict(seconds=time.perf_counter()-started,kind='END_TO_END_CACHE_BUILD_OR_VERIFIED_REUSE',items=ledger.counter.value))
    if not (root/'evidence/CURSOR_CHECK.json').exists():child(root,'qualify.py','--root',root,'--stage','cursor')
    for route in ('REF_W0','FAST_CACHE0','FAST_CACHE2'):
        for trial in (0,1):
            if not (root/'evidence/cpu_bench'/f'{route}-{trial}.json').exists():child(root,'bench_cpu.py','--root',root,'--route',route,'--trial',trial)
    rates={r:statistics.median(data.read(root/'evidence/cpu_bench'/f'{r}-{t}.json')['items_per_second'] for t in (0,1)) for r in ('REF_W0','FAST_CACHE0','FAST_CACHE2')}
    route=max(('FAST_CACHE2','FAST_CACHE0'),key=lambda r:rates[r])
    data.write(root/'evidence/CPU_SELECTION.json',dict(status='PASS_ACTUAL_BOUNDED_CPU_MEASUREMENTS',route=route,rates=rates,H200_speed='NOT_YET_OBSERVED'))

def segment(root,arm,phase,route,limit,resume=0):
    from train import Binding
    b=Binding(root,arm,phase,route,limit,resume);done=b.output/'SEGMENT_COMPLETE.json'
    if done.exists() and data.read(done)['limit']==limit:return b.output
    child(root,'train.py','--root',root,'--arm',arm,'--phase',phase,'--route',route,'--limit',limit,'--resume',resume)
    assert data.read(done)['limit']==limit
    return b.output

def logs_equal(a,b,count):
    from train import lines
    for name in ('sample_audit.jsonl','step_rng.jsonl','full_loss.jsonl','temporal_components.jsonl','gripper_objective.jsonl','gripper_generated_error.jsonl','gripper_native_loss.jsonl'):
        assert lines(a/name)[:2*count]==lines(b/name)[:2*count],('EXACT_LOG_MISMATCH',name)
    assert lines(a/'update_gradient_norms.jsonl')[:count]==lines(b/'update_gradient_norms.jsonl')[:count]
    for p in (a,b):assert data.read(p/'S4_parity.json')['status']=='PASS_PI0247_U_S4_PARITY'
    assert data.read(a/'S4_first_update_gradients.json')==data.read(b/'S4_first_update_gradients.json')

def bench_compare(root,route,trial):
    from train import Binding,lines
    a,b=(Binding(root,'CONTROL',f'bench{trial}',r,96).output for r in ('REF_W0',route))
    logs_equal(a,b,96)
    for step in (8,96):
        assert data.read(a/f'SNAPSHOT{step}.json')==data.read(b/f'SNAPSHOT{step}.json'),('EXACT_MODEL_OPTIMIZER_RNG_SCHEDULER',step)
        assert data.read(a/f'GRADIENTS{step}.json')==data.read(b/f'GRADIENTS{step}.json'),('EXACT_GRADIENTS',step)
    assert lines(a/'OUTPUT_HASHES.jsonl')==lines(b/'OUTPUT_HASHES.jsonl')
    def timing(p):
        rows=lines(p/'TIMINGS.jsonl');assert [r['unit'] for r in rows]==list(range(1,97))
        selected=rows[16:];assert all(r['completed_seconds']>0 for r in selected)
        return dict(median_seconds=statistics.median(r['completed_seconds'] for r in selected),completed_updates=80,
                    mean_seconds=statistics.mean(r['completed_seconds'] for r in selected),max_process_tree_RSS_bytes=max(r['process_tree_RSS_bytes'] for r in rows),
                    peak_VRAM_bytes=max(r['peak_reserved_bytes'] for r in rows))
    ref,fast=timing(a),timing(b);gain=ref['median_seconds']/fast['median_seconds']-1
    result=dict(status='PASS_EXACT_ACTUAL_H200_EQUIVALENCE',route=route,trial=trial,reference=ref,candidate=fast,throughput_gain=gain,
        model_optimizer_gradients_outputs_rng_scheduler='EXACT_BYTE_IDENTITY',microbatches=192,
        hashes={str(p.relative_to(root)):data.sha(p) for base in (a,b) for p in base.glob('SNAPSHOT*.json')})
    data.write(root/'evidence/h200'/f'{route}-{trial}.json',result)
    return result

def h200(root):
    candidate=data.read(root/'evidence/CPU_SELECTION.json')['route']
    for route,trial in [('REF_W0',0),(candidate,0),(candidate,1),('REF_W0',1)]:segment(root,'CONTROL',f'bench{trial}',route,96)
    comparisons={candidate:[bench_compare(root,candidate,t) for t in (0,1)]}
    def passes(rows):return all(r['throughput_gain']>0 for r in rows) and statistics.median(r['throughput_gain'] for r in rows)>=.10
    if not passes(comparisons[candidate]):
        other='FAST_CACHE0' if candidate=='FAST_CACHE2' else 'FAST_CACHE2'
        for t in (0,1):segment(root,'CONTROL',f'bench{t}',other,96)
        comparisons[other]=[bench_compare(root,other,t) for t in (0,1)]
    medians={'REF_W0':statistics.median(r['reference']['median_seconds'] for r in comparisons[candidate])}
    medians.update({r:statistics.median(x['candidate']['median_seconds'] for x in rows) for r,rows in comparisons.items()})
    selected=min(medians,key=medians.get)
    # An equivalent REF fallback is expressly allowed when cache speed is unestablished.
    result=dict(status='H200_SPEEDUP_PASS' if any(passes(v) for v in comparisons.values()) else 'SPEEDUP_NOT_ESTABLISHED',
        route=selected,median_seconds=medians,equivalence='PASS',raw_updates=sum(1 for _ in (root/'raw_completed_updates.jsonl').open()),
        comparison_receipts=[str(p.relative_to(root)) for p in (root/'evidence/h200').glob('*.json')])
    data.write(root/'evidence/H200_SELECTION.json',result)

def resume_compare(root,arm,route):
    import torch
    from train import Binding, fingerprint
    sys.path[:0]=data.read(root/'design/INPUT_BINDING.json')['module_paths']
    from pi0260_r3_compare_canary import model_delta,optimizer_delta
    a=Binding(root,arm,'bench0' if arm=='CONTROL' else 'continuous',route,96 if arm=='CONTROL' else 8).output
    b=Binding(root,arm,'split',route,8).output
    logs_equal(a,b,8)
    y=b/'checkpoint-8'
    if arm=='CONTROL':
        # REF may be the fastest equivalent route; compare its recorded continuous-eight
        # fingerprint to the split checkpoint without a new continuous query.
        snapshot=data.read(a/'SNAPSHOT8.json')
        from safetensors.torch import load_file
        tensors={}
        for file in sorted(y.glob('model-*.safetensors')):tensors.update({k:fingerprint(v) for k,v in load_file(str(file)).items()})
        assert tensors==snapshot['model']
        assert fingerprint(torch.load(y/'optimizer.pt',map_location='cpu',weights_only=False))==snapshot['optimizer']
        assert fingerprint(torch.load(y/'scheduler.pt',map_location='cpu',weights_only=False))==snapshot['scheduler']
        rng=torch.load(y/'rng_state.pth',map_location='cpu',weights_only=False)
        assert fingerprint(rng)==snapshot['rng']
        md=od=0
    else:
        x=a/'checkpoint-8';md,_=model_delta(x,y);od,_=optimizer_delta(x,y);assert md==od==0
        for name in ('scheduler.pt','rng_state.pth'):assert fingerprint(torch.load(x/name,map_location='cpu',weights_only=False))==fingerprint(torch.load(y/name,map_location='cpu',weights_only=False))
        assert data.read(a/'ISOLATED_GRIPPER_GRADIENT.json')==data.read(b/'ISOLATED_GRIPPER_GRADIENT.json')
    data.write(root/arm/'canary_compare.json',dict(status='PASS_PI0277_EXACT_RESUME',arm=arm,route=route,model_max_abs=md,optimizer_max_abs=od,actual_continuous8_vs4_restore4=True))

def canary(root):
    route=data.read(root/'evidence/H200_SELECTION.json')['route']
    for arm in ('CONTROL','GRIP'):
        if (root/arm/'canary_compare.json').exists():continue
        if arm=='GRIP':segment(root,arm,'continuous',route,8)
        split=root/arm/'split/training'
        if not (split/'checkpoint-4/trainer_state.json').exists():segment(root,arm,'split',route,4)
        segment(root,arm,'split',route,8,4)
        resume_compare(root,arm,route)
        retire_discarded(root,arm)

def retire_discarded(root,arm):
    """Only completed qualification payloads, after their exact comparison closes."""
    assert data.read(root/arm/'canary_compare.json')['status']=='PASS_PI0277_EXACT_RESUME'
    allowed=('bench0-FAST_CACHE0','bench0-FAST_CACHE2','bench0-REF_W0','split') if arm=='CONTROL' else ('continuous','split')
    for phase in allowed:
        for step in (4,8):
            cp=root/arm/phase/'training'/f'checkpoint-{step}'
            if not cp.exists():continue
            assert cp.resolve().is_relative_to(root.resolve()) and not cp.is_symlink()
            members=[]
            for p in sorted(cp.rglob('*')):
                assert not p.is_symlink()
                if p.is_file():members.append(dict(path=str(p.relative_to(cp)),bytes=p.stat().st_size,sha256=data.sha(p)))
            receipt=root/'evidence/retired'/f'{arm}-{phase}-{step}.json'
            data.write(receipt,dict(status='VERIFIED_DISCARDED_QUALIFICATION_PAYLOAD',path=str(cp),files=members,
                reason='EXACT_CANARY_COMPLETE_NOT_FORMAL_TRAINER',bytes=sum(r['bytes'] for r in members)))
            for row in members:
                p=cp/row['path'];assert p.stat().st_size==row['bytes'];p.unlink()
            for p in sorted((p for p in cp.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
            cp.rmdir()
            saved=data.read(receipt);saved['status']='DELETED_AND_VERIFIED';data.write(receipt,saved)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=('cpu','cursor','h200','canary','train'),required=True);a=p.parse_args()
    if a.stage=='cursor':cursor_check(a.root)
    elif a.stage=='cpu':cpu(a.root)
    elif a.stage=='h200':h200(a.root)
    elif a.stage=='canary':canary(a.root)
    else:
        route=data.read(a.root/'evidence/H200_SELECTION.json')['route']
        for arm in ('CONTROL','GRIP'):segment(a.root,arm,'main',route,4096)
