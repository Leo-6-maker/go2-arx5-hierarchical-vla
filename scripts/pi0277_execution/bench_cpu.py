"""One bounded fresh-process CPU input pass inside the allocated H200 job."""
import argparse, ctypes, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import cpu_inputs as inputs

def tree_rss():
    pending=[os.getpid()];total=0
    while pending:
        pid=pending.pop();p=Path('/proc')/str(pid)
        try:
            for line in (p/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):total+=int(line.split()[1])*1024
            pending.extend(int(x) for x in (p/'task'/str(pid)/'children').read_text().split())
        except FileNotFoundError:pass
    if total>180*2**30:raise RuntimeError('PI0277_PROCESS_TREE_RAM_CAP')
    return total

def run(root,route,trial):
    assert os.environ.get('PBS_JOBID') and os.environ.get('OPENBLAS_CORETYPE')=='Prescott'
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    library=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*'))
    fn=ctypes.CDLL(str(library)).openblas_get_corename64_;fn.restype=ctypes.c_char_p;assert fn().decode()=='Prescott'
    dst=root/'evidence/cpu_bench'/f'{route}-{trial}.json';assert not dst.exists(),'Reuse finished pass, never rebenchmark scores'
    ctx=torch.multiprocessing.get_context('spawn');ledger=inputs.ItemLedger(root/'runtime/cpu_items.jsonl',ctx)
    ds=inputs.Inputs(root,route,ledger,f'cpu_bench_{trial}')
    ds.initialize();before_items=ledger.counter.value
    warm=iter(inputs.loader(ds,stop=64,pin_memory=False))
    for i,v in enumerate(warm):assert v['_pi0276_visit']['micro_index']==i
    # Explicitly close the finite warmup loader before constructing measured worker queues.
    del warm
    before=torch.get_rng_state().clone();rng=np.random.get_state();import random;py_rng=random.getstate()
    loader=inputs.loader(ds,stop=1024,pin_memory=False);iterator=iter(loader);records=[];durations=[];peak=tree_rss()
    start=time.perf_counter()
    for i in range(1024):
        tick=time.perf_counter();value=next(iterator);durations.append(time.perf_counter()-tick)
        visit=value['_pi0276_visit'];assert visit['micro_index']==i
        records.append(visit)
        if i%32==0:peak=max(peak,tree_rss())
    elapsed=time.perf_counter()-start
    try:next(iterator)
    except StopIteration:pass
    else:raise AssertionError('Unexpected extra measured sample')
    del iterator,loader
    assert torch.equal(before,torch.get_rng_state()) and random.getstate()==py_rng
    after=np.random.get_state();assert after[0]==rng[0] and np.array_equal(after[1],rng[1]) and after[2:]==rng[2:]
    assert not torch.cuda.is_initialized()
    inputs.write(dst,dict(status='PASS_ACTUAL1024_CPU_INPUT_ORDER_AND_PARITY',route=route,trial=trial,job=os.environ['PBS_JOBID'],
        measured_calls=1024,warmup_calls=64,attempted_items=ledger.counter.value-before_items,cumulative_items=ledger.counter.value,
        seconds=elapsed,items_per_second=1024/elapsed,per_item_wait_s=durations,peak_process_tree_RSS_bytes=peak,
        actual_order_sha256=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        loader_parent_rng_unchanged=True,actual_pin_memory=False,pin_verification='REQUIRED_IN_H200_TRAINING',
        model_queries=0,optimizer_updates=0,OS_cache='NOT_EVICTED_NOT_CLAIMED_COLD'))
    print('CPU_BENCH_COMPLETE',route,trial,elapsed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--route',choices=('REF_W0','FAST_CACHE0','FAST_CACHE2'),required=True);p.add_argument('--trial',type=int,choices=(0,1),required=True);a=p.parse_args();run(a.root,a.route,a.trial)
