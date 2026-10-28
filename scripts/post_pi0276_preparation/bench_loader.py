"""Bounded fresh-process CPU benchmark; default is a plan, never training."""
import argparse,csv,inspect,json,multiprocessing,os,pickle,random,statistics,subprocess,sys,time
from pathlib import Path
from cpu_dataset import RunConfig,TrainDataset,ItemLedger,Consumption,collate_one,cpu_guard,read

ORDER=[0,2,2,0,0,2]
class FixedSampler:
    def __init__(self,start,stop):self.start,self.stop=start,stop
    def __iter__(self):return iter(range(self.start,self.stop))
    def __len__(self):return self.stop-self.start
def primary_busy():
    import psutil
    for p in psutil.process_iter(['cmdline']):
        try:
            args=p.info['cmdline'] or []
            if any('/PI0276/' in a and Path(a).name in ('evaluate.py','infer.py','materialize.py','archive.py','model_replay.py','report.py','video.py','closeout.py') for a in args):return True
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return False

def worker_init(index):cpu_guard()

def loader(dataset,workers,start,stop,seed):
    import torch
    options=dict(dataset=dataset,batch_size=1,collate_fn=collate_one,sampler=FixedSampler(start,stop),num_workers=workers,
                 pin_memory=False,generator=torch.Generator().manual_seed(seed))
    if workers:options.update(multiprocessing_context='spawn',worker_init_fn=worker_init,prefetch_factor=2,persistent_workers=True,timeout=30)
    if 'in_order' in inspect.signature(torch.utils.data.DataLoader).parameters:options['in_order']=True
    return torch.utils.data.DataLoader(**options)

def one_pass(config,workers,destination):
    cpu_guard();import torch,numpy as np
    ctx=multiprocessing.get_context('spawn');ledger=ItemLedger(destination/'ITEM_CALLS.jsonl',ctx)
    dataset=TrainDataset(config,ledger);start_count=ledger.counter.value
    before=(random.getstate(),pickle.dumps(np.random.get_state()),torch.get_rng_state().clone())
    setup=time.perf_counter();warm=loader(dataset,workers,0,64,config.loader_seed)
    for i,row in enumerate(warm):
        if row['visit']['micro_index']!=i:raise ValueError('Warmup order changed')
    startup=time.perf_counter()-setup
    warm.sampler.stop=2048;measured=warm;cursor=Consumption();wait=[];timings=[];raw_bytes=0
    begin=time.perf_counter();iterator=iter(measured);first=None
    for i in range(2048):
        now=time.perf_counter();row=next(iterator);elapsed=time.perf_counter()-now
        if first is None:first=time.perf_counter()-begin
        if row['visit']['micro_index']!=i or row['visit']['uid']!=dataset.plan[i]['uid']:raise ValueError('Measured visit order changed')
        cursor.consume(i);wait.append(elapsed);timings.append(row['timing']);raw_bytes+=row['raw_bytes']
    duration=time.perf_counter()-begin
    if random.getstate()!=before[0] or pickle.dumps(np.random.get_state())!=before[1] or not torch.equal(torch.get_rng_state(),before[2]):raise ValueError('Parent scientific RNG changed')
    if cursor.checkpoint()!=dict(global_step=1024,next_micro=2048):raise ValueError('Consumed cursor mismatch')
    return dict(status='CPU_PIPELINE_MEASURED',workers=workers,samples=2048,samples_per_second=2048/duration,duration_s=duration,
        startup_warmup_s=startup,first_measured_batch_s=first,dequeue_median_s=statistics.median(wait),dequeue_p95_s=sorted(wait)[1945],
        component_means_s={k:statistics.fmean(r[k] for r in timings) for k in ('io_s','processor_s','hash_s')},raw_bytes=raw_bytes,
        item_calls=ledger.counter.value-start_count,cumulative_item_calls=ledger.counter.value,parent_RNG_unchanged=True,
        all_original_condition_label_mask_hashes_match=True,storage='memmap' if config.cache else 'original_NPZ',pin_memory=False,
        CUDA_initialized=torch.cuda.is_initialized(),python=sys.version,torch=torch.__version__,numpy=np.__version__,new_model_calls=0)

def run(config_path,destination,cpus):
    import psutil
    if not 1<=cpus<=8:raise ValueError('Legal allocation of1..8 CPUs required')
    if primary_busy():return dict(status='NOT_MEASURED_RESOURCE',reason='Current PI0276 evaluation/materialization/archive owns CPU or I/O priority',passes=[])
    config=RunConfig.load(config_path);destination.mkdir(parents=True,exist_ok=True);passes=[];order=ORDER.copy()
    budget=destination/'CPU_TIME_BUDGET.json';costs=read(budget) if budget.exists() else []
    upper_hours=sum(v['allocated_CPU_core_hours_upper_bound'] for v in costs)
    prior=destination/'CPU_BENCHMARK.json'
    if prior.exists():
        previous=read(prior)
        if previous['status']=='CPU_PIPELINE_MEASURED':return previous
        passes=previous.get('passes',[])
    attempts_path=destination/'BENCHMARK_ATTEMPTS.jsonl'
    if attempts_path.exists() and len(attempts_path.read_text().splitlines())!=len(costs):raise RuntimeError('Interrupted CPU attempt needs time accounting before resume')
    if len(passes)>=6:
        a=statistics.median(p['samples_per_second'] for p in passes[:6] if p['workers']==0);b=statistics.median(p['samples_per_second'] for p in passes[:6] if p['workers']==2)
        if b>=1.15*a and max(p['peak_process_tree_RSS_bytes'] for p in passes[:6] if p['workers']==2)<32*2**30 and cpus>=5:order.extend([4,4,4])
    for number,workers in enumerate(order):
        if number<len(passes):continue
        if workers+1>cpus:raise ValueError('Worker CPUs plus owner exceed allocation')
        if primary_busy():return dict(status='PARTIAL_RESOURCE_CONTENTION',passes=passes)
        if upper_hours>=16:raise RuntimeError('Cumulative CPU time cap exhausted')
        journal=destination/'BENCHMARK_ATTEMPTS.jsonl';attempts=[json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
        if sum(r['workers']==workers for r in attempts)>=3:raise RuntimeError('Fresh-process repetition cap reached; preserve partial results')
        with journal.open('a') as f:f.write(json.dumps(dict(pass_index=number,workers=workers,started_ns=time.time_ns()))+'\n')
        output=destination/f'pass-{number:02d}-W{workers}.json'
        command=[sys.executable,__file__,'--config',str(config_path),'--output',str(destination),'--allocated-cpus',str(cpus),'--one-pass',str(workers),'--pass-result',str(output)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        begin=time.monotonic();proc=subprocess.Popen(command,env=env);peak_rss=peak_pss=0;cpu_seen={};max_threads=0;max_processes=0
        try:
            while proc.poll() is None:
                parent=psutil.Process(proc.pid);tree=[parent,*parent.children(recursive=True)];rss=pss=threads=0
                for p in tree:
                    try:
                        memory=p.memory_full_info();rss+=memory.rss;pss+=getattr(memory,'pss',0);threads+=p.num_threads()
                        cpu=p.cpu_times();cpu_seen[p.pid]=max(cpu_seen.get(p.pid,0),cpu.user+cpu.system)
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                peak_rss=max(peak_rss,rss);peak_pss=max(peak_pss,pss);max_threads=max(max_threads,threads);max_processes=max(max_processes,len(tree))
                elapsed=time.monotonic()-begin
                if rss>64*2**30 or psutil.virtual_memory().available<4*2**30 or elapsed>min(600,(16-upper_hours)*3600/cpus) or primary_busy():raise RuntimeError('Bounded CPU pass stopped for memory, time or mainline contention')
                time.sleep(.1)
            if proc.returncode:raise RuntimeError(f'CPU pass failed; no automatic performance retry: {proc.returncode}')
        except BaseException:
            if proc.poll() is None:
                for child in psutil.Process(proc.pid).children(recursive=True):child.kill()
                proc.kill();proc.wait()
            raise
        finally:
            costs.append(dict(pass_index=number,workers=workers,returncode=proc.returncode,allocated_CPU_core_hours_upper_bound=cpus*(time.monotonic()-begin)/3600))
            budget.write_text(json.dumps(costs,indent=2)+'\n')
        elapsed=time.monotonic()-begin;upper_hours+=cpus*elapsed/3600
        if upper_hours>16:raise RuntimeError('16 CPU-core-hour conservative bound exhausted')
        value=read(output);value.update(peak_process_tree_RSS_bytes=peak_rss,peak_PSS_bytes=peak_pss or None,
            sampled_process_tree_CPU_seconds=sum(cpu_seen.values()),CPU_time_sampling_is_lower_bound=True,
            allocated_CPU_core_hours_upper_bound=cpus*elapsed/3600,max_threads=max_threads,max_processes=max_processes,cold_process_not_cold_disk=True)
        passes.append(value);(destination/'CPU_BENCHMARK.json').write_text(json.dumps(dict(status='PARTIAL_MEASURED',passes=passes),indent=2)+'\n')
        if number==5:
            a=statistics.median(p['samples_per_second'] for p in passes if p['workers']==0);b=statistics.median(p['samples_per_second'] for p in passes if p['workers']==2)
            if b>=1.15*a and max(p['peak_process_tree_RSS_bytes'] for p in passes if p['workers']==2)<32*2**30 and cpus>=5:order.extend([4,4,4])
    a=statistics.median(p['samples_per_second'] for p in passes if p['workers']==0);b=statistics.median(p['samples_per_second'] for p in passes if p['workers']==2)
    return dict(status='CPU_PIPELINE_MEASURED',passes=passes,recommendation='W2' if b>=1.15*a else 'W0',
                recommendation_scope='CPU_LOADER_CANDIDATE_ONLY_GPU_EQUIVALENCE_NOT_RUN',CPU_core_hours_upper_bound=upper_hours)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute-cpu',action='store_true');p.add_argument('--allocated-cpus',type=int,default=0);p.add_argument('--one-pass',type=int,choices=(0,2,4));p.add_argument('--pass-result',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.one_pass is not None:
        if primary_busy() or not a.one_pass+1<=a.allocated_cpus<=8:raise RuntimeError('CPU allocation or mainline priority guard')
        if a.one_pass==4:
            old=read(a.output/'CPU_BENCHMARK.json')['passes'];w0=statistics.median(r['samples_per_second'] for r in old if r['workers']==0);w2=statistics.median(r['samples_per_second'] for r in old if r['workers']==2)
            if w2<1.15*w0 or max(r['peak_process_tree_RSS_bytes'] for r in old if r['workers']==2)>=32*2**30:raise RuntimeError('Conditional W4 gate not met')
        result=one_pass(RunConfig.load(a.config),a.one_pass,a.output);a.pass_result.write_text(json.dumps(result,indent=2)+'\n');return
    result=run(a.config,a.output,a.allocated_cpus) if a.execute_cpu else dict(status='PLAN_ONLY',order=ORDER,base_item_calls=12672,W4_conditional=True,hard_item_cap=32768,new_model_calls=0)
    (a.output/'CPU_BENCHMARK.json').write_text(json.dumps(result,indent=2)+'\n')
    if result.get('passes'):
        fields=['workers','samples_per_second','startup_warmup_s','dequeue_median_s','dequeue_p95_s','item_calls','peak_process_tree_RSS_bytes']
        with (a.output/'CPU_BENCHMARK.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(result['passes'])
    print(json.dumps(result))

if __name__=='__main__':main()
