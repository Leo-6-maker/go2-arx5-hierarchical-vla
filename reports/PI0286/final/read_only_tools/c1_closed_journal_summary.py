"""Closed journal scalar summaries only; not a model, gradient or per-example loss rerun."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,'scripts/pi0276_execution');from coordinator import remote
SCRIPT=r'''
import os,json,hashlib,time,collections,statistics,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='4'
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286');arm='C1_CORRECT1024';p=r/'evidence'/(arm+'-1024_TRAINING_CPU.json');folder=r/'learning'/arm/'main';closed=folder/'JOURNAL.json'
if not p.exists() or not closed.exists():
 print(json.dumps(dict(status='UNOBSERVED_C1_FINAL_CPU_AND_CLOSED_JOURNALS',CPU_exists=p.exists(),JOURNAL_closed_exists=closed.exists(),model=0,optimizer=0)));raise SystemExit(0)
a=json.loads(p.read_text());assert a['status']=='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE'
def sha(blob):return hashlib.sha256(blob).hexdigest()
js={};ids={}
for name,key in [('consumption.jsonl','consumption'),('native_training_forwards.jsonl','native_forward'),('native_loss_modes_LR.jsonl','native_modes')]:
 f=folder/name;all_lines=f.read_bytes().splitlines(keepends=True);blob=b''.join(all_lines[:2048]);proof=a['journal_prefixes'][key];assert sha(blob)==proof['sha256'] and len(all_lines)==2048
 js[key]=[json.loads(x) for x in all_lines];ids[key]=dict(path=str(f),bytes=len(blob),sha256=sha(blob),lines=2048)
rows=js['consumption'];modes=js['native_modes'];visits=[v for x in rows for v in x['visits']];assert len(visits)==8192
counts=collections.Counter((v['source'],str(v['phase'])) for v in visits);phase=collections.Counter(str(v['phase']) for v in visits);sources=collections.Counter(v['source'] for v in visits);losses=[];phase_loss=collections.defaultdict(list);mix=collections.Counter();source_loss=collections.defaultdict(list);composition_losses=collections.defaultdict(list)
for x,m in zip(rows,modes):
 loss=m['native_loss'];assert math.isfinite(loss) and math.isclose(x['loss'],loss/2,rel_tol=0,abs_tol=1e-12)
 losses.append(loss);ps={str(v['phase']) for v in x['visits']};ss={v['source'] for v in x['visits']}
 if len(ps)==1:phase_loss[next(iter(ps))].append(loss)
 else:mix['mixed_phase_microbatches']+=1
 if len(ss)==1:source_loss[next(iter(ss))].append(loss)
 else:mix['mixed_source_microbatches']+=1
 label=json.dumps(sorted(collections.Counter((v['source'],str(v['phase'])) for v in x['visits']).items()));composition_losses[label].append(loss)
def stats(xs):return dict(n=len(xs),mean=statistics.mean(xs),median=statistics.median(xs),min=min(xs),max=max(xs)) if xs else dict(status='UNOBSERVED_NO_SOURCE_PURE_MICROBATCH',n=0)
grad=collections.defaultdict(list);details=[]
for x in rows:
 g=x.get('gradient',{})
 if 'row_module_gradient_norms' in g:
  details.append(dict(step=x['step'],micro=x['micro'],gradient=g))
  for name,value in g['row_module_gradient_norms'].items():grad['row_'+name].append(value)
  if 'DiT_gradient_norm' in g:grad['DiT_gradient_norm'].append(g['DiT_gradient_norm'])
LRs={str(i):stats([m['actual_LRs'][i] for m in modes]) for i in range(2)}
additional=dict(saved_gradient_norms={k:stats(xs) for k,xs in grad.items()},detailed_gradient_samples=len(details),saved_gradient_details=details,actual_LR_stats=LRs,actual_LR_first=modes[0]['actual_LRs'],actual_LR_last=modes[-1]['actual_LRs'],allocator_GPU_allocated_bytes=stats([x['allocated_GPU_bytes'] for x in rows]),allocator_GPU_reserved_bytes=stats([x['reserved_GPU_bytes'] for x in rows]),loader_wait_s=stats([x['loader_wait_s'] for x in rows]),forward_backward_s=stats([x['forward_backward_s'] for x in rows]),new_gradient_computation=0)
print(json.dumps(dict(additional=additional,status='PASS_C1_CLOSED_NATIVE_JOURNAL_DESCRIPTIVE_SOURCE_PHASE_EXPOSURE_AND_STORED_SCALAR_LOSS',epoch=time.time(),training_CPU_sha256=sha(p.read_bytes()),closed_JOURNAL_sha256=sha(closed.read_bytes()),identities=ids,formal_updates=1024,microbatches=2048,chunks=8192,source_counts=dict(sources),phase_counts=dict(phase),source_by_phase={s:{ph:counts[s,ph] for ph in sorted(phase)} for s in sorted(sources)},global_native_loss=stats(losses),first16updates_native_loss_mean=statistics.mean(losses[:32]),last16updates_native_loss_mean=statistics.mean(losses[-32:]),single_phase_microbatch_scalar_losses={ph:stats(xs) for ph,xs in phase_loss.items()},pure_source_microbatch_scalar_losses={s:stats(source_loss[s]) for s in sources},mixed_microbatch_counts=dict(mix),composition_conditioned_microbatch_scalar_losses={k:stats(xs) for k,xs in composition_losses.items()},per_example_source_loss='UNOBSERVED_NOT_RECOVERABLE_FROM_MIXED_MICROBATCH_SCALAR',causality='DESCRIPTIVE_NOT_SOURCE_CAUSAL_GRADIENT_EFFECT',new_model=0,new_gradient=0,new_optimizer=0,new_physics=0,new_codec=0)))
'''
if __name__=='__main__':
 v=json.loads(remote('hop',SCRIPT));p=Path('reports/PI0286/final/C1_CLOSED_SOURCE_PHASE_LOSS.json');p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k] for k in ('status',) if k in v}))
