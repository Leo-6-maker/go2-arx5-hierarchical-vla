import contextlib,hashlib,inspect,io,json,sys,time
from pathlib import Path
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0277/R2/code')
import torch
called=[]
stock_call=torch.nn.Module.__call__
image_calls=[]
def forbidden(module,*a,**kw):
 if any(True for _ in module.parameters()):
  called.append(type(module).__module__+'.'+type(module).__name__);raise RuntimeError('HANDOFF_RELOAD_NO_PARAMETERIZED_NEURAL_FORWARD')
 image_calls.append(type(module).__module__+'.'+type(module).__name__);return stock_call(module,*a,**kw)
torch.nn.Module.__call__=forbidden
import infer
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0277/R3');old=r.with_name('R2');request=json.loads((r/'handoff/FOCUS4096/INPUT_REQUEST.json').read_text());output=Path(request['output']);assert output.is_file()
paths=[old.parent/'evidence/model_attempts.jsonl',old.parent/'evidence/model_heads.jsonl',output,output.with_suffix('.receipt.json')]
def fingerprints():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before=fingerprints();observed={};source=inspect.getsource(infer.main)
line="    model=AutoModel.from_pretrained(binding['view'],local_files_only=True,trust_remote_code=True).eval().to('cuda:0',dtype=torch.bfloat16)"
assert source.count(line)==1
source=source.replace(line,"    tick=time.perf_counter()\n"+line+"\n    torch.cuda.synchronize();observed['model_load_s']=time.perf_counter()-tick")
scope=dict(vars(infer),time=time,observed=observed);exec(compile(source,'HANDOFF_CACHED_LOAD_OBSERVER','exec'),scope)
stream=io.StringIO();sys.stdin=io.StringIO(json.dumps(request)+'\n');whole=time.perf_counter()
with contextlib.redirect_stdout(stream):scope['main']('FOCUS4096')
assert not called and fingerprints()==before
messages=[json.loads(x) for x in stream.getvalue().splitlines() if x.startswith('{')]
assert messages[-1]['status']=='VALID_REUSED' and messages[-1]['output']==str(output)
value=dict(status='PASS_ACTUAL_READONLY_MODEL_LOAD_AND_CACHED_HANDOFF_EXAMPLE',point='FOCUS4096',model_load_s=observed['model_load_s'],process_main_observed_s=time.perf_counter()-whole,measurement='One current readonly AutoModel.from_pretrained.eval.to(cuda BF16) load plus CUDA synchronization; not retrospective historical panel model-load time',delegated_infer_source_sha256=hashlib.sha256(Path(infer.__file__).read_bytes()).hexdigest(),frozen_request=request,cached_output_and_quality_ledgers_unchanged=before,parameterized_neural_Module_calls=0,image_preprocess_Module_calls=len(image_calls),engineering_reload_attempts=2,new_model_queries=0,new_model_heads=0,new_optimizer_steps=0,new_physics_steps=0,selected_checkpoint='NONE',deployment='NONE')
p=r/'evidence/HANDOFF_CACHED_RELOAD_FOCUS4096.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(value,sort_keys=True)+'\n');tmp.replace(p);print(json.dumps(value),flush=True)
