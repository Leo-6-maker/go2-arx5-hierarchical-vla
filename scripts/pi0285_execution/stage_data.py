"""Direct admitted arrays to Hopper, then compare actual native CPU inputs."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def stage():
    core='/home/liuyu-nfs/Projects/go2-arx5/PI0285';hop='/scratch/e1192548/Projects/go2-arx5/experiments/PI0285'
    remote('hop',f"from pathlib import Path\nr=Path({hop!r});assert not (r/'runtime/TRAINING_ALLOCATION.json').exists() and not (r/'runtime/JOB_SUBMISSION.json').exists(),'Adopt accepted training instead of restaging its inputs'\nprint('IDLE_OWN_DATA_DESTINATION')")
    transfer=json.loads(remote('core',f'''import json,subprocess,shlex,time,hashlib
from pathlib import Path
r=Path({core!r});proof=json.loads((r/'runtime/DATA_ADMISSION_COMPLETE.json').read_text())
assert proof['status']=='PASS_ACTUAL_BANK_CPU_READBACK_FULL_CODEC_TWO_WORKERS_CANARY_INPUT'
assert hashlib.sha256((r/'data/DATA_CODEC_ADMISSION.json').read_bytes()).hexdigest()==proof['data_codec_sha256']
key=r/'runtime/hopper_data_push';assert key.stat().st_mode&0o777==0o600
ssh=shlex.join(['ssh','-i',str(key),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(r/'runtime/hopper_known_hosts'),'-o','ConnectTimeout=15'])
before=time.time()
subprocess.run(['rsync','-a','--partial','--append-verify','-e',ssh,str((r/'data').resolve())+'/','e1192548@hopper.nus.edu.sg:./'],check=True,stdout=subprocess.DEVNULL)
v=dict(status='DIRECT_ADMITTED_DATA_SENT_PENDING_HOPPER_FULL_READBACK',elapsed_s=time.time()-before,source_admission=proof,HF_critical_path=False,full_models_copied=0,new_model_queries=0)
print(json.dumps(v))
'''))
    response=remote('hop',f'''import hashlib,json,subprocess,time
from pathlib import Path
r=Path({hop!r});d=r/'data';assert not (r/'runtime/TRAINING_ALLOCATION.json').exists(),'Adopt actual healthy training instead of copying its inputs'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
receipt=json.loads((d/'DATA_CODEC_ADMISSION.json').read_text());assert receipt['status']=='PASS_NEW_COMMON_SCENE_FULL_NATIVE_CODEC_AND_TWO_WORKERS'
assert sha(d/'DATA_CODEC_ADMISSION.json')=={transfer['source_admission']['data_codec_sha256']!r}
for n,h in receipt['bindings'].items():assert sha(d/n)==h,n
bank=json.loads((d/'SOURCE_BANK.json').read_text());members=[]
for ep in bank['admitted_episodes']:
 for n,v in ep['members'].items():
  p=d/n;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],n
  members.append(dict(path=n,bytes=v['bytes'],sha256=v['sha256']))
assert sha(d/'CANARY_INPUT.npz')=={transfer['source_admission']['canary_input_sha256']!r}
# Reuse actual checks, without repeating the full label population or fitting statistics.
code="""import json,functools,torch
from pathlib import Path
from data import Dataset,codec,collate
from cpu_inputs import tensor_hash
r=Path(ROOT);expected=json.loads((r/'DATA_CODEC_ADMISSION.json').read_text());dataset=Dataset(r,r/'CPU_INPUT_CHECK_PLAN.jsonl');encoder=codec(str(r))
assert len(dataset)==len(expected['inputs'])
for i,row in enumerate(expected['inputs']):
 item=dataset[i];item.pop('_pi0282_visit');value=encoder.batch([item])['inputs'];assert tensor_hash(value)==row['input_sha256'],row['uid']
loader=torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=2,multiprocessing_context='spawn',collate_fn=functools.partial(collate,root=str(r)),shuffle=False,generator=torch.Generator().manual_seed(28203))
for i,b in enumerate(loader):assert tensor_hash(b['inputs'])==expected['two_worker_native_batch_hashes'][i]
print('PASS_REAL_HOPPER_NATIVE_INPUTS_AND_ORDERED_TWO_WORKERS_CORELAB_IDENTICAL; model/optimizer/physics=0')
"""
check=r/'runtime/cross_host_inputs.py';check.write_text('ROOT='+repr(str(d))+'\\nif __name__=="__main__":\\n'+''.join(' '+line+'\\n' for line in code.splitlines()))
shell='source '+str(r/'code/environment.sh')+'; export CUDA_VISIBLE_DEVICES=""; "$venv/bin/python" '+str(check)
test=subprocess.run(['/bin/bash','-lc',shell],capture_output=True,text=True);assert test.returncode==0,(test.stdout[-3000:],test.stderr[-6000:])
v=dict(status='PASS_DIRECT_ALL_ADMITTED_DATA_MEMBERS_AND_CROSS_HOST_NATIVE_TWO_WORKER_INPUTS',members=members,total_bytes=sum(x['bytes'] for x in members),data_codec_sha256=sha(d/'DATA_CODEC_ADMISSION.json'),source_admission={transfer['source_admission']!r},CPU_check=test.stdout,new_model_queries=0,new_optimizer_updates=0,new_PBS_submissions=0)
(r/'evidence/DIRECT_DATA_READBACK.json').write_text(json.dumps(v,indent=2)+'\\n')
card=dict(release='7e840324c71aef47b401bd861b22710774334c97',canonical=6058329368,stage='PLACE4096',launch_allowed=True,parent=None,parent_global_update=0,updates=4096,warmup=205,peak_LRs=[1e-4,1e-5],plan_path=str(d/'PLACE4096.jsonl'),plan_sha256=sha(d/'PLACE4096.jsonl'),data_readback_sha256=sha(r/'evidence/DIRECT_DATA_READBACK.json'),selected='NONE',deployment='NONE')
p=r/'design/PLACE4096_CARD.json'
if p.exists():assert json.loads(p.read_text())==card,'Frozen card changed'
else:p.write_text(json.dumps(card,indent=2)+'\\n')
print(json.dumps({{k:x for k,x in v.items() if k!='members'}}))
''')
    result=json.loads(response);path=Path('reports/PI0285/preparation/DIRECT_DATA_READBACK.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(response)


if __name__=='__main__':stage()
