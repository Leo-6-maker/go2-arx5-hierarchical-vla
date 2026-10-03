"""Reuse closed donor/U replay and verify the complete PREFIX main panel from saved arrays."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import json,sys,hashlib
from pathlib import Path
import numpy as np
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0269';RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0269')
sys.path.insert(0,str(ROOT/'code'))
import report as helper
helper.admitted()
import torch
torch.set_num_threads(4)
def forbidden(*args,**kwargs):raise RuntimeError('SAVED_ARRAY_ANALYSIS_NO_MODEL_OR_OPTIMIZER')
torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
prior_path=ROOT/'evidence/offline-only/DONOR_UNIFORM_MAIN_CPU_PRELIMINARY.json'
prior=helper.read(prior_path);assert prior['main_rows']==len(prior['rows'])==len(prior['sources'])==128
assert prior['status']=='COMPLETE_DONOR_AND_UNIFORM_MAIN_REPLAY_PREFIX_AND_COMPONENT_PENDING'
for source in prior['sources']:
 parent=(RAW.parent/'PI0268/R1' if source['point']=='ROT2048' else RAW)/'virtual'/source['point']/source['case']
 assert helper.sha(parent/'result.json')==source['record_sha256'] and helper.sha(parent/'trajectory.npz')==source['arrays_sha256']
point='PREFIX4096';paths=sorted((RAW/'virtual'/point).glob('*/result.json'));assert len(paths)==64
new=[];sources=list(prior['sources'])
for path in paths:
 record=helper.read(path);assert record['validity']=='VALID' and record['point']==point
 assert helper.sha(path.parent/'trajectory.npz')==record['arrays_sha256']
 with np.load(path.parent/'trajectory.npz',allow_pickle=False) as z:
  spec=json.loads(str(z['spec_json']));t,p,r=z['time'],z['actual_p'],z['actual_r']
  result=helper.score(t,p,r,z['installed_p'],z['installed_r'],spec,record['scorer']['safety_failure']);assert result==record['scorer']
  if spec['task']=='M3':
   broad,entry,arc=helper.independent_entry(t,p,r,spec)
   assert entry==result['entry_index'] and (None if broad is None else float(t[broad]))==result['earliest_broad_readiness_s']
  refusals=sum(g['result']['decision']=='REJECT' for g in record['gates']);collisions=len(record['collision_events']);failure=record['stop_reason'] is not None
  retention=helper.reliable(t,p,r,spec['goal'],spec['goal_rotation'],result,gate_refusals=refusals,collisions=collisions,execution_failure=failure)
  new.append(dict(point=point,case=path.parent.name,task=record['task'],command=record['command'],backend=record['backend'],success=result['success'],reliable=retention['reliable'],gate_refusals=refusals,dangerous_collision=bool(collisions),execution_failure=failure,retention=retention,original_score=result))
 sources.append(dict(point=point,case=path.parent.name,record_sha256=helper.sha(path),arrays_sha256=record['arrays_sha256']))
rows=prior['rows']+new;assert len(rows)==len(sources)==192
original=dict(prior['original'],PREFIX4096=helper.qualify(new,'success'));reliability=dict(prior['reliability'],PREFIX4096=helper.qualify(new,'reliable'))
compact=[]
for row in rows:
 item={k:v for k,v in row.items() if k!='original_score'}
 item['V2_success']=row['original_score']['legacy_v2']['success'];item['actual_final_xyz_m']=row['original_score']['final_xyz_m'];item['actual_final_so3_rad']=row['original_score']['final_so3_rad']
 compact.append(item)
result=dict(status='COMPLETE_MAIN192_CPU_REPLAY_COMPONENT_AND_FULL_NATIVE_REPLAY_PENDING',main_rows=192,reused_verified_rows=128,new_CPU_replayed_rows=64,original=original,reliability=reliability,rows=compact,sources=sources,previous_CPU_receipt_sha256=helper.sha(prior_path),source_code_sha256=helper.sha(Path(__file__)),source_report_sha256=helper.sha(ROOT/'code/report.py'),shared_V3_scorer=True,independent_entry_only_new_PREFIX=True,not_final_completion=True,model_queries=0,gradient_queries=0,teacher_calls=0,render_calls=0,physics_steps=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
helper.write(ROOT/'evidence/offline-only/MAIN192_CPU_PRELIMINARY.json',result)
print(json.dumps({k:result[k] for k in ('status','main_rows','reused_verified_rows','new_CPU_replayed_rows','original','reliability')}),flush=True)
