"""Read-only audit of the pinned completed original PD4 eight-condition panel.

No model load, certifier invocation, integration or experiment dispatch occurs.
Paths are explicit arguments; the scientific result/spec/plan hashes stay fixed.
"""
import argparse,json,time,hashlib,subprocess
from pathlib import Path
import numpy as np
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0292.eval_contract import matrix10
from scripts.pi0291_arx5_data.native_dataset import pose10
from scripts.sla.audit_query_feasibility_v7 import geometry_detail
from go2arx5_vla.world_model import fusion_trajectory_adapter as e
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows,rot6d_rows_to_matrix
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group',required=True,type=Path);p.add_argument('--training-root',required=True,type=Path);p.add_argument('--output',type=Path);args=p.parse_args();G=args.group;Q=args.training_root
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p,h):
 assert sha(p)==h,('HASH_MISMATCH',str(p));return read(p)
assert sha(G/'EXECUTION_SPEC.json')=='a07b1071288af69b328360240ccdd96f6814011a237f1f6d1afe8be72984643f'
assert sha(G/'PLAN.private.json')=='140b9087dc3200981c5a754b4acc819b2b304a63c675cb170c844d51af6bb3ec'
unit=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show','wm02-'+G.name,'--property=MainPID,SubState,Result'],text=True,timeout=15).splitlines());assert unit['MainPID']=='0'
assert sha(G/'RESULT.private.json')=='d9185b7dff07e364368efccd04a1e2d26c382900c6b02000c7a8904d6e6a2401'
assert sha(G/'SUMMARY.private.json')=='a3276511ddad7cd9126710b2cdac74406411ea502289fbb5516f8267322700dc'
lease=read(G/'control/LEASE.json');worker=read(G/'WORKER_EXIT.private.json');terminal=read(G/'RESULT.private.json');assert lease['status']=='CLOSED'and lease['worker_exit_code']==worker['exit_code']==terminal['exit_code']==0
summary=read(G/'SUMMARY.private.json');assert summary['terminal']==summary['requested']==8 and summary['complete']and len(terminal['outcomes'])==8
assert pin(Q/'RESULT.private.json','c0b2037ffe1fbef373724b12a3e3c4fe50e3448d3163df85c2175fb5a8736220')['actual_optimizer_updates']==1000
plan=read(G/'PLAN.private.json');assert [x['native_run']for x in terminal['outcomes']]==[x['native_run']for x in plan['assignments']]
rows=[];queries=[];total_consumed=0;start=time.time();cpu=time.process_time()
for a,row in zip(plan['assignments'],terminal['outcomes'],strict=True):
 R=G/'runs'/row['native_run'];N=R/'native-01';A=R.with_name(R.name+'-gpu');F=A/'fit-01';ns=read(R/'EXECUTION_SPEC.json');native=pin(N/'RESULT.json',row['result_sha256']);gpu=pin(F/'RESULT.private.json',row['GPU_result_sha256'])
 for actor in(R,A):
  l=read(actor/'control/LEASE.json');w=read(actor/'WORKER_EXIT.private.json');assert l['status']=='CLOSED'and l['worker_exit_code']==w['exit_code']
 assert native['shutdown_clean']and all(x['exit_code']==0 for x in native['owned_process_exits'])and not native['physics_stops']
 assert native['native_binary_sha256']=='39e5136603f51c092b2f3e595245468dbb0c12dbca79818b0df08da8dd4af7dd'
 assert gpu['actual_SLA_parent_loads']==1 and gpu['actual_SLA_queries']==row['actual_SLA_queries']
 assert gpu['SLA_updates']==gpu['WM_updates']==gpu['Actor_updates']==0
 captures=read(N/'TASK_LEARNING_CAPTURES.private.json')['records'];records=gpu['records'];assert len(captures)==len(records)==row['actual_SLA_queries']
 cp=N/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json';commands=read(cp)['records']if cp.exists()else[]
 past=list(read(N/'STABLE_TASK_CYCLES/-1/RESULT.private.json')['windows']);past.extend(dict(x,actual_installed_consumed_EEF10_BASE7=x['installed_consumed_EEF10'])for x in commands)
 events=[json.loads(x)for x in(N/'COMMANDS.private.jsonl').read_text().splitlines()];accepted=0;rejects=0;local=[]
 for i,(q,cap)in enumerate(zip(records,captures,strict=True)):
  qp=F/'queries'/f'{i:02d}';assert read(qp/'RESULT.json')==q and sha(qp/'ACTION.npz')==q['action_sha256']
  assert cap['sha256']==q['observation_sha256']and sha(cap['path'])==cap['sha256']
  observation=unseal(read(cap['path']));shadow=unseal(observation['original_shadow_ABI']);frame=np.asarray(shadow['source_policy_view']['T_source_from_target']);actual=matrix10(shadow['target_observation']['actual_state_eef10'])
  sf=Path(cap['path']).parent/'STATE.private.bin';assert sha(sf)==cap['state_sha256']==observation['files']['STATE.private.bin']
  step=cap['native_step'];assert step==5703+250*i and q['seed']==a['query_seeds'][i]
  history=[x for x in past if x['native_end_step']<=step][-2:];assert [x['native_end_step']for x in history]==[step-25,step]
  reference=matrix10(history[-1]['actual_installed_consumed_EEF10_BASE7'][:10])
  with np.load(qp/'ACTION.npz',allow_pickle=False)as z:raw=z['physical_H40'].copy();normalized=z['normalized'];assert raw.shape==(40,10)and normalized.shape==(1,40,32)
  target=pose10(np.linalg.inv(frame)@matrix10(raw),raw[:,9]);desired=target.copy();desired[:,9]=(e.WORKING_CLOSED_M+(e.WORKING_OPEN_M-e.WORKING_CLOSED_M)*target[:,9])/e.MECHANICAL_TRAVEL_M
  installed=desired.copy();installed[:,:3]=reference[:3,3]+target[:,:3]-actual[:3,3];installed[:,3:9]=rotation_matrix_to_rot6d_rows(reference[:3,:3]@actual[:3,:3].T@rot6d_rows_to_matrix(target[:,3:9]))
  decision_path=N/f'TASK_LEARNING_DECISION_{i:03d}.private.json';decision=read(decision_path)if decision_path.exists()else None;decision_sha=sha(decision_path)if decision else None;consumed=[x for x in commands if x['student_decision_sha256']==decision_sha]if decision else[]
  dg=geometry_detail(desired,actual);rg=geometry_detail(installed,reference)
  if decision:
   assert decision['index']==i and decision['action_sha256']==q['action_sha256']
   assert np.allclose(decision['raw_policy_H40'],raw,rtol=0,atol=0)and np.allclose(decision['mapped_target_H40'],target,rtol=0,atol=1e-12)
   assert not decision['teacher_fallback']and not decision['endpoint_basis_applied']and not decision['expert_segment_generator_applied']and decision['command_clips']==0
   assert dg['passes']and rg['passes']
  else:assert not consumed and not(dg['passes']and rg['passes'])
  cert=[x for x in events if x.get('native_step')==step and x.get('phase')in('proposed','reference-certificate-rejected')and'action_EEF10'in x and np.allclose(x['action_EEF10'],installed,rtol=0,atol=1e-12)]
  if decision:assert len(cert)==1
  for k,c in enumerate(consumed):
   assert(c['native_start_step'],c['native_end_step'])==(step+25*k,step+25*(k+1))
   assert np.allclose(c['policy_EEF10'],target[k],rtol=0,atol=1e-12)
   assert np.allclose(c['installed_consumed_EEF10'],installed[k],rtol=0,atol=1e-12)
  assert len(consumed)<=10
  if not decision:status='EEF_FULL_H40_REJECT'
  elif consumed:status='P10_CONSUMED'if len(consumed)==10 else'PARTIAL_P10_CONSUMED'
  else:status='JOINT_REJECT_BEFORE_INSTALL'
  if consumed:accepted+=1
  else:rejects+=1
  certificate=cert[0].get('native_reference_certificate')if cert else None
  qrow=dict(native_run=R.name,view=row['method'],layout=row['layout'],query=i,native_step=step,seed=q['seed'],status=status,action_sha256=q['action_sha256'],query_result_sha256=sha(qp/'RESULT.json'),state_sha256=sha(sf),observation_sha256=cap['sha256'],decision_sha256=decision_sha,consumed_points=len(consumed),consumed_ticks=25*len(consumed),desired_geometry=dg,candidate_reference_geometry=rg,original_certificate=certificate,original_certificate_event_phase=cert[0]['phase']if cert else None,original_certificate_reason=cert[0].get('reason')if cert else None,raw_mapped_and_consumed_arrays_verified=True,raw_H40_unchanged=True,Teacher_fallback=False,clips=0)
  queries.append(qrow);local.append(qrow)
 assert len(commands)==sum(x['consumed_points']for x in local);total_consumed+=len(commands)
 score=N/'TASK_LEARNING_SCORE.private.json';score=read(score)if score.exists()else None
 rows.append(dict(native_run=R.name,view=row['method'],layout=row['layout'],restart=16,technical_valid=row['technical_valid'],task_success=row['task_success'],hard_safety_stop=row['hard_safety_stop'],native_result_sha256=row['result_sha256'],GPU_result_sha256=row['GPU_result_sha256'],SLA_parent_loads=gpu['actual_SLA_parent_loads'],queries=len(records),accepted_query_prefixes=accepted,rejected_queries=rejects,consumed_points=len(commands),native_integrations=native['actual_target_integrations'],initialization_integrations=5703,student_integrations=25*len(commands),other_integrations=native['actual_target_integrations']-5703-25*len(commands),confirmed_commands_sha256=sha(cp)if cp.exists()else None,command_journal_sha256=sha(N/'COMMANDS.private.jsonl'),first_failure=row['first_failure'],shutdown_clean=native['shutdown_clean'],native_process_exit_codes=[x['exit_code']for x in native['owned_process_exits']],score=score,pair_CPU_s=row['CPU_s'],GPU_worker_wall_s=row['GPU_worker_wall_s'],native_worker_wall_s=row['wall_s']))
assert len(rows)==8 and len(queries)==sum(x['SLA_queries']for x in summary['methods'].values())
result=dict(schema='SLA_PD4_ORIGINAL8_FULL_RAW_INSTALLED_CONSUMED_AUDIT_V71',as_of=time.strftime('%Y-%m-%dT%H:%M:%S%z'),group_result_sha256=sha(G/'RESULT.private.json'),summary_sha256=sha(G/'SUMMARY.private.json'),plan_sha256=sha(G/'PLAN.private.json'),spec_sha256=sha(G/'EXECUTION_SPEC.json'),queue_worker_exit0=True,all_actor_leases_closed=True,summary=summary,conditions=rows,queries=queries,actual_SLA_parent_loads=sum(x['SLA_parent_loads']for x in rows),actual_SLA_queries=len(queries),student_consumed_points=total_consumed,actual_physical_requests=8,actual_native_integrations=sum(x['native_integrations']for x in rows),retained_engineering_physical_requests=2,retained_engineering_integrations=3078,actual_training_updates_added=0,actual_training_visits_added=0,additional_model_generations=0,additional_certifier_calls=0,additional_physical_requests=0,additional_integrations=0,old_RP8_passed=False,full_WBC_state_restored=False,formal_gate_changed=False,WBC_changes=False,queue_wall_s=terminal['wall_s'],audit_CPU_s=time.process_time()-cpu,audit_wall_s=time.time()-start)
if args.output is not None:
 with args.output.open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items()if k not in('conditions','queries')}))
