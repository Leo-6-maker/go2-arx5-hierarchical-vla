import json,hashlib,collections,time,os,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0280');sys.path.insert(0,str(r/'code'))
from source_control import encode,decode,next_tick,self_check
self_check()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
card=json.loads((r/'design/SOURCE_PHYSICS_CARD.json').read_text());rows=[];latencies=[]
for name,sha in card['source_code_sha256'].items():assert digest(r/'code'/name)==sha
model=[json.loads(x) for x in (r/'evidence/model_budget.jsonl').read_text().splitlines()]
source=[x for x in model if x['id'].startswith(str(r/'source')+'/')]
counts=collections.Counter(x['kind'] for x in source);assert counts=={'policy_attempt':240,'action_head':960}
physics_counts=collections.Counter(json.loads(x)['kind'] for x in (r/'source/evidence/budget.jsonl').read_text().splitlines());assert physics_counts==dict(starts=4,control=2640,physics=13200,rgb=2640)
expected_conditions=[10000,10001,10002,10003]
for seed in expected_conditions:
 folder=r/'source/episodes'/('PI0280-'+str(seed))/'attempt-000';result=json.loads((folder/'RESULT.json').read_text());queries=json.loads((folder/'QUERIES.json').read_text())
 assert result['status']=='VALID_SOURCE_PHYSICS_COMPLETE' and result['queries']==len(queries)==60
 for name,sha in result['files'].items():assert digest(folder/name)==sha
 initial=np.load(folder/'initial.npz')['articulations/panda_wristcam'][0];root=np.eye(4);root[:3,3]=initial[:3];root[:3,:3]=Rotation.from_quat(initial[[4,5,6,3]]).as_matrix()
 tool=np.eye(4);tool[:3,:3]=Rotation.from_euler('z',np.pi/4).as_matrix();tool[2,3]=-.1034
 with np.load(folder/'observations.npz') as saved:images={k:saved[k] for k in ['base_camera_raw','hand_camera_raw']};obs_time=saved['time_s']
 with np.load(folder/'physics.npz') as saved:state={k:saved[k] for k in saved.files}
 with np.load(folder/'installed.npz') as saved:installed={k:saved[k] for k in saved.files}
 with np.load(folder/'commands.npz') as saved:commands=saved['action']
 assert len(state['success'])==3301 and len(commands)==len(installed['action'])==660
 success=state['success'][::5][20:];predicate=state['is_cubeA_on_cubeB']&state['is_cubeA_static']&~state['is_cubeA_grasped'];assert np.array_equal(state['success'],predicate)
 persistent=bool(success[560:601].all());assert persistent==result['persistent_success']
 max_target_error=0.;max_rotation_error=0.;input_pose_error=0.;gate=collections.Counter()
 for index,q in enumerate(queries):
  control=20 if index==0 else next_tick(index-1);until=min(660,next_tick(index));assert q['control']==control and q['replan_tick']==until
  p=Path(q['input']);out=Path(q['output']);assert digest(p)==q['input_sha256'] and digest(out)==q['output_sha256']
  with np.load(p) as saved:
   assert np.array_equal(saved['exterior_image_1_left'],np.stack([images['base_camera_raw'][control-20],images['base_camera_raw'][control]]))
   assert np.array_equal(saved['wrist_image_left'],np.stack([images['hand_camera_raw'][control-20],images['hand_camera_raw'][control]]))
   assert np.allclose(saved['history_times_s'],[control*.05-1,control*.05])
   robot_state=state['qpos'][control*5];assert np.array_equal(saved['state'][10:],robot_state[:7].astype(np.float32))
   native=np.linalg.inv(root)@state['tcp'][control*5]@tool;expected=np.r_[native[:3,3],encode(native[:3,:3]),1-(robot_state[-2:].sum()-.00004)/.07996,robot_state[:7]].astype(np.float32)
   input_pose_error=max(input_pose_error,float(np.max(abs(saved['state']-expected))));assert np.max(abs(saved['state']-expected))<1e-5
  with np.load(out) as saved:
   assert saved['decoded_joint_position'].shape==(1,40,7) and saved['raw'].shape==(1,40,132)
   assert all(np.isfinite(saved[k]).all() for k in saved.files if k=='raw' or k.startswith('decoded_'));latencies.append(float(saved['latency_s']))
  with np.load(out.parent/'canonical.npz') as saved:
   ts=saved['timestamp'][:8];pos=saved['position'][:8];quat=saved['quaternion'][:8];gripper=saved['gripper'][:8]
  assert np.allclose(np.diff(ts),1/15);gate[q['gate']['decision']]+=1
  slerp=Slerp(ts,Rotation.from_quat(quat))
  for tick in range(control,until):
   clock=tick*.05;target=np.eye(4);target[:3,3]=[np.interp(clock,ts,pos[:,i]) for i in range(3)];target[:3,:3]=slerp(min(clock,ts[-1])).as_matrix();local=np.linalg.inv(root)@target
   max_target_error=max(max_target_error,float(np.max(abs(installed['base_target'][tick]-local))))
   assert np.max(abs(installed['base_target'][tick]-local))<1e-5
   j=min(7,max(0,int(np.searchsorted(ts,clock,side='right')-1)));assert abs(commands[tick,-1]-gripper[j])<1e-6;assert abs(installed['action'][tick,-1]-(2*np.clip(gripper[j],0,1)-1))<1e-6
  idcounts=collections.Counter(x['kind'] for x in source if x['id']==str(out));assert idcounts==dict(policy_attempt=1,action_head=4)
 rows.append(dict(condition=seed,status=result['status'],persistent_success=persistent,queries=60,control_calls=660,physics_steps=3300,RGB_pairs=660,gate_decisions=dict(gate),input_reconstruction_max_abs=input_pose_error,installed_first8_reference_max_abs=max_target_error,result_sha256=digest(folder/'RESULT.json'),source_code_sha256=card['source_code_sha256']))
immutable=json.loads((r/'evidence/SOURCE_POLICY_IMMUTABLE.json').read_text());assert immutable['effective_weights_unchanged']
starts=list((r/'source').glob('**/START.json'));assert len(starts)==4
manifest=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted((r/'source').rglob('*')) if p.is_file()]
(r/'evidence/SOURCE_ARTIFACTS.json').write_text(json.dumps(dict(files=manifest,total_bytes=sum(x['bytes'] for x in manifest)),indent=2)+'\n')
start=r/'runtime/source-physics.start.json';finish=r/'evidence/SOURCE_PHYSICS_RESULT.json';elapsed=(finish.stat().st_mtime_ns-start.stat().st_mtime_ns)/1e9
value=dict(status='PASS_SOURCE_RUNTIME_CLOCK_INPUT_OUTPUT_BUDGET_AUDIT',rows=rows,source_policy_counts=dict(counts),source_physics_counts=dict(physics_counts),original_predicate_success_fraction=[sum(x['persistent_success'] for x in rows),4],starts=4,engineering_replacements=0,phase_elapsed_s_by_receipt_mtimes=elapsed,phase_start_receipt_unix_ns=start.stat().st_mtime_ns,phase_completion_receipt_unix_ns=finish.stat().st_mtime_ns,policy_latency_s=dict(sum=sum(latencies),median=float(np.median(latencies)),minimum=min(latencies),maximum=max(latencies)),all_outputs_bytes=sum(x['bytes'] for x in manifest),producer_pid=715680,policy_pid=715793,processes_exited=not Path('/proc/715680').exists() and not Path('/proc/715793').exists(),final_OS_rusage='UNAVAILABLE_INITIAL_LAUNCH_NO_WAIT4_WRAPPER',retained_measured_CPU_snapshot_path=str(r/'evidence/SOURCE_PROCESS_TIMING.json'),model_parameters_unchanged=True,training_updates=0,new_audit_model_queries=0,new_audit_physics_steps=0)
(r/'evidence/SOURCE_RUNTIME_AUDIT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
