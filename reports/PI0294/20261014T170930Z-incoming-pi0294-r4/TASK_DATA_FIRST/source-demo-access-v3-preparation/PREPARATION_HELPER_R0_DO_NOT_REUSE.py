import collections,hashlib,itertools,json,math,struct,time
from pathlib import Path
import numpy as np,mujoco
from scripts.pi0294.task_capture_runtime import read,sha,cpu_stage
from scripts.pi0292.durable import write
from scripts.pi0294.task_state import parse_state
p=root/'source-demo-retime-join-v2';out=root/'source-demo-access-v3-preparation';assert not out.exists()
ledger0=read(parent/'R4_RESOURCE_LEDGER.json');oldhash={str(f.relative_to(p)):sha(f) for f in [p/'SERIES_CLOSED.json',p/'SEARCH_LEDGER.json',p/'FINAL_PRIVATE_READBACK.json']}
with cpu_stage(root,'SDR3_PREPARATION_READONLY01',cores=1,cpu_s_cap=120):
 out.mkdir();ep=p/'episodes/SDR2_SEED_W0_T1';model=mujoco.MjModel.from_binary_path(str(ep/'task-scene.private.mjb'));body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base_link')
 geoms=[int(g) for g in np.flatnonzero(model.geom_bodyid==body) if model.geom_contype[g] or model.geom_conaffinity[g]];rows=[];maxz={g:-float('inf') for g in geoms};counts=0
 assert 11 in geoms and int(model.geom_type[11])==6
 def rotation(q):
  dst=np.empty(9);mujoco.mju_quat2Mat(dst,np.asarray(q,dtype=float));return dst.reshape(3,3)
 with (ep/'state.bin.journal').open('rb') as f:
  while raw:=f.read(4):
   n,=struct.unpack('<I',raw);packet=parse_state(f.read(n));q=np.asarray(packet['qpos']);R=rotation(q[3:7]);counts+=1
   for g in geoms:
    rg=R@rotation(model.geom_quat[g]);center=q[:3]+R@model.geom_pos[g];aabb=model.geom_aabb[g]
    high=float((center+rg@aabb[:3])[2]+np.abs(rg[2])@aabb[3:]);maxz[g]=max(maxz[g],high)
 assert counts==5300
 for g in geoms:rows.append(dict(id=g,name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g),type=int(model.geom_type[g]),local_half_sizes=model.geom_size[g].tolist(),margin_m=float(model.geom_margin[g]),saved_AABB_max_z_m=maxz[g]))
 bound=max(maxz[g]+float(model.geom_margin[g]) for g in geoms);newtop=math.ceil((bound+.02+.05)/.01-1e-10)*.01
 geom=dict(status='READ_ONLY_SAVED_ROOT_GEOMETRY_BOUND',samples=counts,geoms=rows,old_table_top_m=.44,old_table_bottom_m=.39,unchanged_table_half_sizes=[.17,.24,.025],
  observed_root_geometry_plus_margin_top_m=bound,extra_geometric_clearance_m=.02,height_quantum_m=.01,proposed_table_top_m=newtop,delta_z_m=newtop-.44,height_cap_m=.60,within_fixed_design_cap=newtop<=.60,
  lower_face_clearance_over_saved_envelope_m=newtop-.05-bound,interpretation='SAVED_SWEPT_ENVELOPE_ONLY_NOT_NEW_DYNAMIC_SAFETY_OR_TASK_PASS',native_model_sha256=sha(ep/'task-scene.private.mjb'),journal_sha256=sha(ep/'state.bin.journal'),new_physics=0,new_IK=0,new_RRT=0,new_render=0,new_neural=0)
 write(out/'T1_SAVED_GEOMETRY_BOUND.json',geom)
 old=root/'source-demo-retarget-v1';source=read(old/'SOURCE_EXTRACTION.json');bank=Path(read(old/'SOURCE_BINDING.private.json')['source_bank']);manifest0=read(bank/'DATASET_MANIFEST.json');prefix=[]
 for r in source['records']:
  if r['task']!='T2':continue
  m=next(x for x in manifest0['episodes'] if x['id']==r['id']);raw=Path(m['raw_root']);assert all(sha(raw/n)==d for n,d in r['raw_source_hashes'].items())
  samples=read(raw/'samples.json');events=r['events'];event=next(e for e in events if e['stage']=='contact');index=int(np.searchsorted(np.arange(640)*.05,event['time_s']));before=samples[:index];through=samples[:index+1];object0=np.asarray(samples[0]['privileged']['object_pose'])[:3,3];pad=[]
  for i,s in enumerate(through):
   for c in s['privileged']['contacts']:
    names=[c.get('a'),c.get('b')]
    if any('pad' in str(x) for x in names) and any('cube' in str(x) for x in names):pad.append(i)
  prefix.append(dict(id=r['id'],contact_event_s=event['time_s'],join_index=index,retained_end_index=r['cut_end_inclusive_index'],
   retained_count=r['cut_end_inclusive_index']-index+1,original_cut_start_index=r['cut_start_index'],
   prior_contact_rows=pad[:20],prior_contact_count=len(pad),first_privileged_contact=next((dict(index=i,contacts=s['privileged']['contacts']) for i,s in enumerate(through) if s['privileged']['contacts']),None),
   native_grasp_before=any(s['privileged']['current_grasped'] for s in before),ever_lifted_before=any(s['privileged']['ever_lifted'] for s in before),ever_carried_before=any(s['privileged']['ever_carried'] for s in before),
   object_horizontal_max_before_m=max(float(np.linalg.norm(np.asarray(s['privileged']['object_pose'])[:2,3]-object0[:2])) for s in before),
   object_vertical_max_before_m=max(abs(float(np.asarray(s['privileged']['object_pose'])[2,3]-object0[2])) for s in before),
   gripper_at_join=samples[index]['supervision']['committed_gripper'],join_pose=samples[index]['supervision']['committed'],samples_sha256=sha(raw/'samples.json')))
 write(out/'T2_SOURCE_CONTACT_PREFIX_AUDIT.private.json',dict(status='SAVED_SOURCE_PREFIX_READONLY',rows=prefix,new_physics=0,new_IK=0,new_RRT=0,new_render=0,new_neural=0))
 for rel,d in oldhash.items():assert sha(p/rel)==d
 print('SDR3_PREPARATION='+json.dumps(dict(geometry=geom,T2_prefix=prefix,old_V2_hashes_unchanged=True,old_hashes=oldhash)))
ledger=read(parent/'R4_RESOURCE_LEDGER.json');assert not ledger['CPU_reservations'];account=dict(status='SETTLED_READONLY_PREPARATION',before_CPU_core_s=ledger0['Core_CPU_s'],after_CPU_core_s=ledger['Core_CPU_s'],preparation_CPU_core_s=ledger['Core_CPU_s']-ledger0['Core_CPU_s'],
 parent={k:ledger[k] for k in ['main_starts','main_integrations','RGB_pairs','Core_CPU_s']},new_physics=0,new_IK=0,new_RRT=0,new_render=0,new_neural=0,source_derived=False,launch_allowed=False)
write(out/'PREPARATION_ACCOUNTING.json',account);print('SDR3_PREP_ACCOUNTING='+json.dumps(account))
