import hashlib,json
from pathlib import Path
import numpy as np,mujoco
from scripts.pi0294.task_capture_runtime import read,sha,cpu_stage
from scripts.pi0292.durable import write
out=root/'source-demo-access-v3-preparation';old=root/'source-demo-retarget-v1';v2=root/'source-demo-retime-join-v2';ledger0=read(parent/'R4_RESOURCE_LEDGER.json')
with cpu_stage(root,'SDR3_PREFIX_IDS_R1',cores=1,cpu_s_cap=120):
 source=read(old/'SOURCE_EXTRACTION.json');bank=Path(read(old/'SOURCE_BINDING.private.json')['source_bank']);manifest0=read(bank/'DATASET_MANIFEST.json');rows=[]
 for r in source['records']:
  if r['task']!='T2':continue
  m=next(x for x in manifest0['episodes'] if x['id']==r['id']);raw=Path(m['raw_root']);samples=read(raw/'samples.json');wrapper=read(raw/'scene.xml.json');assert hashlib.sha256(wrapper['xml'].encode()).hexdigest()==wrapper['sha256']==r['source_model_xml_sha256'];model=mujoco.MjModel.from_xml_string(wrapper['xml'])
  cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'cube_geom');pads=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ['left_pad','right_pad']];assert min([cube]+pads)>=0
  index=int(np.searchsorted(np.arange(640)*.05,next(e['time_s'] for e in r['events'] if e['stage']=='contact')));before=samples[:index];seen=[];allpairs={}
  for i,s in enumerate(samples[:index+1]):
   for c in s['privileged']['contacts']:
    ids=[c['geom0'],c['geom1']];names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in ids];allpairs[tuple(names)]=allpairs.get(tuple(names),0)+1
    if cube in ids and any(g in ids for g in pads):seen.append(dict(index=i,names=names,distance_m=c['distance_m']))
  pr=next(x for x in read(v2/'PROGRAMS.json')['rows'] if x['id']==r['id']);contact_time=next(e['program_time_s'] for e in pr['events'] if e['stage']=='contact');align_time=next(e['program_time_s'] for e in pr['events'] if e['stage']=='align')
  rows.append(dict(id=r['id'],source_model_sha256=wrapper['sha256'],source_geom_ids=dict(cube=cube,pads=pads),join_index=index,retained_end_index=r['cut_end_inclusive_index'],retained_count=r['cut_end_inclusive_index']-index+1,
   removed_V2_pose_indices=[r['cut_start_index'],index-1],pad_cube_contacts_before_join=[x for x in seen if x['index']<index],pad_cube_contacts_at_join=[x for x in seen if x['index']==index],all_contact_pairs_through_join=[dict(names=list(k),records=n) for k,n in allpairs.items()],
   grasp_before=any(s['privileged']['current_grasped'] for s in before),lift_before=any(s['privileged']['ever_lifted'] for s in before),carry_before=any(s['privileged']['ever_carried'] for s in before),
   object_horizontal_max_before_m=max(float(np.linalg.norm(np.asarray(s['privileged']['object_pose'])[:2,3]-np.asarray(samples[0]['privileged']['object_pose'])[:2,3])) for s in before),
   gripper_at_join=samples[index]['supervision']['committed_gripper'],suffix_original_local_duration_s=pr['active_duration_s']-contact_time,source_close_phase_local_s=align_time,
   source_samples_sha256=sha(raw/'samples.json'),V2_program_sha256=pr['sha256']))
 value=dict(status='PASS_ORIGINAL_GEOM_IDS_PREFIX_AUDIT',rows=rows,prior_receipt_sha256=sha(out/'T2_SOURCE_CONTACT_PREFIX_AUDIT.private.json'),
  correction='Prior name-based a/b lookup did not read actual geom0/geom1 IDs; prior_contact_count is not valid evidence. R1 resolves original source model names/IDs without solver/physics.',new_physics=0,new_IK=0,new_RRT=0,new_render=0,new_neural=0)
 write(out/'T2_SOURCE_CONTACT_PREFIX_AUDIT_R1.json',value);print('SDR3_PREFIX_R1='+json.dumps(value))
ledger=read(parent/'R4_RESOURCE_LEDGER.json');assert not ledger['CPU_reservations'];account=dict(status='SETTLED_READONLY_PREPARATION_R1',before_CPU_core_s=ledger0['Core_CPU_s'],after_CPU_core_s=ledger['Core_CPU_s'],preparation_CPU_core_s=ledger['Core_CPU_s']-ledger0['Core_CPU_s'],parent={k:ledger[k] for k in ['main_starts','main_integrations','RGB_pairs','Core_CPU_s']},new_physics=0,new_IK=0,new_RRT=0,new_render=0,new_neural=0,launch_allowed=False)
write(out/'PREPARATION_ACCOUNTING_R1.json',account);print('SDR3_PREP_ACCOUNTING_R1='+json.dumps(account))
