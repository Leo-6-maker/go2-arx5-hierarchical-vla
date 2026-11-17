"""Finite static population preparation before any V5.2 training/selection."""
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np
from scripts.pi0292.durable import write
from go2arx5_vla.world_model.storage import digest

PLAN='90d39196bdc345b977ee54ed64e750cf3561aa72'


def read(p):return json.loads(Path(p).read_bytes())
def pin(p):return dict(path=str(p),sha256=digest(p))
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def seed(tag):return int(hashlib.sha256(('SLA_V52/'+tag).encode()).hexdigest()[:8],16)


def initial_packet(case):
    for native in sorted(Path(case).glob('native-*')):
        p=native/'state.bin.journal'
        if not p.exists():continue
        with p.open('rb')as f:
            size=f.read(4)
            if not size:continue
            n=struct.unpack('<I',size)[0]
            if not 40<=n<=2*1024**2:continue
            raw=f.read(n);magic,step,t,nq,nv,nu=struct.unpack_from('<8sQdIII',raw)
            if magic!=b'GRTSTATE'or nq!=34:continue
            q=np.frombuffer(raw,dtype='<f8',count=nq,offset=36).copy()
            return dict(object_xyz=q[27:30].tolist(),initial_qpos=q.tolist(),packet_sha256=hashlib.sha256(raw).hexdigest(),native_step=step)
    return None


def historical_centers(inventory):
    import mujoco
    centers=[];proof=[];seen={}
    for item in inventory['models']:
        path=Path(item['path']);actual=digest(path)
        if actual in seen:
            centers.extend(seen[actual]);continue
        # Historical mismatches are never admitted for execution. For spatial
        # exclusion include current geometry AND independently saved initial
        # observations; no history is rewritten as a qualified model.
        try:
            m=mujoco.MjModel.from_xml_path(str(path))if path.suffix=='.xml'else mujoco.MjModel.from_binary_path(str(path))
            points=[]
            for j in range(m.njnt):
                if int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_FREE)and m.jnt_qposadr[j]>0:
                    a=int(m.jnt_qposadr[j]);points.append(m.qpos0[a:a+3].copy().tolist())
                    if m.nkey:points.extend(m.key_qpos[:,a:a+3].tolist())
            for saved in item['actual_initial_objects']:points.append(np.asarray(saved['object_pose'])[:3,3].tolist())
        except ValueError:
            points=[]
            for context in item['contexts']:
                packet=initial_packet(context['case'])
                if packet:points.append(packet['object_xyz'])
            if not points:raise ValueError('HISTORICAL_GEOMETRY_UNRESOLVED:'+str(path))
        centers.extend(points);seen[actual]=points;proof.append(dict(model_sha256=actual,declared_sha256=item['sha256'],initial_centers=points,context_count=len(item['contexts'])))
    missing={}
    for item in inventory['missing_models']:missing.setdefault(item['model'],[]).append(item)
    for model,contexts in sorted(missing.items()):
        found=[]
        for context in contexts:
            p=initial_packet(context['case'])
            if p:found.append(p)
        if not found:raise ValueError('MISSING_HISTORICAL_MODEL_AND_INITIAL_STATE:'+model)
        centers.extend(p['object_xyz']for p in found);proof.append(dict(model_sha256=contexts[0]['expected_sha256'],model_missing=True,recorded_initial_states=found))
    a=np.unique(np.asarray(centers,float),axis=0)
    if len(a)<10:raise ValueError('INCOMPLETE_EXCLUSION_POPULATION')
    return a,proof


def balanced_roll(rows):
    generated=[r for r in rows if'action_sha256'in r];chosen=[]
    for family in('F1','F2'):
        available=[r for r in generated if r['family']==family];ids=sorted({r['case']for r in available});by={i:sorted([r for r in available if r['case']==i],key=lambda r:r['query'])for i in ids}
        mandatory=[r for r in available if not r['consumption']['full_H40_compiled']]
        if family=='F2':mandatory.extend([r for r in by['00-student-f2']if r['query']>=35])
        selected={(r['case'],r['query']):r for r in mandatory}
        while len(selected)<48:
            possible=[]
            for case in ids:
                remaining=[r for r in by[case]if(case,r['query'])not in selected]
                if not remaining:continue
                used=[r['query']for k,r in selected.items()if k[0]==case];extent=max(by[case][-1]['query'],1)
                def separation(r):return min((abs(r['query']-u)/extent for u in used),default=-abs(r['query']/extent-.5))
                best=min(remaining,key=lambda r:(-separation(r),r['query']))
                possible.append((len(used),case,best))
            if not possible:raise ValueError('ROLL96_DISTINCT_INPUT_UNDERFILLED')
            _,_,best=min(possible,key=lambda x:(x[0],x[1]));selected[(best['case'],best['query'])]=best
        chosen.extend(selected[k]for k in sorted(selected))
    assert len(chosen)==96 and len({(r['case'],r['query'])for r in chosen})==96
    return chosen


def run(spec):
    import mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from go2arx5_vla.world_model.pilot1_curriculum import robot_contract
    from scripts.pi0294.mp_client import contact_permitted
    from scripts.wm02.mobile_d0_v52 import packet_range
    root=Path(spec['registered_root']);start=time.time()
    if mujoco.get_mjcb_control()is not None:raise PermissionError('STATIC_READER_WITHOUT_CONTROL_CALLBACK')
    if spec['plan_commit']!=PLAN or spec['new_physical_requests_budget']or spec['new_model_generations_budget']:raise PermissionError('STATIC_V52_PREPARATION_ONLY')
    p=spec['historical_inventory'];inventory=read(p['path']);assert digest(p['path'])==p['sha256']
    old,proof=historical_centers(inventory);write(root/'EXCLUSION_GEOMETRY.private.json',dict(centers=old.tolist(),provenance=proof))
    bank=read(spec['CN38_bank']['path']);assert digest(spec['CN38_bank']['path'])==spec['CN38_bank']['sha256'];templates={}
    for ep in bank['episodes']:
        s=read(ep['spec']['path']);condition=s['mobile_condition'];templates.setdefault((condition['family'],condition['layout']),s)
    assert all((f,'T%d'%i)in templates for f in('F1','F2')for i in range(4))
    descriptors=[];layouts=[];rejected=[];checks=0;scene_root=root/'scenes';scene_root.mkdir()
    for family in('F1','F2'):
        base_spec=templates[(family,'T0')];original_path=base_spec['model_path'];assert digest(original_path)==base_spec['frozen_files'][original_path]
        original=mujoco.MjModel.from_binary_path(original_path);cube=original.body('task_cube').id;address=int(original.jnt_qposadr[original.body_jntadr[cube]]);origin=original.key_qpos[0,address:address+3].copy();table=original.geom('task_table').id
        journal=Path(base_spec['registered_root'])/'native-01/state.bin.journal';prefix=list(packet_range(journal,1,5703));assert len(prefix)==5703
        # Entire candidate pool and clustering rule are independent of model
        # outputs. Only static support/contact and old-state overlap filter it.
        candidates=[(a,c)for a in(8,14,20,26,32,38)for c in(-18,-12,-6,0,6,12,18)]if family=='F1'else[(c,a)for a in(8,14,20,26,32,38)for c in(2,8,14,20,26,32,38)]
        candidates.sort(key=lambda xy:(xy[0]**2+xy[1]**2,identity([PLAN,family,*xy])))
        selected=[]
        for xy in candidates:
            if len(selected)==18:break
            if time.time()>=spec['deadline_epoch']-60:raise TimeoutError('FINITE_STATIC_SPLIT_PREPARATION')
            xyz=origin+np.r_[np.asarray(xy)/1000.,0.]
            if np.min(np.linalg.norm(old[:,:2]-xyz[:2],axis=1))<.005-1e-10 or any(np.linalg.norm(xyz[:2]-r['object_xyz_m'][:2])<.005-1e-10 for r in layouts):continue
            if np.any(np.abs(xyz[:2]-original.geom_pos[table,:2])+.02+.005>original.geom_size[table,:2]):continue
            m=mujoco.MjModel.from_binary_path(original_path);m.qpos0[address:address+3]=xyz;m.qpos_spring[address:address+3]=xyz;m.key_qpos[:,address:address+3]=xyz
            contract=robot_contract(original,m);d=mujoco.MjData(m);bad=None
            with forbid_integration(mujoco)as forbidden:
                for packet in prefix:
                    d.qpos[:]=packet['qpos'];d.qpos[address:address+3]+=xyz-origin;d.qvel[:]=packet['qvel'];d.ctrl[:]=packet['ctrl'];d.time=packet['time_s'];mujoco.mj_forward(m,d);checks+=1
                    for contact in d.contact:
                        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g))for g in(contact.geom1,contact.geom2)]
                        if not contact_permitted(names,contact.dist,'approach'):bad=dict(step=packet['step'],pair=names,distance_m=float(contact.dist));break
                    if bad:break
                if forbidden:raise RuntimeError('STATIC_PREPARATION_INTEGRATION_ATTEMPT')
            if bad:rejected.append(dict(family=family,xy_mm=xy,reason=bad));continue
            rank=len(selected);split='DEV'if rank<8 else'TEST';number=rank if rank<8 else rank-8;name='%s-%s-%02d'%(split,family,number);folder=scene_root/name;folder.mkdir();model=folder/'SCENE.private.mjb';mujoco.mj_saveModel(m,str(model),None)
            reloaded=mujoco.MjModel.from_binary_path(str(model));robot_contract(original,reloaded)
            for field in('cam_pos','cam_quat','cam_fovy','cam_bodyid','cam_mode','cam_targetbodyid'):np.testing.assert_array_equal(getattr(original,field),getattr(reloaded,field))
            row=dict(layout_id=name,split=split,family=family,xy_offset_mm=list(xy),object_xyz_m=xyz.tolist(),scene=pin(model),base_scene_sha256=digest(original_path),initial_qpos0_sha256=hashlib.sha256(m.qpos0.tobytes()).hexdigest(),initial_key0_sha256=hashlib.sha256(m.key_qpos[0].tobytes()).hexdigest(),initial_state_cluster=identity([family,*np.round(xyz,6).tolist()]),minimum_old_object_xy_separation_m=float(np.min(np.linalg.norm(old[:,:2]-xyz[:2],axis=1))),robot_contract=contract,saved_startup_FK_checks=5703,static_only=True)
            layouts.append(row);selected.append(row)
            for n in range(2):descriptors.append(dict(**row,episode_id=name+'-n%d'%n,reset=seed(name+'/reset/%d'%n),noise_seed=seed(name+'/noise/%d'%n),language=base_spec['language'],initialization='ORIGINAL_FROZEN_INITIALIZE_5703',task_tick_max=30000,total_tick_max=35703))
            write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='FREEZE_SPLITS',layouts=len(layouts),passive_FK_calls=checks,new_generations=0,new_integrations=0))
        if len(selected)!=18:raise ValueError('INDEPENDENT_LAYOUT_POOL_UNDERFILLED_'+family)
    split=dict(schema='SLA_MOBILE_V52_GEOMETRY_CLUSTER_SPLIT',plan_commit=PLAN,created_epoch=time.time(),frozen_before_training=True,training_updates_before_freeze=0,DEV_count=32,TEST_count=40,minimum_object_xy_cluster_separation_m=.005,exclusion=pin(root/'EXCLUSION_GEOMETRY.private.json'),layouts=layouts,episodes=descriptors,domain='LOCAL_OUTWARD_OBJECT_OFFSETS_8_TO_38_MM_SAME_TASK_MATERIAL_CAMERA',TEST_model_outputs_read=False,new_generations=0,new_integrations=0)
    assert len(descriptors)==72 and sum(x['split']=='DEV'for x in descriptors)==32
    write(root/'EVALUATION_SPLIT.private.json',split);write(root/'STATIC_REJECTS.private.json',rejected)
    screen=[];confirmation=[];corrections=[]
    for i,(family,layout)in enumerate((f,'T%d'%j)for f in('F1','F2')for j in range(4)):
        s=templates[(family,layout)];common=dict(family=family,layout=layout,scene=dict(path=s['model_path'],sha256=s['frozen_files'][s['model_path']]),template_spec=pin(Path(s['registered_root'])/'EXECUTION_SPEC.json'),language=s['language'],initialization='ORIGINAL_FROZEN_INITIALIZE_5703',task_tick_max=30000,total_tick_max=35703)
        reset=520000+i;noise=seed('E2/%s/%s'%(family,layout));order=('B0','B1','B2')[i%3:]+('B0','B1','B2')[:i%3]
        for arm in order:screen.append(dict(**common,id='E2-%s-%s-%s'%(family,layout,arm),arm=arm,reset=reset,noise_seed=noise))
        confirmation.append(dict(**common,id='CONFIRM-%s-%s'%(family,layout),reset=521000+i,noise_seed=seed('CONFIRM/%s/%s'%(family,layout))))
        if family=='F2'or layout in('T0','T1'):
            for n in range(2):corrections.append(dict(**common,id='CORRECTION-%s-%s-%d'%(family,layout,n),reset=522000+i*2+n,noise_seed=seed('CORRECTION/%s/%s/%d'%(family,layout,n))))
    assert len(screen)==24 and len(confirmation)==8 and len(corrections)==12
    write(root/'E2_SCREEN.private.json',screen);write(root/'CONFIRM_PILOT.private.json',confirmation)
    corrections.sort(key=lambda r:(r['family']!='F2',r['reset']%2,r['layout'],r['reset']));write(root/'E3_CORRECTION_PANEL.private.json',corrections)
    d0=read(spec['D0_result']['path']);assert digest(spec['D0_result']['path'])==spec['D0_result']['sha256']
    saved=read(d0['artifacts']['ALL_QUERY_BOUNDARIES.private.json']['path']);chosen=balanced_roll(saved)
    original=read(spec['D0_spec']['path']);cases={}
    for item in original['V51_case_specs']:
        s=read(item['path']);cases[Path(s['registered_root']).name]=s
    roll=[]
    for index,r in enumerate(chosen):
        s=cases[r['case']];q=Path(s['registered_root'])/'native-01/shift7-queries'/('%04d'%r['query']);meta=read(q/'INPUT_META.private.json');request=read(q/'MODEL_REQUEST.private.json');response=read(q/'model-response/RESPONSE.json')
        roll.append(dict(index=index,case=r['case'],family=r['family'],query=r['query'],input=meta['input'],previous_reference=meta['previous_reference'],noise_id=request['noise_id'],language=request['language'],B0_saved_output=response['action'],query_metadata=pin(q/'INPUT_META.private.json')))
    write(root/'ROLL96.private.json',roll)
    result=dict(schema='SLA_MOBILE_V52_PREPARATION_ACTUAL',exit_code=0,source_sha=spec['source_sha'],split=pin(root/'EVALUATION_SPLIT.private.json'),ROLL96=pin(root/'ROLL96.private.json'),E2_screen=pin(root/'E2_SCREEN.private.json'),confirmation=pin(root/'CONFIRM_PILOT.private.json'),correction_panel=pin(root/'E3_CORRECTION_PANEL.private.json'),DEV_count=32,TEST_count=40,layouts=36,old_initial_centers=len(old),passive_FK_calls=checks,static_rejected=len(rejected),new_generations=0,new_native_requests=0,new_integrations=0,new_updates=0,new_visits=0,wall_s=time.time()-start,CPU_s=time.process_time())
    write(root/'RESULT.private.json',result);return result
