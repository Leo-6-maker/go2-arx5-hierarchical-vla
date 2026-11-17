"""Read the fixed V5.1 corpus and C20; no model, controller or mj_step call."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scripts.pi0292.durable import write
from go2arx5_vla.world_model.storage import digest


def read(path):return json.loads(Path(path).read_bytes())
def lines(path):return [json.loads(x)for x in Path(path).read_text().splitlines()]
def pin(path):return dict(path=str(path),sha256=digest(path))


def boundary(rows):
    p=np.asarray(rows,float)[...,:3]
    velocity=np.diff(p,axis=-2)/.05
    acceleration=np.diff(np.concatenate((np.zeros_like(velocity[...,:1,:]),velocity,np.zeros_like(velocity[...,:1,:])),axis=-2),axis=-2)/.05
    norms=np.linalg.norm(acceleration,axis=-1)
    return dict(start=float(norms[0]),end=float(norms[-1]),interior=float(max(norms[1:-1])),
        start_margin=float(.25-norms[0]),end_margin=float(.25-norms[-1]),
        peak=float(max(norms)),peak_index=int(np.argmax(norms)))


def packet_range(path,lower,upper):
    import struct
    from scripts.pi0294.task_state import parse_state
    with Path(path).open('rb')as f:
        while length:=f.read(4):
            n=struct.unpack('<I',length)[0]
            if not 40<=n<=2*1024**2:raise ValueError('RECORDED_PACKET_LENGTH')
            head=f.read(16);step=struct.unpack_from('<Q',head,8)[0]
            if step>upper:break
            if step<lower:f.seek(n-16,1);continue
            raw=head+f.read(n-16)
            if len(raw)!=n:raise ValueError('TRUNCATED_PACKET')
            yield parse_state(raw)


def geometry(model,data):
    import mujoco
    cube=model.geom('task_cube_geom').id;table=model.geom('task_table').id
    wrist=[i for i in range(model.ngeom)if model.geom_bodyid[i]==model.body('x5_link6').id and model.geom_contype[i]]
    pads=[model.geom(n).id for n in('task_left_pad','task_right_pad')]
    def distance(a,b):return float(mujoco.mj_geomDistance(model,data,a,b,1.,None))
    return dict(wrist_object_m=min(distance(g,cube)for g in wrist),wrist_table_m=min(distance(g,table)for g in wrist),
        pad_object_m=[distance(g,cube)for g in pads],pad_table_m=[distance(g,table)for g in pads])


def contact_analysis(case,spec,result,experts,root):
    import mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration
    end=result['actual_native_integrations'];lower=end-2500;native=case/'native-01'
    model=mujoco.MjModel.from_binary_path(spec['model_path']);data=mujoco.MjData(model)
    assert mujoco.get_mjcb_control()is None
    rows=[];last=None;count=0
    with forbid_integration(mujoco)as forbidden:
        for p in packet_range(native/'state.bin.journal',lower,end):
            if time.time()>=spec['D0_deadline_epoch']:raise TimeoutError('FINITE_READ_ONLY_CONTACT_AUDIT')
            data.qpos[:]=p['qpos'];data.qvel[:]=p['qvel'];data.ctrl[:]=p['ctrl'];data.time=p['time_s'];mujoco.mj_forward(model,data);count+=1
            tcp=model.site('x5_ee').id;mount=model.body('x5_base_link').id
            position=data.site_xpos[tcp].copy();rotation=data.site_xmat[tcp].reshape(3,3).copy();base=data.xpos[mount].copy();R=data.xmat[mount].reshape(3,3).copy();relative=R.T@(position-base)
            contribution=None
            if last:
                b0,R0,r0,x0=last;parts=np.stack((base-b0,(R-R0)@r0,R@(relative-r0)))
                if not np.allclose(parts.sum(0),position-x0,rtol=0,atol=1e-12):raise ValueError('EXACT_FK_DISPLACEMENT_DECOMPOSITION')
                contribution=parts.tolist()
            last=(base,R,relative,position)
            rows.append(dict(step=p['step'],time_s=p['time_s'],**geometry(model,data),TCP=position.tolist(),TCP_rotation=rotation.tolist(),mount=base.tolist(),base_translation_rotation_arm_displacement=contribution))
        if forbidden:raise RuntimeError('READ_ONLY_DIAGNOSTIC_ATTEMPTED_INTEGRATION')
    assert count==2501
    final=rows[-1];objects=data.xpos[model.body('task_cube').id].copy();best=None
    for expert in experts:
        if expert['family']!=spec['mobile_condition']['family']:continue
        for sample in expert['samples']:
            tcp=np.asarray(sample['actual_TCP']);obj=np.asarray(sample['object_pose']);distance=float(np.linalg.norm((tcp[:3,3]-obj[:3,3])-(np.asarray(final['TCP'])-objects)))
            key=(distance,expert['id'],sample['native_end_step'])
            if best is None or key<best[0]:best=(key,expert,sample)
    key,expert,sample=best
    e_model=mujoco.MjModel.from_binary_path(expert['spec']['model_path']);e_data=mujoco.MjData(e_model)
    with forbid_integration(mujoco)as forbidden:
        e_data.qpos[:]=sample['qpos'];e_data.qvel[:]=sample['qvel'];mujoco.mj_forward(e_model,e_data)
        near=dict(expert_id=expert['id'],native_step=sample['native_end_step'],relative_TCP_object_distance_m=key[0],geometry=geometry(e_model,e_data),native_grasp=sample['native_grasp'],same_full_control_state=False,BC_label_eligible=False)
        if forbidden:raise RuntimeError('EXPERT_STATIC_COMPARISON_ATTEMPTED_INTEGRATION')
    write(root/'CONTACT_5S.private.json',dict(states=rows,nearest_expert_by_relative_TCP_object_position=near,full_control_restore=False,static_geometry_only=True))
    return dict(recorded_ticks=count,first_step=lower,last_step=end,first_illegal_tick=end,final_geometry={k:v for k,v in final.items()if k.endswith('_m')},minimum_wrist_object_m=min(r['wrist_object_m']for r in rows),minimum_wrist_table_m=min(r['wrist_table_m']for r in rows),nearest_expert=near,artifact=pin(root/'CONTACT_5S.private.json'),new_integrations=0,controller_selected_reference='UNKNOWN')


def run(spec):
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,rotation_matrix_to_rot6d_rows
    from go2arx5_vla.world_model.fusion_trajectory_adapter import _geometry
    root=Path(spec['registered_root']);started=time.time();assert spec['new_model_generations_budget']==spec['new_physical_requests_budget']==0
    previous=read(spec['V51_terminal']['path']);assert digest(spec['V51_terminal']['path'])==spec['V51_terminal']['sha256']
    cases=[]
    for pin_ in spec['V51_case_specs']:
        s=read(pin_['path']);assert digest(pin_['path'])==pin_['sha256'];case=Path(s['registered_root']);r=read(case/'RESULT.private.json')
        assert digest(case/'RESULT.private.json')==pin_['result_sha256'];cases.append((case,s,r))
    experts=[];expert_rows=[]
    for case,s,r in cases:
        if s['sim_only_role']!='EXPERT':continue
        samples=lines(case/'native-01/MOBILE_SAMPLES.private.jsonl');commands=lines(case/'native-01/COMMANDS.private.jsonl')
        for c in commands:
            if c.get('phase')=='mobile-continuous-expert-query':expert_rows.append(dict(id=case.name,family=s['mobile_condition']['family'],query=c['query'],stage=c['stage'],**boundary(c['submitted_policy_H40'])))
        experts.append(dict(id=case.name,family=s['mobile_condition']['family'],spec=s,samples=samples))
    assert len(experts)==8
    all_rows=[];inputs=0;actions=0;contacts=[]
    for case,s,r in cases:
        if s['sim_only_role']!='STUDENT':continue
        samples={x['native_end_step']:x for x in lines(case/'native-01/MOBILE_SAMPLES.private.jsonl')};base0=np.asarray(samples[5703]['actual_mount'])[:3,3]
        for q in sorted((case/'native-01/shift7-queries').iterdir()):
            meta=read(q/'INPUT_META.private.json');p=meta['input'];assert digest(p['path'])==p['sha256'];inputs+=1
            with np.load(p['path'],allow_pickle=False)as z:state=z['state_eef10'];frame=z['policy_from_world'];reference=z['accepted_reference_TCP'];body=z['body_rgb'];wrist=z['wrist_rgb']
            def visibility(image):return int(np.sum((image[...,0]>100)&(image[...,0]>1.5*image[...,1])&(image[...,0]>1.5*image[...,2])))
            sample=samples[meta['native_step']];base=np.asarray(sample['actual_mount'])[:3,3];actual=np.linalg.inv(frame)@matrix10(state)
            row=dict(case=case.name,family=s['mobile_condition']['family'],query=int(q.name),native_step=meta['native_step'],input_sha256=p['sha256'],body_red_pixel_proxy=visibility(body),wrist_red_pixel_proxy=visibility(wrist),proxy_is_object_recognition=False,base_directional_m=float((base-base0)[0 if s['mobile_condition']['family']=='F1'else 1]),reference_actual_position_error_m=float(np.linalg.norm(reference[:3,3]-actual[:3,3])))
            if(q/'model-response/RESPONSE.json').exists():
                response=read(q/'model-response/RESPONSE.json');a=response['action'];assert digest(a['path'])==a['sha256'];actions+=1
                with np.load(a['path'],allow_pickle=False)as z:raw=z['physical_H40']
                target=pose10(np.linalg.inv(frame)@matrix10(raw),raw[:,9]);desired=target.copy();desired[:,9]=(.002+.040*target[:,9])/.044;installed=desired.copy();installed[:,:3]=reference[:3,3]+target[:,:3]-actual[:3,3];installed[:,3:9]=rotation_matrix_to_rot6d_rows(reference[:3,:3]@actual[:3,:3].T@rot6d_rows_to_matrix(target[:,3:9]))
                left=_geometry(desired,actual);right=_geometry(installed,reference)
                keys=('maximum_translation_speed_m_s','maximum_discrete_translation_acceleration_m_s2','maximum_finger_speed_m_s')
                error=max(abs(left[k]-right[k])for k in keys)
                if error>1e-10:raise ValueError('DESIRED_REFERENCE_DERIVATIVE_INVARIANCE')
                width=.002+.040*raw[:,9]
                row.update(action_sha256=a['sha256'],**boundary(raw),desired_geometry=left,reference_geometry=right,desired_reference_derivative_error=error,first_width_to_previous_command_speed_m_s=float((width[0]-meta['previous_reference']['width_m'])/.05),first_desired_actual_position_error_m=float(np.linalg.norm(target[0,:3]-actual[:3,3])),consumption=read(q/'QUERY_ACTUAL.private.json'))
            all_rows.append(row)
        if r['unsafe']:
            ss=dict(s,D0_deadline_epoch=spec['deadline_epoch']);contacts.append(contact_analysis(case,ss,r,experts,root))
        write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='D0_SAVED_INPUT_ACTIONS',inputs=inputs,actions=actions,new_generations=0,new_integrations=0))
    assert inputs==320 and actions==319
    dataset=Path(spec['dataset']);manifest=read(dataset/'MANIFEST.private.json');assert digest(dataset/'MANIFEST.private.json')==spec['dense_manifest_sha256'];label_rows=[]
    for ep in manifest['episodes']:
        p=dataset/ep['labels']['file'];assert digest(p)==ep['labels']['sha256']
        with np.load(p,allow_pickle=False)as z:
            for i,raw in enumerate(z['nominal_policy_H40']):label_rows.append(dict(episode=ep['id'],family=ep['family'],anchor=i,phase=str(z['phase'][i]),**boundary(raw)))
    assert len(label_rows)==1376
    recorded_rejects=[r for r in all_rows if r.get('consumption')and not r['consumption']['full_H40_compiled']]
    assert len(recorded_rejects)==3
    for row in all_rows:
        if 'desired_geometry'not in row:continue
        g=row['desired_geometry'];limits={'maximum_translation_speed_m_s':.05,'maximum_angular_speed_rad_s':.25,'maximum_finger_speed_m_s':.04,'maximum_discrete_translation_acceleration_m_s2':.25,'maximum_discrete_angular_acceleration_rad_s2':1.}
        passed=all(g[k]<=v*(1+1e-10)for k,v in limits.items());assert passed==row['consumption']['full_H40_compiled']
    write(root/'ALL_QUERY_BOUNDARIES.private.json',all_rows);write(root/'C20_LABEL_BOUNDARIES.private.json',label_rows);write(root/'EXPERT_BOUNDARIES.private.json',expert_rows)
    def summary(rows):
        return {key:dict(min=float(np.min([r[key]for r in rows])),median=float(np.median([r[key]for r in rows])),p95=float(np.percentile([r[key]for r in rows],95)),max=float(np.max([r[key]for r in rows])))for key in ('start','end','interior','start_margin','end_margin')}
    generated=[r for r in all_rows if'action_sha256'in r]
    result=dict(schema='SLA_MOBILE_V52_D0_ACTUAL',exit_code=0,inputs=inputs,actions=actions,C20_windows=len(label_rows),expert_episodes=8,expert_H40=len(expert_rows),saved_actions_by_family={f:summary([r for r in generated if r['family']==f])for f in ('F1','F2')},C20_summary=summary(label_rows),expert_summary=summary(expert_rows),C20_endpoint_exceeds=sum(r['start']>.25*(1+1e-10)or r['end']>.25*(1+1e-10)for r in label_rows),contact=contacts,artifacts={n:pin(root/n)for n in ('ALL_QUERY_BOUNDARIES.private.json','C20_LABEL_BOUNDARIES.private.json','EXPERT_BOUNDARIES.private.json')},controller_selected_reference='UNKNOWN',full_control_state_counterfactual=False,expert_student_BC_pairing=False,visibility='SAVED_RGB_RED_PIXEL_PROXY_NOT_SEMANTIC_RECOGNITION',new_updates=0,new_optimization_visits=0,new_generations=0,new_native_requests=0,new_integrations=0,wall_s=time.time()-started,CPU_s=time.process_time())
    write(root/'RESULT.private.json',result);return result
