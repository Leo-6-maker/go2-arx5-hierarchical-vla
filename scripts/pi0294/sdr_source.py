"""Extract only frozen source commands; measured states remain separate evidence."""
import hashlib
from pathlib import Path
import numpy as np
from .task_capture_runtime import read,sha
from .sdr_runtime import FOLDER,require_adoption
from go2arx5_vla.trajectory.so3 import geodesic_angle,matrix_to_quat,quat_to_matrix,quat_slerp

def tool_binding(model,data,*,prefix=''):
    import mujoco
    site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'x5_ee')
    tcp=np.eye(4);tcp[:3,:3]=data.site_xmat[site].reshape(3,3);tcp[:3,3]=data.site_xpos[site]
    centers=[];axes=[];sizes=[];limits=[]
    for side in ['left','right']:
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,prefix+side+'_slide')
        g=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,prefix+side+'_pad')
        if min(j,g)<0:raise ValueError('SOURCE_TOOL_NAMES_MISSING')
        axis=tcp[:3,:3].T@data.xaxis[j]
        center=tcp[:3,:3].T@(data.geom_xpos[g]-tcp[:3,3])-axis*data.qpos[model.jnt_qposadr[j]]
        centers.append(center);axes.append(axis);sizes.append(model.geom_size[g].copy());limits.append(model.jnt_range[j].copy())
    closing=axes[0]/np.linalg.norm(axes[0]);x=np.array([1.,0.,0.]);x-=closing*(x@closing);x/=np.linalg.norm(x)
    transform=np.eye(4);transform[:3,:3]=np.column_stack([x,closing,np.cross(x,closing)]);transform[:3,3]=np.mean(centers,axis=0)
    if not np.allclose(axes[0],-axes[1],atol=1e-10) or not np.allclose(limits,[[0,.044],[0,.044]],atol=1e-10):raise ValueError('SOURCE_TOOL_STROKE_OR_AXIS')
    return dict(C_tcp_pad=transform.tolist(),zero_pad_centers_tcp=np.asarray(centers).tolist(),
                closing_axes_tcp=np.asarray(axes).tolist(),pad_half_sizes=np.asarray(sizes).tolist(),limits=np.asarray(limits).tolist(),
                measured_TCP=tcp.tolist(),physics_steps=0)

def extract(root):
    import mujoco
    from scripts.pi0292.durable import write
    root=Path(root);manifest,parent=__import__('scripts.pi0294.task_capture_runtime',fromlist=['validate_entry']).validate_entry(root,read_only=True)
    require_adoption(root,manifest);p=root/FOLDER;output=p/'SOURCE_EXTRACTION.json'
    if output.exists():return read(output)
    bank=Path(read(p/'SOURCE_BINDING.private.json')['source_bank']);plan=read(p/'SOURCE_PLAN.json')
    if sha(bank/'DATASET_MANIFEST.json')!=plan['own_manifest_sha256'] or sha(bank/'statistics.json')!=plan['own_statistics_sha256']:raise ValueError('SOURCE_BANK_IDENTITY_CHANGED')
    original=read(bank/'DATASET_MANIFEST.json');records=[];folder=p/'sources';folder.mkdir(exist_ok=True)
    for entry in plan['source_episodes']:
        row=next(r for r in original['episodes'] if r['id']==entry['id'])
        if row['split']!='TRAIN' or row.get('origin')!='new' or sha(bank/row['file'])!=entry['numeric_sha256']:raise ValueError('SOURCE_TRAIN_SELECTION_CHANGED')
        raw=Path(row['raw_root'])
        for name,digest in entry['raw_source_hashes'].items():
            if sha(raw/name)!=digest:raise ValueError('SOURCE_RAW_MEMBER_CHANGED:'+name)
        if not read(raw/'SCORE.json')['persistent_success'] or not read(raw/'AUDIT.json')['admitted']:raise ValueError('SOURCE_PRIOR_SUCCESS_REQUIRED')
        samples=read(raw/'samples.json');refs=read(raw/'references.json');events=read(raw/'TEACHER_EVENTS.json');card=read(raw/'CARD.json')
        if len(samples)!=641 or len(refs)!=80 or samples[-1]['supervision'] is not None:raise ValueError('SOURCE_641_640_PREFIX_IDENTITY')
        poses=[];grip=[];actual=[];obj=[];actual_grip=[]
        for i,sample in enumerate(samples):
            actual.append(sample['execution']['actual_tcp']);obj.append(sample['privileged']['object_pose']);actual_grip.append(sample['execution']['actual_gripper_gap_m'])
            if i==640:continue
            c=sample['supervision'];ref=refs[i//8];point=i%8
            if (not c['actually_committed'] or not c['valid'] or c['sequence']!=i or c['point_index']!=point or c['ref_id']!=ref['ref_id']
                or ref['committed_prefix_count']!=8 or abs(c['command_time_s']-i*.05)>1e-8):raise ValueError('SOURCE_CAUSAL_PREFIX')
            if not np.array_equal(c['committed'],ref['planned_H40'][point]) or c['committed_gripper']!=ref['gripper_H40'][point]:raise ValueError('SOURCE_INSTALLED_REFERENCE_CHANGED')
            poses.append(c['committed']);grip.append(c['committed_gripper'])
        poses,grip,actual,obj,actual_grip=map(np.asarray,(poses,grip,actual,obj,actual_grip))
        if not all(np.isfinite(v).all() for v in [poses,grip,actual,obj,actual_grip]):raise ValueError('SOURCE_NONFINITE')
        first_stage='align' if entry['task']=='T1' else 'close';transport='carry' if entry['task']=='T1' else 'push'
        start=next(e['time_s'] for e in events if e['stage']==first_stage);end=next(e['time_s'] for e in events if e['stage']=='hold')
        onset=next(e['time_s'] for e in events if e['stage']==transport)
        finish=next(e['time_s'] for e in events if e['previous']==transport)
        grid=np.arange(640)*.05
        cut0=int(np.searchsorted(grid,start));cut1=int(np.searchsorted(grid,end))
        if not 0<=cut0<cut1<640:raise ValueError('SOURCE_CUT_MUST_HAVE_COMMITTED_HOLD')
        files=read(raw/'FILE_MANIFEST.json');xml_path=raw/'scene.xml.json';xml_member=next(r for r in files if r['path']=='scene.xml.json')
        if sha(xml_path)!=xml_member['sha256']:raise ValueError('SOURCE_MODEL_WRAPPER_IDENTITY')
        text=read(xml_path)
        if hashlib.sha256(text['xml'].encode()).hexdigest()!=text['sha256']:raise ValueError('SOURCE_MODEL_XML_IDENTITY')
        model=mujoco.MjModel.from_xml_string(text['xml']);data=mujoco.MjData(model)
        for j,name in enumerate(['x5_joint'+str(i) for i in range(1,7)]+['left_slide','right_slide']):
            joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
            source_values=np.r_[samples[0]['execution']['actual_joint_q'],samples[0]['execution']['actual_finger_q']]
            data.qpos[model.jnt_qposadr[joint]]=source_values[j]
        mujoco.mj_forward(model,data);binding=tool_binding(model,data)
        if not np.allclose(binding['measured_TCP'],actual[0],atol=1e-10,rtol=0):raise ValueError('SOURCE_TCP_MODEL_FRAME_MISMATCH')
        name=entry['id']+'.npz';target=folder/name
        np.savez_compressed(target,source_time=grid,committed_TCP=poses,committed_gripper=grip,
            actual_TCP=actual,actual_object=obj,actual_gripper_gap_m=actual_grip)
        record=dict(id=entry['id'],task=entry['task'],status='PASS_EXTRACTED_ORIGINAL_COMMITTED_SOURCE',split='TRAIN',
            observations=641,unique_committed_rows=640,original_H40_references=80,
            source_numeric_sha256=entry['numeric_sha256'],raw_source_hashes=entry['raw_source_hashes'],
            source_model_xml_sha256=text['sha256'],derived_file=name,derived_sha256=sha(target),
            cut_start_index=cut0,cut_end_inclusive_index=cut1,transport_start_s=onset,transport_end_s=finish,
            cut_duration_s=float(grid[cut1]-grid[cut0]),source_card=card,events=events,tool_binding=binding,
            sample_command_parity='EXACT_ALL_640',new_physics=0,new_IK_queries=0)
        records.append(record);write(p/'SOURCE_EXTRACTION.partial.json',dict(records=records,source_sha=manifest['code']['vla_commit']))
    value=dict(status='PASS_SIX_FROZEN_TRAIN_SOURCE_EXTRACTIONS',source_sha=manifest['code']['vla_commit'],records=records,new_physics=0,new_global_IK_calls=0,new_render=0,new_model_queries=0)
    write(output,value);return value

def retarget(record,arrays,target_state,target_card,target_tool):
    source_tool=np.asarray(record['tool_binding']['C_tcp_pad']);target_tool=np.asarray(target_tool['C_tcp_pad'])
    source_pose=np.asarray(arrays['committed_TCP']);pads=source_pose@source_tool
    os=np.asarray(arrays['actual_object'])[0,:3,3];ot=np.asarray(target_state['object_pose'])[:3,3]
    gs=np.r_[record['source_card']['goal_xy'],os[2]];gt=np.r_[target_card['goal_xy'],ot[2]]
    ds=gs-os;dt=gt-ot
    if min(np.linalg.norm(ds[:2]),np.linalg.norm(dt[:2]))<=1e-10:raise ValueError('SOURCE_TASK_DIRECTION_ZERO')
    yaw=np.arctan2(dt[1],dt[0])-np.arctan2(ds[1],ds[0]);c,s=np.cos(yaw),np.sin(yaw);r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    grid=np.asarray(arrays['source_time']);a=int(np.searchsorted(grid,record['transport_start_s']));b=int(np.searchsorted(grid,record['transport_end_s']))
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(pads[a:b+1,:3,3],axis=0),axis=1))]
    if arc[-1]<=1e-12:raise ValueError('SOURCE_ZERO_TRANSPORT_ARC')
    u=np.ones(640);u[:a]=0;u[a:b+1]=arc/arc[-1]
    out=pads.copy();out[:,:3,3]=ot+(pads[:,:3,3]-os)@r.T+u[:,None]*(gt-ot-r@(gs-os))
    out[:,:3,:3]=r@pads[:,:3,:3];out=out@np.linalg.inv(target_tool)
    lo,hi=record['cut_start_index'],record['cut_end_inclusive_index'];out=out[lo:hi+1];time_grid=grid[lo:hi+1]-grid[lo]
    v=np.linalg.norm(np.diff(out[:,:3,3],axis=0),axis=1)/.05
    w=np.asarray([geodesic_angle(x,y) for x,y in zip(out[:-1,:3,:3],out[1:,:3,:3])])/.05
    slowdown=max(1.,float(v.max(initial=0)/.06),float(w.max(initial=0)/.3))
    return dict(poses=out,gripper=np.asarray(arrays['committed_gripper'])[lo:hi+1],time=time_grid*slowdown,
        source_indices=np.arange(lo,hi+1),slowdown=slowdown,source_cut_start_s=float(grid[lo]),
        raw_max_linear_speed_m_s=float(v.max(initial=0)),raw_max_angular_speed_rad_s=float(w.max(initial=0)),
        active_duration_s=float(time_grid[-1]*slowdown),task_rotation=r,transport_weight=u[lo:hi+1])
