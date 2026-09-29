"""Read the actual PI0259_H0_V1 sealed layout; no producer module execution."""
import hashlib
import html
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from pi0259_review import Packet,require,pose,project,read,write,sha,text,ZERO
from evidence_so3 import rot6d_rows_to_matrix,geodesic_angle


def raster(data, ids, task, command, start):
    counts={name:[int(np.count_nonzero(data[s+'_segmentation']==sid)) for s in ('body','wrist')] for name,sid in ids.items()}
    need={f'target{command}':4,f'axis{command}_0':2,f'axis{command}_1':2} if task=='M1' else {'obstacle':16,'goal':4} if task=='M2' else {'black_A':4,'white_B':4}
    failures=[n for n,limit in need.items() if max(counts[n])<limit]
    color_means={};centers=[]
    if task=='M1':
        for axis in (0,1):
            name=f'axis{command}_{axis}';stream=('body','wrist')[int(np.argmax(counts[name]))]
            mask=data[stream+'_segmentation']==ids[name];pixels=data[stream+'_rgb'][mask]
            mean=pixels.mean(axis=0) if len(pixels) else np.zeros(3)
            color_means[name]=mean.tolist()
            if not len(pixels) or mean[axis]<=max(np.delete(mean,axis)):failures.append(name+'_COLOR')
            centers.append((stream,np.argwhere(mask).mean(axis=0) if mask.any() else None))
        if centers[0][0]==centers[1][0] and all(x[1] is not None for x in centers) and np.array_equal(centers[0][1],centers[1][1]):failures.append('AXIS_CENTROIDS_IDENTICAL')
    sectors=sum(any(max(counts[f'contour{i}'])>0 for i in (2*s,2*s+1)) for s in range(12))
    if task=='M3' and start and sectors<8:failures.append('CONTOUR_LT8_SECTORS')
    return dict(necessary_pass=not failures,failures=failures,required=need,counts=counts,axis_rgb_means=color_means,sectors=sectors)


def make_card(data,record,checks,name,out,digest):
    canvas=np.full((720,1280,3),20,np.uint8)
    text(canvas,'KINEMATIC REFERENCE - NO WBC EXECUTION',20,28,.65)
    text(canvas,name,20,53,.4)
    text(canvas,f"rig {record['rig_index']} | synthetic t={record['synthetic_reference_time']:.2f}s | 96x96 source RGB; nearest display 3x",20,80,.47)
    for j,s in enumerate(('body','wrist')):
        x=20+j*325
        canvas[124:412,x:x+288]=cv2.resize(data[s+'_rgb'][...,::-1],(288,288),interpolation=cv2.INTER_NEAREST)
        text(canvas,s+' POLICY RGB',x,115,.45)
        seg=data[s+'_segmentation'].astype(np.int64)
        rgb=np.stack([(seg*k)%251 for k in (31,71,131)],axis=-1).astype(np.uint8)
        canvas[454:646,x:x+192]=cv2.resize(rgb,(192,192),interpolation=cv2.INTER_NEAREST)
        text(canvas,'DIAGNOSTIC segmentation',x,443,.4)
        depth=np.abs(data[s+'_depth']);finite=np.isfinite(depth);scaled=np.zeros((96,96),np.uint8)
        if finite.any():scaled[finite]=np.clip(depth[finite]/max(float(depth[finite].max()),1e-6)*255,0,255).astype(np.uint8)
        canvas[454:550,x+196:x+292]=cv2.cvtColor(scaled,cv2.COLOR_GRAY2BGR)
        text(canvas,'depth m',x+196,573,.35)
    y=120
    for task in ('M1','M2','M3'):
        for command in (0,1):
            c=checks[task][str(command)];label='NECESSARY PASS' if c['necessary_pass'] else 'FAIL'
            text(canvas,f'{task} command{command}: {label}',690,y,.47);y+=22
            text(canvas,', '.join(c['failures'])[:79] or 'Natural RGB discernibility still separate',690,y,.35);y+=21
            text(canvas,' '.join(f'{k}:{max(c["counts"][k])}/{v}px' for k,v in c['required'].items()),690,y,.34);y+=38
    text(canvas,'Source mask counts; no bounding-box visibility inference. No model-driven motion.',20,677,.43)
    text(canvas,'manifest SHA256 '+digest,20,708,.38)
    require(cv2.imwrite(str(out),canvas),'card write failed')


def review_h0(manifest,digest,out,urdf_path):
    packet=Packet(manifest,digest);out=Path(out).resolve()
    require(not out.is_relative_to(packet.root) and not packet.root.is_relative_to(out),'OUTPUT_OVERLAPS_SEALED_PACKET')
    require(not out.exists(),'new companion output directory required')
    binding=read(packet.file('design/FK_source_binding.json'))
    require(sha(urdf_path)==binding['urdf']['sha256'],'URDF binding mismatch')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from pi0258_geometry import full_fk
    from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics,homogeneous,rpy_to_matrix
    urdf=ET.parse(urdf_path).getroot();kin=NativeArmKinematics.from_urdf(urdf_path)
    reg=read(packet.file('design/registry.json'));cue_map={r['family_id']:r['cues'] for r in read(packet.file('design/cue_geometry.json'))}
    out.mkdir(parents=True);records=[];findings=[];geometry=[];max_fk=0.;max_state=0.;max_camera=0.;projection_rows=[];repeat_pairs=0
    # Validate every saved numeric reference and FK prefix; failed NaN suffixes remain missing.
    for original in packet.m['outcome_index']:
        sid=original['sequence']['sequence_id'];name=f'calibration/{sid}/geometry.npz'
        with np.load(packet.file(name),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        valid=np.isfinite(a['joints']).all(axis=1);root=pose(a['root_transform']);ts=a['synthetic_reference_time']
        require(np.all(np.diff(ts)>0),'geometry clock invalid')
        independently=np.array([root@kin.tcp_pose(q) for q in a['joints'][valid]])
        require(np.allclose(independently,a['FK_world'][valid],atol=1e-9,rtol=0),'geometry FK mismatch: '+name)
        require(np.allclose(a['initial_TCP_world'],independently[0],atol=1e-9,rtol=0),'initial FK mismatch')
        require(np.allclose(a['desired_position'][0],independently[0,:3,3],atol=1e-9,rtol=0) and np.allclose(a['desired_rotation'][0],independently[0,:3,:3],atol=1e-9,rtol=0),'reference start not FK')
        pos=np.linalg.norm(independently[:,:3,3]-a['desired_position'][valid],axis=1)
        rot=geodesic_angle(independently[:,:3,:3],a['desired_rotation'][valid])
        feasible=valid.copy();feasible[valid]=(pos<=.002+1e-12)&(rot<=.01+1e-12)
        idx=a['active_indices'];windows=np.array([np.all(feasible[np.minimum(i+np.arange(40),len(feasible)-1)]) for i in idx])
        motion={'M1':6,'M2':10,'M3':16}[original['sequence']['task']]
        quarters=[bool(windows[(ts[idx]>=q*motion/4)&(ts[idx]<(q+1)*motion/4)].any()) for q in range(4)]
        ledger=read(packet.file(f'calibration/{sid}/solver_ledger.json'))
        contacts=[c for solve in ledger['attempts'] for c in solve['checks'].get('contacts',[])]
        geometry.append(dict(source_id=sid,task=original['sequence']['task'],original_sha256=packet.files[name]['sha256'],saved_valid_prefix=int(valid.sum()),independent_fk_position_max_m=float(pos.max()),independent_fk_so3_max_rad=float(rot.max()),window_fraction=float(windows.mean()),quartiles=quarters,pose_prefix_qualified=bool(windows.mean()>=.95 and all(quarters)),producer_status=original['status'],collision_contacts_as_recorded=contacts,collision_recomputed=False))
    names=sorted((n for n in packet.files if n.startswith('calibration_rigs/') and n.endswith('.json')),key=lambda n:hashlib.sha256(n.encode()).hexdigest())
    for name in names:
        record=read(packet.file(name))
        if 'array_file' not in record:continue
        array_name=str(Path(name).parent/record['array_file']).replace('\\','/')
        require(packet.files[array_name]['sha256']==record['array_sha256'],'record array binding mismatch')
        with np.load(packet.file(array_name),allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        require(np.array_equal(data['root_transform_world'],record['root_transform_world']) and np.array_equal(data['root_transform_world'],binding['root_transform']),'root binding mismatch')
        require(all(record['joint_positions'][k]==v for k,v in binding['default_joints'].items() if k not in kin.joint_names),'non-arm nominal DOFs changed')
        if name.endswith('/common_start_repeat.json'):
            common_name=str(Path(name).parent/'common_start.npz').replace('\\','/')
            with np.load(packet.file(common_name),allow_pickle=False) as common:
                require(set(common.files)==set(data) and all(np.array_equal(common[k],data[k],equal_nan=True) for k in common.files),'COMMON_START_REPEAT_CHANGED')
            repeat_pairs+=1
        poses=full_fk(urdf,record['joint_positions'],pose(record['root_transform_world']))
        tcp=poses['x5_link6']@homogeneous(np.eye(3),np.array(binding['tcp_offset_xyz']))
        delta=max(float(np.max(abs(tcp-pose(data['FK_world'])))),float(np.max(abs(tcp-pose(record['FK_TCP_world_matrix'])))))
        max_fk=max(max_fk,delta);require(delta<1e-5,'image/FK mismatch: '+name)
        require(all(np.max(abs(poses[k]-pose(v)))<1e-5 for k,v in record['link_transforms_world'].items()),'link FK mismatch')
        native=np.linalg.inv(pose(data['T_world_native']))@tcp;state=data['state_v1']
        delta=max(float(np.max(abs(state[:3]-native[:3,3]))),float(geodesic_angle(rot6d_rows_to_matrix(state[3:9]),native[:3,:3])))
        max_state=max(max_state,delta);require(delta<1e-5 and state[9]==.5,'native StateV1 mismatch')
        require(np.array_equal(data['arm_joints'],[record['joint_positions'][n] for n in kin.joint_names]),'render q mismatch')
        for stream in ('body','wrist'):
            rgb=data[stream+'_rgb'];seg=data[stream+'_segmentation'];depth=data[stream+'_depth']
            require(rgb.shape==(96,96,3) and rgb.dtype==np.uint8 and seg.shape==depth.shape==(96,96),'source resolution changed')
            require(np.issubdtype(seg.dtype,np.integer),'mask not integer')
            view=np.array(record['camera_view'][stream]);proj=np.array(record['camera_projection'][stream]);camera=pose(np.linalg.inv(view.T))
            if stream=='body':
                expected=(np.array(record['root_transform_world'])@np.r_[reg['renderer']['rigs_base_positions'][record['rig_index']],1])[:3]
                axis=np.array(reg['renderer']['look_at_world'])-expected;axis/=np.linalg.norm(axis)
                require(np.allclose(camera[:3,3],expected,atol=1e-6,rtol=0) and np.allclose(-camera[:3,2],axis,atol=1e-6,rtol=0),'body camera extrinsic mismatch')
                right=np.cross(axis,[0.,0.,1.]);right/=np.linalg.norm(right)
                expected_rotation=np.column_stack((right,np.cross(right,axis),-axis))
                max_camera=max(max_camera,float(np.max(abs(expected_rotation-camera[:3,:3]))))
                require(np.allclose(expected_rotation,camera[:3,:3],atol=1e-6,rtol=0),'body camera up/rotation mismatch')
                require(abs(2*np.arctan(1/proj[0,0])*180/np.pi-70)<1e-4,'body FOV mismatch')
            else:
                expected=poses['x5_link6']@homogeneous(rpy_to_matrix(np.array([0,3*np.pi/4,0])),np.array([.28,-.10,.12]))
                require(np.allclose(camera[:3,3],expected[:3,3],atol=1e-6,rtol=0),'wrist camera attachment mismatch')
                # Isaac camera attachment axes -> OpenGL optical right/up/back axes.
                convention=np.array([[0,0,-1],[-1,0,0],[0,1,0]])
                delta=float(np.max(abs(camera[:3,:3]-expected[:3,:3]@convention)));max_camera=max(max_camera,delta)
                require(delta<1e-6,'wrist full camera rotation mismatch')
            ids=record['scene_ids'];cues=cue_map[record['family_id']]
            for cue,sid in ids.items():
                mask=seg==sid
                if mask.any():require(np.isfinite(depth[mask]).all() and np.all(depth[mask]<0),'cue depth sign/nonfinite')
                point=np.array(cues[cue]['position']);pixel,inside=project(point[None],view,proj,'row_vector_opengl')
                centroid=np.argwhere(mask)[:,::-1].mean(axis=0)+.5 if mask.any() else None
                camera_z=float((np.r_[point,1]@view)[2])
                surface_error=None
                if mask.any() and 'radius' in cues[cue]:
                    y,x=np.where(mask);z=depth[mask]
                    optical=np.c_[-z*((x+.5)/48-1)/proj[0,0],-z*(1-(y+.5)/48)/proj[1,1],z,np.ones(len(z))]
                    points=(optical@camera.T)[:,:3]
                    error=np.abs(np.linalg.norm(points-point,axis=1)-cues[cue]['radius'])
                    surface_error=dict(median_m=float(np.median(error)),max_m=float(error.max()))
                projection_rows.append(dict(source=name,stream=stream,cue=cue,pixels=int(mask.sum()),projected_xy=pixel[0].tolist() if np.isfinite(pixel).all() else None,centroid_xy=None if centroid is None else centroid.tolist(),center_depth_m=camera_z,raster_depth_range_m=[float(depth[mask].min()),float(depth[mask].max())] if mask.any() else None,depth_backprojected_sphere_surface_error=surface_error))
        checks={t:{str(c):raster(data,record['scene_ids'],t,c,record['synthetic_reference_time']==0) for c in (0,1)} for t in ('M1','M2','M3')}
        if record.get('sequence'):
            with np.load(packet.file(f'calibration/{record["sequence"]}/geometry.npz')) as a:
                i=record['geometry_index'];require(abs(a['synthetic_reference_time'][i]-record['synthetic_reference_time'])<1e-9 and np.array_equal(a['joints'][i],data['arm_joints']),'image/geometry clock/pose mismatch')
                require(np.array_equal(a['desired_position'][i],record['requested_position']) and np.array_equal(a['desired_rotation'][i],record['requested_rotation']),'requested reference binding mismatch')
        card_name=f'frame-{len(records):03d}.png';make_card(data,record,checks,name,out/card_name,digest)
        records.append(dict(source=name,source_sha256=packet.files[name]['sha256'],array=array_name,array_sha256=record['array_sha256'],rig=record['rig_index'],family=record['family_id'],time=record['synthetic_reference_time'],sequence=record.get('sequence'),task=record.get('task'),command=record.get('command'),checks=checks,card=card_name))
    rigs=[]
    for rig in range(3):
        starts=[r for r in records if r['rig']==rig and r['source'].endswith('/common_start.json')]
        start_tasks=[t for t in ('M1','M2','M3') if len(starts)==6 and all(r['checks'][t][str(c)]['necessary_pass'] for r in starts for c in (0,1))]
        later=[r for r in records if r['rig']==rig and r['sequence']]
        supported=[t for t in start_tasks if len([r for r in later if r['task']==t])==48 and all(r['checks'][t][str(r['command'])]['necessary_pass'] for r in later if r['task']==t)]
        per_task={t:dict(saved=sum(r['task']==t for r in later),registered_later=48 if len(start_tasks)>=2 else None,failed_visibility=sum(r['task']==t and not r['checks'][t][str(r['command'])]['necessary_pass'] for r in later),missing_registered=48-sum(r['task']==t for r in later) if len(start_tasks)>=2 else None) for t in ('M1','M2','M3')}
        rigs.append(dict(rig=rig,start_supported=start_tasks,full_supported=supported,later_samples=len(later),later_by_task=per_task))
    # Return concrete failed originals, not just the producer's camera-outcome label.
    for r in records:
        relevant=[(r['task'],r['command'])] if r['sequence'] else [(t,c) for t in ('M1','M2','M3') for c in (0,1)]
        for task,command in relevant:
            c=r['checks'][task][str(command)]
            if c['failures']:findings.append(dict(source=r['source'],sha256=r['source_sha256'],array_sha256=r['array_sha256'],task=task,command=command,failures=c['failures']))
    write(out/'raster_records.json',records);write(out/'projection_depth.json',projection_rows);write(out/'geometry_review.json',geometry)
    receipt=dict(status='CORRECTNESS_BLOCKED' if not any(len(r['full_supported'])>=2 for r in rigs) else 'H0_NUMERIC_REVIEW_PARTIAL',reason='COMMON_RIG_OBSERVABILITY_NOT_ADMITTED',producer_manifest_sha256=digest,reviewer_code_sha256=sha(Path(__file__)),entry_sha256=sha(Path(__file__).with_name('pi0259_review.py')),files_checked=len(packet.files),image_pairs=len(records),rigs=rigs,findings=findings,maximum_saved_fk_delta=max_fk,maximum_state_delta=max_state,maximum_camera_rotation_matrix_delta=max_camera,urdf_sha256=sha(urdf_path),new_calls=ZERO,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',limitations=['No independent FCL collision replay; saved contacts reported as recorded.','No TRAIN/processor/model package exists; H1/H2 remain pending.','Partial calibration snapshots are not complete trajectory videos.','Natural RGB inspection and explicit visual receipt accompany machine counts.'],geometry_pose_counts={t:sum(x['task']==t and x['pose_prefix_qualified'] for x in geometry) for t in ('M1','M2','M3')})
    write(out/'receipt.json',receipt)
    receipt['exact_common_start_repeat_pairs']=repeat_pairs
    surface=[p['depth_backprojected_sphere_surface_error'] for p in projection_rows if p['depth_backprojected_sphere_surface_error']]
    receipt['depth_surface_reconstruction']=dict(cue_camera_samples=len(surface),maximum_surface_error_m=max(x['max_m'] for x in surface),median_of_medians_m=float(np.median([x['median_m'] for x in surface])),interpretation='Depth backprojected via actual view/projection to saved world sphere geometry; tessellation residual, not a new admission threshold.')
    write(out/'receipt.json',receipt)
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PI0259 H0 independent review</title><style>body{max-width:1280px;margin:auto;background:#14191f;color:#eee;font:17px system-ui}a{color:#8cf}img{width:100%}</style>', '<h1>H0: '+receipt['status']+'</h1><p>KINEMATIC REFERENCE — NO WBC EXECUTION</p><p>144 原始 96×96 图像对；mask/depth 单独展示。几何参考不代表模型执行。</p>', '<p><a href="receipt.json">独立收据与原件 hash</a> · <a href="geometry_review.json">FK/阶段与历史失败</a> · <a href="raster_records.json">全部像素计数</a> · <a href="projection_depth.json">投影/深度诊断</a></p>']
    for r in records:parts.append('<details><summary>'+html.escape(r['source'])+'</summary><img loading="lazy" src="'+r['card']+'"></details>')
    (out/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    return receipt
