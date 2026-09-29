"""Finite PI0259 rig calibration, preserving raw policy and diagnostic rasters."""
import isaacgym
from pathlib import Path
import copy
import json
import numpy as np
from pi0259_renderer import Renderer, ROOT
from pi0255_motion3 import sha,write,scene
from go2arx5_vla.trajectory.so3 import matrix_to_quat


def visibility(frames,ids,task,command):
    counts={n:{c:int(np.sum(frames[c+'_segmentation']==sid)) for c in ('body','wrist')} for n,sid in ids.items()}
    def count(n):return max(counts[n].values())
    if task=='M1':required={'target'+str(command):4,'axis%d_0'%command:2,'axis%d_1'%command:2}
    elif task=='M2':required={'obstacle':16,'goal':4}
    else:required={'black_A':4,'white_B':4}
    failures=[n for n,threshold in required.items() if count(n)<threshold]
    centers={};color_checks={}
    if task=='M1':
        for a in (0,1):
            n='axis%d_%d'%(command,a);c=max(counts[n],key=counts[n].get);mask=frames[c+'_segmentation']==ids[n]
            pixels=frames[c][mask];yx=np.argwhere(mask)
            centers[n]=dict(camera=c,xy=np.mean(yx[:,::-1],axis=0).tolist() if len(yx) else None)
            mean=np.mean(pixels,axis=0) if len(pixels) else np.zeros(3)
            color_checks[n]=bool(len(pixels) and mean[a]>max(np.delete(mean,a)))
            if not color_checks[n]:failures.append(n+'_COLOR')
        first,second=[centers['axis%d_%d'%(command,a)] for a in (0,1)]
        if first['camera']==second['camera'] and first['xy'] is not None and second['xy'] is not None and np.linalg.norm(np.array(first['xy'])-second['xy'])<=0:
            failures.append('AXIS_CENTROIDS_IDENTICAL')
    sectors=[any(count('contour'+str(i))>0 for i in (2*s,2*s+1)) for s in range(12)]
    if task=='M3' and sum(sectors)<8:failures.append('CONTOUR_LT8_SECTORS')
    return dict(pass_necessary=not failures,failures=failures,required=required,counts=counts,
        axis_centroids=centers,axis_color_checks=color_checks,visible_contour_sectors=sectors)


def save_packet(folder,name,renderer,q,reg,metadata):
    folder.mkdir(parents=True,exist_ok=True);assert not (folder/(name+'.json')).exists()
    frames,info=renderer.place(q);pose=np.array(info['FK_TCP_world_matrix']);anchor=np.array(reg['initial_TCP_world'])[:3,3]
    state=np.r_[pose[:3,3]-anchor,pose[:2,:3].reshape(-1),.5]
    transformed=np.eye(4);transformed[:3,3]=anchor
    arrays={('body_rgb' if k=='body' else 'wrist_rgb' if k=='wrist' else k):v for k,v in frames.items()}
    arrays.update(arm_joints=np.array(q),FK_world=pose,state_v1=state,T_world_native=transformed,
        root_transform_world=np.array(info['root_transform_world']))
    np.savez_compressed(folder/(name+'.npz'),**arrays)
    info.update(metadata);info.update(array_file=name+'.npz',array_sha256=sha(folder/(name+'.npz')),
        state_origin='KINEMATIC_FK_NOT_DYNAMIC_MEASUREMENT',evidence_class='KINEMATIC_RENDERED_REFERENCE')
    info['visibility']={task:{str(c):visibility(frames,renderer.scene_ids,task,c) for c in (0,1)} for task in ('M1','M2','M3')}
    write(folder/(name+'.json'),info)
    return info,frames


def run():
    reg=json.loads((ROOT/'design/registry.json').read_text());results=[];pairs=0;selected=None
    assert json.loads((ROOT/'evidence/cpu_fixtures.json').read_text())['status']=='PASS'
    assert (ROOT/'evidence/calibration_geometry.json').exists()
    output=ROOT/'calibration_rigs';assert not output.exists(),'first-valid calibration already started'
    output.mkdir();write(output/'started.json',dict(source_sha256=sha(__file__),renderer_sha256=sha(ROOT/'runtime/pi0259_renderer.py'),registry_sha256=sha(ROOT/'design/registry.json')))
    for rig in range(3):
        configured=copy.deepcopy(reg);configured['renderer']['selected_rig']=rig;rigroot=output/('rig%d'%rig)
        starts=[];candidate_tasks=[]
        for family in reg['calibration_families']:
            renderer=Renderer(configured,family)
            try:
                folder=rigroot/family['family_id']
                info,frames=save_packet(folder,'common_start',renderer,np.array(reg['arm_default']),configured,
                    dict(rig_index=rig,family_id=family['family_id'],synthetic_reference_time=0.,role='COMMON_START_ALL_TASKS_COMMANDS'))
                repeated,again=save_packet(folder,'common_start_repeat',renderer,np.array(reg['arm_default']),configured,
                    dict(rig_index=rig,family_id=family['family_id'],synthetic_reference_time=0.,role='PAIR_INPUT_REPRODUCTION'))
                pairs+=2;assert all(np.array_equal(frames[k],again[k]) for k in frames), 'same root/DOF render differs'
                starts.append(info)
            finally:renderer.close()
        for task in ('M1','M2','M3'):
            if all(x['visibility'][task][str(c)]['pass_necessary'] for x in starts for c in (0,1)):candidate_tasks.append(task)
        outcome=dict(rig_index=rig,start_supported_tasks=candidate_tasks,calibration_frames=[],missing_geometry=[],supported_tasks=[])
        if len(candidate_tasks)>=2:
            # One scene per family; both commands use its fixed matrices, never target tracking.
            for family in reg['calibration_families']:
                renderer=Renderer(configured,family)
                try:
                    for seq in [s for s in reg['sequences'] if s['family_id']==family['family_id']]:
                        case=seq['sequence_id'];geometry=ROOT/'calibration'/case/'geometry.npz'
                        with np.load(geometry) as a:data={k:a[k].copy() for k in a.files}
                        for t in reg['renderer']['calibration_times'][seq['task']][1:]:
                            index=int(round(t/.05));name=case+'-t%04d'%index
                            if not data['sample_qualified'][index]:
                                outcome['missing_geometry'].append(dict(sequence=case,task=seq['task'],time=t,reason='NO_ACCEPTED_IK_PREFIX_AT_REGISTERED_POSE'));continue
                            info,_=save_packet(rigroot/family['family_id'],name,renderer,data['joints'][index],configured,
                                dict(rig_index=rig,family_id=family['family_id'],sequence=case,task=seq['task'],command=seq['command'],synthetic_reference_time=t,
                                    geometry_sha256=sha(geometry),geometry_index=index,requested_position=data['desired_position'][index].tolist(),requested_rotation=data['desired_rotation'][index].tolist()))
                            pairs+=1;assert pairs<=512
                            outcome['calibration_frames'].append(dict(sequence=case,task=seq['task'],time=t,pass_necessary=info['visibility'][seq['task']][str(seq['command'])]['pass_necessary']))
                finally:renderer.close()
            for task in candidate_tasks:
                frames=[x for x in outcome['calibration_frames'] if x['task']==task]
                if not any(x['task']==task for x in outcome['missing_geometry']) and len(frames)==48 and all(x['pass_necessary'] for x in frames):outcome['supported_tasks'].append(task)
        outcome['pairs_so_far']=pairs;results.append(outcome);write(rigroot/'outcome.json',outcome)
        print(json.dumps(dict(rig_index=rig,start_supported_tasks=candidate_tasks,supported_tasks=outcome['supported_tasks'],pairs=pairs)),flush=True)
        if len(outcome['supported_tasks'])>=2:selected=rig;break
    write(ROOT/'evidence/camera_calibration.json',dict(schema='PI0259_CAMERA_CALIBRATION_V1',rigs=results,selected_rig=selected,image_pairs=pairs,
        status='H0_REVIEW_PENDING' if selected is not None else 'OBSERVATION_DESIGN_BLOCKED',physics_steps=0,models=0,
        raster_conventions=dict(rgb='RGB uint8;96x96;row-down,column-right',segmentation='native integer actor IDs',depth='IsaacGym camera-axis negative metres; -inf background',
            projection='row homogeneous world @ view @ projection; NDC x right,y up; pixel x=(ndcx+1)*48,y=(1-ndcy)*48; raster pixel centers x+.5,y+.5'),
        missing=['No processor or codec gate before fresh TRAIN data; raw gate on complete calibration references exported separately.']))


if __name__=='__main__':run()
