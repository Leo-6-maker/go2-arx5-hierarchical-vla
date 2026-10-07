"""Freeze one target scene from original task and delivered robot geometry, before A3 scores."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from bind_pristine import ROOT,digest

def main():
    path=ROOT/'design/WBC_TASK_SCENE.json';assert not path.exists(),'Fixed scene cannot be changed after outcomes'
    source=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0276/episodes')
    rows=[];positions=[]
    for seed in range(10000,10004):
        folder=source/('TRAIN-%d-eef'%seed)/'attempt-000'
        with np.load(folder/'physics.npz') as data:
            initial={k:data[k][0].tolist() for k in ('tcp','qpos','cubeA_pose','cubeB_pose')}
            positions.extend([data['cubeA_pose'][0,:3],data['cubeB_pose'][0,:3]])
        with np.load(folder/'commands.npz') as data:assert data['action'].shape==(640,10)
        rows.append(dict(condition=seed,source_path=str(folder),initial=initial,source_files={n:digest(folder/n) for n in ('commands.npz','physics.npz','RESULT.json')}))
    with np.load(ROOT/'wbc/hold/reference.npz') as data:
        measured_reset=data['position'][0];anchor=data['anchor']
    # Source table is horizontal. Preserve world-up and put its task origin at the native shoulder anchor.
    transform=np.eye(4);transform[:3,3]=[anchor[0],anchor[1],measured_reset[2]-np.asarray(rows[0]['initial']['tcp'])[2,3]]
    lo=np.min(positions,axis=0)[:2]-.08;hi=np.max(positions,axis=0)[:2]+.08
    support_center=np.r_[(lo+hi)/2,-.02]+transform[:3,3];support_size=np.r_[hi-lo,.04]
    # Panda fingers close along local Y, approach along +Z. ARX5 fingers close along Y and approach along +X.
    source_from_target_tool=Rotation.from_euler('y',-np.pi/2).as_matrix()
    card=dict(status='FROZEN_FROM_GEOMETRY_BEFORE_A3_QUALITY',conditions=rows,
        task_to_native_world=transform.tolist(),source_TCP_from_target_TCP_rotation=source_from_target_tool.tolist(),
        tool_mapping_basis='source Panda +Z approach/+Y closing -> target ARX5 +X approach/+Y closing; both origins at finger center',
        support_center_world_m=support_center.tolist(),support_size_m=support_size.tolist(),support_fixed=True,
        support_rule='union of original four cube center XY extents plus fixed 8cm physical tool clearance; 4cm support thickness, original task top at z=0',
        cube_half_size_m=[.02,.02,.02],cube_geometry='unchanged original StackCube 4cm cube',
        robot_reset_seed=4242,robot_reset='delivered fixture configuration; no teleport or pose search',
        reference_rule='original saved EEF trajectories only; no initial approach or policy correction added; original gate can reject the initial orientation',
        no_result_based_scene_changes=True,pretrained_A3_scores_read_by_scene_designer=False,
        prerequisite='native WBC fixtures must retain original termination and be safe; failure blocks dependent task replay',selected='NONE',deployment='NONE')
    path.write_text(json.dumps(card,indent=2)+'\n');print(json.dumps(dict(status=card['status'],path=str(path),sha256=digest(path),transform=transform.tolist(),support_size=support_size.tolist())))

if __name__=='__main__':main()
