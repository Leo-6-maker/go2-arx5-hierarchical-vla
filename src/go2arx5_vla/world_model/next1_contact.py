"""Finite graded contact references; no WBC, safety, physics or state edits."""
import json
from pathlib import Path
import numpy as np
from .storage import digest

GRADES={
    'FREE_X10':dict(task_atoms=96,ramp_atoms=32,fraction=None),
    'TOWARD_ABOVE_25':dict(task_atoms=160,ramp_atoms=96,fraction=.25),
    'TOWARD_ABOVE_50':dict(task_atoms=160,ramp_atoms=96,fraction=.5),
    'DESCEND_20_OPEN':dict(task_atoms=160,ramp_atoms=116,fraction=1.),
    'ABOVE_OBJECT':dict(task_atoms=160,ramp_atoms=108,fraction=1.),
}


def validate(spec,e):
    grade=e.get('contact_grade')
    expected=dict(FREE_X10=None,TOWARD_ABOVE_25='FREE_X10',TOWARD_ABOVE_50='TOWARD_ABOVE_25',ABOVE_OBJECT='TOWARD_ABOVE_50',DESCEND_20_OPEN='ABOVE_OBJECT')
    if grade not in GRADES or e.get('predecessor_contact_grade')!=expected.get(grade)or e['task_atoms']!=GRADES[grade]['task_atoms'] or any(e['offset_xyz_m']) or spec.get('parallel_batch_spec') or e['scene_id']!='P01':
        raise PermissionError('SINGLE_SLOT_PREREGISTERED_CONTACT_GRADE_REQUIRED')
    p=Path(spec['contact_static']);v=json.loads(p.read_text())
    if digest(p)!=spec['contact_static_sha256'] or v.get('all_static_waypoints_feasible')is not True or v.get('grasp_height_offset_m')!=.02 or v.get('new_physical_integrations')!=0:
        raise PermissionError('ACTUAL_CURRENT_STEADY_STATIC_PATH_REQUIRED')
    p=Path(spec['contact_reference_check']);q=json.loads(p.read_text())
    if digest(p)!=spec['contact_reference_check_sha256']or q.get('grade')!=grade or not q.get('all_static_chunks_feasible')or q.get('new_physical_integrations')!=0:
        raise PermissionError('ORIGINAL_STATIC_REFERENCE_CERTIFICATES_REQUIRED')
    if grade=='DESCEND_20_OPEN'and e['reference_rule']!='FIXED_HOLD32_BURN8_CONTACT40_V2':raise PermissionError('DESCENT_REQUIRES_FRESH_CONTACT40_QUALIFICATION')
    if e.get('predecessor_contact_grade'):
        p=Path(spec['previous_stage']);previous=json.loads(p.read_text());proof=p.parent/'CONTACT_GRADE_RESULT.private.json'
        if not proof.exists()or previous.get('status')!='NATIVE_STAGE_PASSED' or previous.get('physics_stops') or not previous.get('shutdown_clean'):
            raise PermissionError('ACTUAL_PREVIOUS_CONTACT_GRADE_PASS_REQUIRED')
        q=json.loads(proof.read_text())
        if q.get('grade')!=e['predecessor_contact_grade']or not q.get('completed'):
            raise PermissionError('ACTUAL_MATCHING_CONTACT_PREDECESSOR_REQUIRED')
    return GRADES[grade]


def planned_path(initial,object_pose,grade):
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,quat_to_matrix,quat_slerp
    cfg=GRADES[grade];start=np.asarray(initial,dtype=float);obj=np.asarray(object_pose,dtype=float)
    if start.shape!=(4,4) or obj.shape!=(4,4) or not np.isfinite([start,obj]).all():raise ValueError('FINITE_ACTUAL_POSES_REQUIRED')
    end=start.copy()
    if cfg['fraction']is None:end[0,3]+=.01
    else:
        f=cfg['fraction'];above=obj[:3,3]+np.array([0.,0.,.02 if grade=='DESCEND_20_OPEN'else .1]);end[:3,3]+=(above-start[:3,3])*f
        end[:3,:3]=quat_to_matrix(quat_slerp(matrix_to_quat(start[:3,:3]),matrix_to_quat(np.eye(3)),f))
    n=cfg['ramp_atoms'];t=np.arange(n+1,dtype=float)
    if grade in('ABOVE_OBJECT','DESCEND_20_OPEN'):
        # Fixed eight-atom acceleration/deceleration; no adaptive rate search.
        b=8.;area=n-b
        progress=np.where(t<b,t*t/(2*b),np.where(t<=n-b,t-b/2,area-(n-t)**2/(2*b)))/area
    else:
        u=t/n;progress=10*u**3-15*u**4+6*u**5
    path=np.repeat(start[None],n+1,axis=0);path[:,:3,3]=start[:3,3]+progress[:,None]*(end[:3,3]-start[:3,3])
    q0=matrix_to_quat(start[:3,:3]);q1=matrix_to_quat(end[:3,:3])
    for i,w in enumerate(progress):path[i,:3,:3]=quat_to_matrix(quat_slerp(q0,q1,float(w)))
    return path


def chunk(path,offset):
    if type(offset)is not int or offset<0:raise ValueError('NONNEGATIVE_GLOBAL_REFERENCE_OFFSET_REQUIRED')
    return path[np.minimum(offset+np.arange(40),len(path)-1)].copy()
