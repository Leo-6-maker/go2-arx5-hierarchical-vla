"""Saved-state FK/geometry diagnosis only; no dynamics, IK or admission call."""
import argparse
import itertools
import json
from pathlib import Path
import struct
import time

import numpy as np
import mujoco


def read(path):
    return json.loads(Path(path).read_bytes())


def analyze(path):
    from scripts.pi0294.task_state import parse_state
    from scripts.pi0292.eval_contract import matrix10
    from go2arx5_vla.trajectory.so3 import geodesic_angle
    spec=read(path);root=Path(path).parent;out=root/'native-01';result=read(root/'RESULT.private.json')
    assert read(root/'control/LEASE.json')['status']=='CLOSED'
    rows=[json.loads(l)for l in(out/'MOBILE_SAMPLES.private.jsonl').read_text().splitlines()]
    events=[json.loads(l)for l in(out/'COMMANDS.private.jsonl').read_text().splitlines()]
    events=[e for e in events if e['phase']=='mobile-continuous-expert-query']
    initial=next(r for r in rows if r['native_end_step']==5703)
    initial_TCP=np.asarray(initial['actual_TCP']);obj=np.asarray(initial['object_pose'])
    goal=np.eye(4);goal[:3,:3]=initial_TCP[:3,:3]
    goal[:3,3]=obj[:3,3]+[0,-.0017,.012]+goal[:3,:3]@[-.012,0,0]
    m=mujoco.MjModel.from_binary_path(spec['model_path']);d=mujoco.MjData(m)
    table=m.geom('task_table').id;pads=[m.geom('task_'+s+'_pad').id for s in('left','right')]
    tcp=m.site('x5_ee').id;cube=m.body('task_cube').id;mount_id=m.body('x5_base_link').id
    arm_joints=[j for j in range(m.njnt)if 19<=m.jnt_qposadr[j]<25]
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=3)))
    task=[r for r in rows if r['segment']=='MOBILE_TASK']
    first_close=next((e['native_step']for e in events if e['stage']=='CLOSE'),None)
    # Last 1s before CLOSE, or last 1s before the native STOP.
    boundary=first_close or result['actual_native_integrations']
    selected=[r for r in task if boundary-500<=r['native_end_step']<=boundary]
    records=[]
    def clearance(corners):
        local=(corners-d.geom_xpos[table])@d.geom_xmat[table].reshape(3,3)
        lo,hi=local.min(axis=0),local.max(axis=0)
        overlap=bool(np.all(lo[:2]<=m.geom_size[table,:2])and np.all(hi[:2]>=-m.geom_size[table,:2]))
        return dict(vertical_clearance_m=float(lo[2]-m.geom_size[table,2]),table_XY_overlap=overlap)
    def frame(row,label):
        d.qpos[:]=row['qpos'];d.qvel[:]=row['qvel']
        mujoco.mj_kinematics(m,d)
        actual=np.eye(4);actual[:3,3]=d.site_xpos[tcp];actual[:3,:3]=d.site_xmat[tcp].reshape(3,3)
        mount=np.eye(4);mount[:3,3]=d.xpos[mount_id];mount[:3,:3]=d.xmat[mount_id].reshape(3,3)
        if 'actual_TCP'in row:
            assert np.max(np.abs(actual-np.asarray(row['actual_TCP'])))<1e-10
        reference=matrix10(np.asarray(row['target_endpoint_EEF10']))if 'target_endpoint_EEF10'in row else None
        event=next(e for e in reversed(events)if e['native_step']<row['native_end_step'])
        cursor=(row['native_end_step']-event['native_step'])//25
        nominal=matrix10(np.asarray(event['submitted_policy_H40'])[min(cursor,39)])
        records.append(dict(label=label,native_step=row['native_end_step'],query=event['query'],stage=event['stage'],
            actual_goal_error_xyz_m=(actual[:3,3]-goal[:3,3]).tolist(),
            actual_goal_rotation_error_rad=float(geodesic_angle(actual[:3,:3],goal[:3,:3])),
            actual_vs_installed_rotation_error_rad=float(geodesic_angle(actual[:3,:3],reference[:3,:3]))if reference is not None else None,
            nominal_goal_error_m=float(np.linalg.norm(nominal[:3,3]-goal[:3,3])),
            installed_goal_error_m=float(np.linalg.norm(reference[:3,3]-goal[:3,3]))if reference is not None else None,
            actual_gripper_fraction=float(np.mean(d.qpos[[m.jnt_qposadr[m.joint('task_'+s+'_slide').id]for s in('left','right')]])/.044),pads=[]))
        records[-1].update(actual_TCP=actual.tolist(),actual_mount=mount.tolist(),
            arm_qpos={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j):float(d.qpos[m.jnt_qposadr[j]])for j in arm_joints},
            arm_qvel={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j):float(d.qvel[m.jnt_dofadr[j]])for j in arm_joints},
            native_root_qvel6=d.qvel[:6].tolist())
        for pad in pads:
            corners=d.geom_xpos[pad]+(signs*m.geom_size[pad])@d.geom_xmat[pad].reshape(3,3).T
            local=(corners-actual[:3,3])@actual[:3,:3]
            rotation_only=local@goal[:3,:3].T+actual[:3,3]
            position_only=local@actual[:3,:3].T+goal[:3,3]
            nominal_goal=local@goal[:3,:3].T+goal[:3,3]
            desired_current=local@nominal[:3,:3].T+nominal[:3,3]
            diagnostic=dict(pad=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,pad),
                actual=clearance(corners),nominal_goal=clearance(nominal_goal),
                actual_position_goal_rotation=clearance(rotation_only),
                goal_position_actual_rotation=clearance(position_only),
                nominal_consumed=clearance(desired_current),
                direct_actual_signed_geom_distance_m=float(mujoco.mj_geomDistance(m,d,pad,table,.1,None)))
            raised=nominal_goal+[0,0,.008]
            relative=(raised-obj[:3,3])@obj[:3,:3]
            half=m.geom_size[m.geom('task_cube_geom').id]
            diagnostic['raised_goal_8mm']=clearance(raised)
            diagnostic['raised_goal_object_vertical_overlap_m']=float(max(0.,min(relative[:,2].max(),half[2])-max(relative[:,2].min(),-half[2])))
            if reference is not None:diagnostic['installed_reference']=clearance(local@reference[:3,:3].T+reference[:3,3])
            records[-1]['pads'].append(diagnostic)
    for row in selected:frame(row,'CONFIRMED_50MS_SAMPLE')
    if result['unsafe']:
        last=[]
        with(out/'state.bin.journal').open('rb')as f:
            while size:=f.read(4):
                blob=f.read(struct.unpack('<I',size)[0]);last=(last+[blob])[-2:]
        for raw in last:
            p=parse_state(raw);row=dict(native_end_step=p['step'],qpos=p['qpos'],qvel=p['qvel'])
            frame(row,'ORIGINAL_NATIVE_STOP_STATE_NOT_A_CONFIRMED_ACTION_WINDOW')
    return dict(condition=spec['mobile_condition'],result_sha256=__import__('hashlib').sha256((root/'RESULT.private.json').read_bytes()).hexdigest(),
        geometry='RECORDED_QPOS_FORWARD_KINEMATICS_RIGID_PAD_COUNTERFACTUAL_NOT_IK',
        pad_half_sizes_m=[m.geom_size[p].tolist()for p in pads],table_half_sizes_m=m.geom_size[table].tolist(),
        counterfactual_scope='Same measured finger configuration; substitute TCP pose algebraically. Not a realizable WBC-state or collision qualification.',
        records=records)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',required=True);a=parser.parse_args()
    start=time.monotonic();cpu=time.process_time()
    output=[analyze(p)for p in read(Path(a.group)/'EXECUTION_SPEC.json')['cases']]
    print(json.dumps(dict(schema='MOBILE_PAD_TABLE_SAVED_GEOMETRY_V3',cases=output,
        new_physical_requests=0,new_integrations=0,new_forward_calls=0,new_model_queries=0,new_certifier_calls=0,
        FK_only=True,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)))


if __name__=='__main__':main()
