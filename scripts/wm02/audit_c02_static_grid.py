"""Read-only TRAIN geometry design; fixed order, no model/physics outcomes."""
import argparse
import json
from pathlib import Path
import time


def audit(native_spec,gpu_spec,scope,scope_sha,grid_path,grid_sha):
    import numpy as np
    import mujoco
    from scripts.wm02.audit_c01_static_state_reachability import audit as reach,pinned
    from scripts.wm02.audit_fusion_source_phase_curriculum import placements,scan
    from go2arx5_vla.world_model.native_replay import packets
    start,cpu=time.monotonic(),time.process_time()
    grid=json.loads(pinned(grid_path,grid_sha))
    if (grid['schema']!='WM02_C02_STATIC_TRAIN_DESIGN_GRID_V1'or grid['max_new_integrations']!=0
            or grid['max_GR00T_queries']!=0 or grid['max_optimizer_updates']!=0):raise ValueError('STATIC_ONLY_FROZEN_DESIGN')
    found=reach(native_spec,gpu_spec,scope,scope_sha,source_targets=grid['source_targets'])
    n=json.loads(Path(native_spec).read_bytes());journal=Path(n['registered_root'])/'native-01/state.bin.journal'
    pinned(journal,'b79eece43a7fa97587698e7d93e3e0e7255738f990a7ae34fa2f131663ec268b')
    saved=packets(journal)
    if [r['step']for r in saved]!=list(range(1,34304)):raise ValueError('ALL_SAVED_POSES')
    base=mujoco.MjModel.from_binary_path(n['model_path']);table=mujoco.mj_name2id(base,mujoco.mjtObj.mjOBJ_GEOM,'task_table');cube=mujoco.mj_name2id(base,mujoco.mjtObj.mjOBJ_GEOM,'task_cube_geom');adr=int(base.jnt_qposadr[base.body_jntadr[base.geom_bodyid[cube]]])
    arm=[int(base.jnt_qposadr[mujoco.mj_name2id(base,mujoco.mjtObj.mjOBJ_JOINT,'x5_joint'+str(i))])for i in range(1,7)]
    frame=np.asarray(n['paused_source_from_target']);initial=saved[-1]['qpos'];rows=[];selected=None
    for index,candidate in enumerate(found['targets']):
        row=dict(design_index=index,source_xyz=grid['source_targets'][index],reachability=candidate,original_prefix_geometry=None,hypothetical_joint_path_geometry=None)
        rows.append(row)
        if not candidate['static_pose_match']or candidate['outside_training_minmax_components']:
            row['status']='REJECTED_KINEMATICS_OR_SOURCE_STATE_RANGE';continue
        target=np.linalg.inv(frame)@np.asarray(candidate['achieved_source_pose'])
        placement=placements(target,frame,base.geom_pos[table],initial[adr:adr+3],base.geom_size[table])[1]
        placement['table_xyz'][2]=placement['cube_xyz'][2]-.02-base.geom_size[table,2]
        row['placement']={k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in placement.items()}
        model=mujoco.MjModel.from_binary_path(n['model_path']);frozen=[x.copy()for x in(model.body_mass,model.body_inertia,model.actuator_gainprm,model.actuator_biasprm,model.dof_damping,model.jnt_range)]
        qtarget=np.asarray(candidate['arm_q']);trials=[]
        for i,u in enumerate(np.linspace(0.,1.,401)):
            w=u**3*(10+u*(-15+6*u));q=initial.copy();q[arm]=initial[arm]+w*(qtarget-initial[arm]);trials.append((i,q))
        path=scan(model,trials,table_id=table,cube_id=cube,cube_qpos_adr=adr,placement=placement);row['hypothetical_joint_path_geometry']=path
        if not path['contact_free_at_saved_poses']:
            row['status']='REJECTED_HYPOTHETICAL_PATH_GEOMETRY';continue
        prefix=scan(model,[(r['step'],r['qpos'])for r in saved],table_id=table,cube_id=cube,cube_qpos_adr=adr,placement=placement);row['original_prefix_geometry']=prefix
        if any(not np.array_equal(a,b)for a,b in zip(frozen,(model.body_mass,model.body_inertia,model.actuator_gainprm,model.actuator_biasprm,model.dof_damping,model.jnt_range))):raise ValueError('FROZEN_ROBOT_CHANGED')
        if not prefix['contact_free_at_saved_poses']:
            row['status']='REJECTED_OLD_SAVED_PREFIX_GEOMETRY';continue
        row['status']='STATIC_GEOMETRY_ONLY_PASS_NEW_PATH_AND_NATIVE_ADMISSION_REQUIRED';selected=index;break
    return dict(schema='WM02_C02_STATIC_DESIGN_GRID_ACTUAL_V1',design_grid_sha256=grid_sha,design_targets=len(grid['source_targets']),
        actual_reachability_targets=len(found['targets']),actual_geometry_candidates=len(rows),selected_design_index=selected,
        selected_is_a_STATIC_ONLY_candidate=selected is not None,rows=rows,reachability_cost=dict(wall_s=found['wall_s'],CPU_s=found['CPU_s'],FK_calls=found['static_FK_calls']),
        new_integrations=0,new_queries=0,new_updates=0,new_training_labels=0,files_written=0,new_native_admission=False,
        original_C01_safety_failure_not_repaired=True,not_a_WBC_reference_or_dynamic_path_certificate=True,
        selection_uses_only_static_geometry_not_model_or_task_outcomes=True,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-spec',type=Path,required=True);p.add_argument('--gpu-spec',type=Path,required=True);p.add_argument('--scope',type=Path,required=True);p.add_argument('--scope-sha256',required=True);p.add_argument('--grid',type=Path,required=True);p.add_argument('--grid-sha256',required=True);a=p.parse_args()
    print(json.dumps(audit(a.native_spec,a.gpu_spec,a.scope,a.scope_sha256,a.grid,a.grid_sha256)));return 0
if __name__=='__main__':raise SystemExit(main())
