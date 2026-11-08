"""Two hypothetical C02 layouts at an in-range FK pose; no dynamics or labels.

Reuse the original direct geometry scanner. The actual saved C01 prefix and a
separate hypothetical arm-joint interpolation are assessed separately. Neither
is an executed C02 trajectory or a Cartesian WBC/path certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def audit(native_spec,reachability,reachability_sha,scope,scope_sha):
    import numpy as np
    import mujoco
    from scripts.wm02.audit_c01_static_state_reachability import pinned
    from scripts.wm02.audit_fusion_source_phase_curriculum import placements,scan
    from go2arx5_vla.world_model.native_replay import packets
    from go2arx5_vla.world_model.fusion_scope import binding_contract
    start,cpu=time.monotonic(),time.process_time()
    authority=json.loads(pinned(scope,scope_sha));binding_contract(authority,time.time())
    if json.loads((Path(scope).parent/'LEASE.json').read_bytes())['status']!='ACTIVE':raise PermissionError('FINITE_ORIGINAL_SCOPE')
    n=json.loads(pinned(native_spec,'5369da7902f730138c96718920a3ac3392f8cc55524b64df71afaf66d1fc092a'))
    result=json.loads(pinned(reachability,reachability_sha))
    if result['prohibited_integration_calls']or result['native_integrations']or result['robot_dynamics_or_scene_changed']:raise ValueError('READONLY_REACHABILITY_PROOF')
    chosen=[r for r in result['targets']if r['requested_source_pose'][2][3]==.54]
    if len(chosen)!=1 or not chosen[0]['static_pose_match']or chosen[0]['outside_training_minmax_components']:raise ValueError('FIXED_054M_IN_RANGE_TARGET_NOT_POSTHOC_CHOICE')
    chosen=chosen[0];model_path=n['model_path'];pinned(model_path,'24fc70f0b9911312f17a083cd2f254252e0003583aac3520a4e3ce642253b45f')
    journal=Path(n['registered_root'])/'native-01/state.bin.journal'
    pinned(journal,'b79eece43a7fa97587698e7d93e3e0e7255738f990a7ae34fa2f131663ec268b')
    rows=packets(journal)
    if [r['step']for r in rows]!=list(range(1,34304)):raise ValueError('EXACT_ORIGINAL_PREFIX')
    model=mujoco.MjModel.from_binary_path(model_path)
    table=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'task_table');cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'task_cube_geom')
    adr=int(model.jnt_qposadr[model.body_jntadr[model.geom_bodyid[cube]]])
    frame=np.asarray(n['paused_source_from_target']);target=np.linalg.inv(frame)@np.asarray(chosen['achieved_source_pose'])
    candidates=placements(target,frame,model.geom_pos[table],rows[-1]['qpos'][adr:adr+3],model.geom_size[table])
    arm=[int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'x5_joint'+str(i))])for i in range(1,7)]
    qa=rows[-1]['qpos'][arm];qb=np.asarray(chosen['arm_q']);trials=[]
    for index,u in enumerate(np.linspace(0.,1.,401)):
        weight=u**3*(10+u*(-15+6*u));q=rows[-1]['qpos'].copy();q[arm]=qa+weight*(qb-qa);trials.append((index,q))
    out=[]
    for candidate in candidates:
        model=mujoco.MjModel.from_binary_path(model_path)
        frozen=[x.copy()for x in (model.body_mass,model.body_inertia,model.actuator_gainprm,model.actuator_biasprm,model.dof_damping,model.jnt_range)]
        prefix=scan(model,[(r['step'],r['qpos'])for r in rows],table_id=table,cube_id=cube,cube_qpos_adr=adr,placement=candidate)
        future=scan(model,trials,table_id=table,cube_id=cube,cube_qpos_adr=adr,placement=candidate)
        if any(not np.array_equal(a,b)for a,b in zip(frozen,(model.body_mass,model.body_inertia,model.actuator_gainprm,model.actuator_biasprm,model.dof_damping,model.jnt_range))):raise ValueError('ROBOT_FROZEN_ARRAYS_CHANGED')
        out.append(dict(candidate={k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in candidate.items()},
            geometry_at_original_saved_C01_prefix=prefix,hypothetical_401_pose_joint_interpolation_geometry=future,
            static_geometry_pass=prefix['contact_free_at_saved_poses']and future['contact_free_at_saved_poses'],native_qualification=False))
    return dict(schema='WM02_C02_SOURCE_STATE_COURSE_STATIC_ACTUAL_V1',rows=out,chosen_source_z_m=.54,
        achieved_source_pose=chosen['achieved_source_pose'],target_world_TCP=target.tolist(),
        original_scene_sha256='24fc70f0b9911312f17a083cd2f254252e0003583aac3520a4e3ce642253b45f',
        original_journal_sha256='b79eece43a7fa97587698e7d93e3e0e7255738f990a7ae34fa2f131663ec268b',
        reachability_sha256=reachability_sha,static_only=True,new_native_integrations=0,new_model_queries=0,new_model_updates=0,
        controller_or_safety_changes=False,new_training_labels=0,old_C01_prefix_failure_not_fixed_by_this_geometry_audit=True,
        hypothetical_path_is_not_actual_trajectory_or_WBC_reference=True,new_scene_dynamic_admission=False,
        files_written=0,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-spec',type=Path,required=True);p.add_argument('--reachability',type=Path,required=True);p.add_argument('--reachability-sha256',required=True);p.add_argument('--scope',type=Path,required=True);p.add_argument('--scope-sha256',required=True);a=p.parse_args()
    print(json.dumps(audit(a.native_spec,a.reachability,a.reachability_sha256,a.scope,a.scope_sha256)));return 0
if __name__=='__main__':raise SystemExit(main())
