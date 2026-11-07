"""Saved-pose geometry audit only; constructs no physical or GPU executor.

The candidate worlds are hypothetical new tasks. This is not a replay of their
dynamics, a controller certificate, training data, or permission to execute.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SCENE='40408bbcbef2469396187c741750905a6ba9ed7161dc031d14bfab5fed62ddd2'
SOURCE_VECTOR=np.array([5.118904796153068e-7,-4.4582175856566764e-7,-.0799996963193938])


def placements(tcp,frame,table,cube,halfsize):
    from scripts.pi0294.dec1.target_observation import rigid
    tcp=rigid(tcp);frame=rigid(frame)
    table=np.asarray(table,float);cube=np.asarray(cube,float);halfsize=np.asarray(halfsize,float)
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (table,cube,halfsize)) or np.any(halfsize<=0):
        raise ValueError('FINITE_ACTUAL_TASK_GEOMETRY')
    target=tcp[:3,3]+frame[:3,:3].T@SOURCE_VECTOR
    translated=table+target-cube
    pedestal=target.copy();pedestal[2]-=cube[2]-table[2]
    return [dict(name='TRANSLATED_ORIGINAL_TABLE',cube_xyz=target,table_xyz=translated,table_halfsize=halfsize.copy()),
            dict(name='SEPARATE_120MM_PEDESTAL',cube_xyz=target,table_xyz=pedestal,table_halfsize=np.array([.06,.06,halfsize[2]]))]


def scan(model,poses,*,table_id,cube_id,cube_qpos_adr,placement=None):
    """Forward only at actual saved poses, with the candidate cube relocated."""
    import mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration
    if mujoco.get_mjcb_control()is not None:raise ValueError('NO_CONTROL_CALLBACK_IN_STATIC_AUDIT')
    if placement is not None:
        model.geom_pos[table_id]=placement['table_xyz'];model.geom_size[table_id]=placement['table_halfsize']
    data=mujoco.MjData(model);environment={table_id,cube_id};bad=0;first=None;deepest=0.;count=0;pairs=0
    if any(model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_BOX for i in environment):raise ValueError('DECLARED_TASK_BOX_GEOMETRY')
    compatible={i:np.array([j for j in range(model.ngeom) if j not in environment and model.geom_bodyid[j]!=0
        and ((int(model.geom_contype[i])&int(model.geom_conaffinity[j])) or (int(model.geom_contype[j])&int(model.geom_conaffinity[i])))],int)for i in environment}
    with forbid_integration(mujoco)as calls:
        for step,qpos in poses:
            qpos=np.asarray(qpos,float)
            if qpos.shape!=(model.nq,)or not np.isfinite(qpos).all():raise ValueError('EXACT_FINITE_SAVED_QPOS')
            data.qpos[:]=qpos
            if placement is not None:data.qpos[cube_qpos_adr:cube_qpos_adr+3]=placement['cube_xyz']
            data.qvel[:]=0
            mujoco.mj_forward(model,data);count+=1
            hits=[]
            # Direct narrow-phase avoids relying on the original compiled static
            # world's BVH after hypothetical table placement/size changes.
            for i,others in compatible.items():
                radii=np.linalg.norm(model.geom_size[i])+model.geom_rbound[others]
                close=others[np.linalg.norm(data.geom_xpos[others]-data.geom_xpos[i],axis=1)<=radii+.01]
                for j in close:
                    distance=float(mujoco.mj_geomDistance(model,data,i,int(j),.01,None));pairs+=1
                    if not np.isfinite(distance):raise ValueError('FINITE_DIRECT_GEOMETRY_DISTANCE')
                    if distance<0:hits.append(distance)
            if hits:
                bad+=1;deepest=min(deepest,min(hits))
                if first is None:first=int(step)
        if calls:raise ValueError('STATIC_AUDIT_CANNOT_INTEGRATE')
    return dict(saved_pose_FK_calls=count,direct_narrow_phase_calls=pairs,robot_task_interpenetration_pose_count=bad,
        first_interpenetration_native_step=first,deepest_interpenetration_m=deepest,
        new_integrations=0,contact_free_at_saved_poses=bad==0,
        new_world_dynamics_or_native_qualification=False)


def audit(request):
    import mujoco
    from go2arx5_vla.world_model.native_replay import packets
    start=time.monotonic();cpu=time.process_time()
    def read(path,pin):
        path=Path(path);raw=path.read_bytes()
        if path.is_symlink()or hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('PINNED_ORIGINAL_BYTES')
        return json.loads(raw)
    root=Path(request['root']);s=read(root/'EXECUTION_SPEC.json',request['spec_sha256'])
    n=read(root/'native-01/RESULT.json',request['result_sha256']);t=read(root/'native-01/GEOMETRIC_TASK_REFERENCE_PROPOSAL.private.json',request['teacher_sha256'])
    if n['status']!='NATIVE_STAGE_PASSED'or not n['shutdown_clean']:raise ValueError('SEALED_QUALIFIED_ORIGINAL_RECORDING')
    scene=Path(s['model_path']);journal=root/'native-01/state.bin.journal'
    if hashlib.sha256(scene.read_bytes()).hexdigest()!=SCENE or hashlib.sha256(journal.read_bytes()).hexdigest()!=request['journal_sha256']:raise ValueError('EXACT_ORIGINAL_SCENE_AND_JOURNAL')
    rows=packets(journal);poses=[(p['step'],p['qpos'])for p in rows if p['step']<=34303]
    if [x[0]for x in poses]!=list(range(1,34304)):raise ValueError('ALL_ORIGINAL_2MS_PREFIX_POSES_REQUIRED')
    m=mujoco.MjModel.from_binary_path(str(scene));table=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'task_table');cube=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'task_cube_geom')
    if min(table,cube)<0:raise ValueError('EXACT_TASK_GEOMETRY_REQUIRED')
    joint=int(m.body_jntadr[int(m.geom_bodyid[cube])]);adr=int(m.jnt_qposadr[joint]);initial_cube=poses[-1][1][adr:adr+3]
    variants=placements(t['actual_decision_TCP'],s['paused_source_from_target'],m.geom_pos[table],initial_cube,m.geom_size[table]);out=[]
    for candidate in [None,*variants]:
        model=mujoco.MjModel.from_binary_path(str(scene))
        frozen=[x.copy()for x in (model.body_mass,model.body_inertia,model.actuator_gainprm,model.actuator_biasprm,model.dof_damping,model.jnt_range)]
        result=scan(model,poses,table_id=table,cube_id=cube,cube_qpos_adr=adr,placement=candidate)
        if any(not np.array_equal(a,b)for a,b in zip(frozen,(model.body_mass,model.body_inertia,model.actuator_gainprm,model.actuator_biasprm,model.dof_damping,model.jnt_range))):raise ValueError('ROBOT_DYNAMICS_OR_LIMITS_CHANGED')
        out.append(dict(name='ORIGINAL_T06'if candidate is None else candidate['name'],**result))
    return dict(schema='WM02_SOURCE_PHASE_CURRICULUM_SAVED_POSE_GEOMETRY_ACTUAL_V1',source_sha=s['source_sha'],pins=request,original_scene_sha256=SCENE,rows=out,
        source_relative_cube_vector_m=SOURCE_VECTOR.tolist(),candidate_cube_target_world_m=variants[0]['cube_xyz'].tolist(),
        original_pose_data_are_not_new_world_trajectories=True,no_training_labels_generated=True,
        old_140mm_noncontact_domain_not_reclassified=True,new_scene_dynamic_admission=False,
        files_written=0,MuJoCo_models_loaded=4,WBC_executors_started=0,neural_models_loaded=0,
        new_queries=0,new_updates=0,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);args=p.parse_args(argv)
    print(json.dumps(audit(json.loads(args.request.read_text())),default=lambda a:a.tolist(),allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
