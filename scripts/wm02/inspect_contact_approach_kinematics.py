"""Constrained static arm approach on a pinned full-body recording; never execute."""
def main(argv=None):
    import argparse,json,time
    from pathlib import Path
    import numpy as np,mujoco
    from go2arx5_vla.world_model.native_replay import load,apply_frame,forbid_integration
    from go2arx5_vla.world_model.storage import digest,write_json
    from scripts.pi0294.task_contacts import allowed
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--grasp-height-offset-m',type=float,default=0.);p.add_argument('--probe-gripper-geometry',action='store_true');a=p.parse_args(argv);start=time.monotonic();cpu=time.process_time();cfg=json.loads(a.config.read_text())
    if not np.isfinite(a.grasp_height_offset_m)or not 0<=a.grasp_height_offset_m<=.035:raise ValueError('FINITE_STATIC_GRASP_HEIGHT_RANGE_REQUIRED')
    if cfg['label']!='TARGET_QUALIFIED_RECORDING_REPLAY':raise ValueError('QUALIFIED_ORIGINAL_INITIAL_STATE_REQUIRED')
    mj,m,d,rows=load(cfg);native=json.loads(Path(cfg['native_result']['path']).read_text());step=native['reference_qualification']['actual_collection_start_step'];row=next(r for r in rows if r['step']==step)
    if cfg.get('steady_prefix'):
        proof=cfg['steady_prefix'];value=json.loads(Path(proof['path']).read_text())
        if digest(proof['path'])!=proof['sha256']or not value.get('qualified')or native['reference_qualification'].get('steady_prefix')!=value:raise ValueError('ACTUAL_SAME_RECORDING_STEADY_PREFIX_REQUIRED')
        step=value['actual_native_step'];row=next(r for r in rows if r['step']==step)
    arm=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'x5_joint'+str(i))for i in range(1,7)]
    if any(j<0 for j in arm):raise ValueError('EXACT_NATIVE_SIX_ARM_JOINTS_REQUIRED')
    qa=m.jnt_qposadr[arm];va=m.jnt_dofadr[arm];low=m.jnt_range[arm,0];high=m.jnt_range[arm,1];site=mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,'x5_ee');cube=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'task_cube');fingers=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'task_'+side+'_slide')for side in ['left','right']]
    jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));solved=[];first=None;iterations=0
    with forbid_integration(mj)as calls:
        apply_frame(mj,m,d,row);original=d.qpos.copy();origin=d.site_xpos[site].copy();initial_R=d.site_xmat[site].reshape(3,3).copy();object_xyz=d.xpos[cube].copy();q0=np.zeros(4);mj.mju_mat2Quat(q0,initial_R.reshape(-1));identity=np.array([1.,0,0,0])
        # Static candidates only: orient above the object, then descend. These
        # poses are not expert labels, actuator commands or a qualified prefix.
        targets=[]
        for phase,end,n in [('above',object_xyz+np.array([0.,0.,.10]),16),('descend',object_xyz+np.array([0.,0.,a.grasp_height_offset_m]),10)]:
            begin=origin if phase=='above'else object_xyz+np.array([0.,0.,.10])
            for i in range(1,n+1):
                w=i/n;quat=(1-w)*q0+w*identity if phase=='above'else identity.copy();quat/=np.linalg.norm(quat);R=np.zeros(9);mj.mju_quat2Mat(R,quat);targets.append((phase,begin+(end-begin)*w,R.reshape(3,3)))
        def error(target,R):
            current=d.site_xmat[site].reshape(3,3)
            angular=sum(np.cross(current[:,i],R[:,i])for i in range(3))*.5
            return np.r_[target-d.site_xpos[site],.15*angular]
        for phase,target,R in targets:
            for _ in range(180):
                iterations+=1;mj.mj_forward(m,d);e=error(target,R)
                if np.linalg.norm(e[:3])<.0005 and np.linalg.norm(e[3:])/.15<.01:break
                mj.mj_jacSite(m,d,jp,jr,site);J=np.vstack((jp[:,va],.15*jr[:,va]));delta=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-5,e);delta*=min(1.,.10/max(np.linalg.norm(delta),1e-12));old=d.qpos[qa].copy();before=np.linalg.norm(e)
                for alpha in [1.,.5,.25,.1]:
                    # Bound a proposed IK iterate; measured state is untouched.
                    d.qpos[qa]=np.minimum(high,np.maximum(low,old+alpha*delta));mj.mj_forward(m,d)
                    if np.linalg.norm(error(target,R))<before:break
                else:d.qpos[qa]=old;break
            mj.mj_forward(m,d);e=error(target,R);collisions=[]
            for c in d.contact:
                names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)for g in(c.geom1,c.geom2)]
                if not allowed(*names)or(frozenset(names)in {frozenset(['task_left_pad','task_cube_geom']),frozenset(['task_right_pad','task_cube_geom'])}and c.dist<-.004):collisions.append(dict(geoms=names,bodies=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))for g in(c.geom1,c.geom2)],distance_m=float(c.dist)))
            record=dict(phase=phase,target_TCP_xyz_m=target.tolist(),actual_static_TCP_xyz_m=d.site_xpos[site].tolist(),position_error_m=float(np.linalg.norm(e[:3])),orientation_error_proxy_rad=float(np.linalg.norm(e[3:])/.15),arm_q_rad=d.qpos[qa].tolist(),unallowed_static_contacts=collisions);solved.append(record)
            if record['position_error_m']>.001 or record['orientation_error_proxy_rad']>.02 or collisions:first='STATIC_IK_OR_ORIGINAL_CONTACT_CONSTRAINT';break
        closing=[]
        if first is None and a.probe_gripper_geometry:
            fqa=m.jnt_qposadr[fingers];initial_fingers=d.qpos[fqa].copy();closed=np.full(2,.0175)
            # Geometric proposals reserve the original 0.04m/s,50ms finger
            # interval. No actuator or force response is simulated here.
            n=max(1,int(np.ceil(np.max(np.abs(initial_fingers-closed))/.002)))
            for k in range(1,n+1):
                d.qpos[fqa]=initial_fingers+(closed-initial_fingers)*k/n;mj.mj_forward(m,d);contacts=[];bad=[]
                for c in d.contact:
                    names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)for g in(c.geom1,c.geom2)];item=dict(geoms=names,bodies=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))for g in(c.geom1,c.geom2)],distance_m=float(c.dist))
                    if 'task_cube_geom'in names and any(name in names for name in ['task_left_pad','task_right_pad']):contacts.append(item)
                    if not allowed(*names)or(contacts and contacts[-1]is item and c.dist<-.004):bad.append(item)
                closing.append(dict(finger_q_m=d.qpos[fqa].tolist(),pad_object_geometric_contacts=contacts,unallowed_contacts=bad))
                if bad:first='STATIC_GRIPPER_ORIGINAL_CONTACT_CONSTRAINT';break
            d.qpos[fqa]=initial_fingers;mj.mj_forward(m,d)
        untouched=np.ones(m.nq,dtype=bool);untouched[qa]=False
        if not np.array_equal(d.qpos[untouched],original[untouched]):raise RuntimeError('STATIC_SOLVER_CHANGED_BASE_LEGS_OBJECT_OR_FINGERS')
        if calls:raise RuntimeError('STATIC_APPROACH_CANNOT_INTEGRATE')
    out=dict(status='PINNED_NATIVE_STATIC_APPROACH_INSPECTED',episode_id=cfg['episode_id'],input_sha256={k:cfg[k]['sha256']for k in ['scene','journal','native_result']},initial_native_step=step,initial_TCP_xyz_m=origin.tolist(),object_xyz_m=object_xyz.tolist(),grasp_height_offset_m=a.grasp_height_offset_m,requested_waypoints=len(targets),inspected_waypoints=len(solved),IK_iterations=iterations,all_static_waypoints_feasible=first is None,first_blocker=first or'NATIVE_DYNAMIC_APPROACH_NOT_YET_QUALIFIED',base_legs_object_fingers_unchanged=True,gripper_close='STATIC_GEOMETRY_ONLY_FINGERS_RESTORED'if closing else'NOT_TESTED',gripper_geometry=closing,contact_grasp_or_dynamic_qualification=False,actual_actuator_commands=0,expert_labels_added=0,new_physical_integrations=0,new_WM_updates=0,new_Actor_updates=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start,waypoints=solved)
    write_json(a.output,out);print(json.dumps({k:v for k,v in out.items()if k!='waypoints'}));return 0

if __name__=='__main__':raise SystemExit(main())
