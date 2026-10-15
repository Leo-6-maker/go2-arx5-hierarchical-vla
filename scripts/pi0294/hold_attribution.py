"""Saved-state FK/Jacobian and command accounting. Never integrates or renders."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scripts.pi0292.durable import read,write
from .r4a import validate_scope,cpu_stage
from go2arx5_vla.trajectory.so3 import quat_to_matrix,matrix_to_quat


def logrot(r):
    theta=np.arccos(np.clip((np.trace(r)-1)/2,-1,1))
    return np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])*(.5 if theta<1e-8 else theta/(2*np.sin(theta)))


def summary(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return dict(status='UNKNOWN_NO_MATCHED_SAMPLES')
    return dict(samples=len(a),mean=np.mean(a,axis=0).tolist(),RMS=np.sqrt(np.mean(a*a,axis=0)).tolist(),
        abs_max=np.max(np.abs(a),axis=0).tolist())


def stamp(row):
    s=row['msg']['header']['stamp'];return s['sec']+s['nanosec']*1e-9


def interp(policy,t):
    times=np.asarray(policy['time_trajectory']);states=np.asarray([s['value'] for s in policy['state_trajectory']])
    if t<times[0]-1e-8 or t>times[-1]+1e-8:raise ValueError('OUTSIDE_SAVED_PLAN')
    return np.array([np.interp(t,times,states[:,k]) for k in range(states.shape[1])])


def strict_snapshots(root,model_path):
    """Same native sensor/control stamp; independent of physics-log ZOH choice."""
    import mujoco as mj
    root=Path(root);validate_scope(root)
    with cpu_stage(root,'R4B_D0_SAME_STAMP_FORCE_CHECK',cores=1,cpu_s_cap=120):
        p=root/'fullbody/native-hold-01';task=read(p/'TASK_START.json');out=root/'attribution/R4B_D0'
        model=mj.MjModel.from_binary_path(str(model_path));data=mj.MjData(model)
        def forbidden(*a,**kw):raise AssertionError('READ_ONLY')
        mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
        rows=[json.loads(x) for x in (p/'ros.jsonl').read_text().splitlines()]
        measured={round(stamp(x),9):x['msg'] for x in rows if x['kind']=='arm'}
        odom={round(stamp(x),9):x['msg'] for x in rows if x['kind']=='odom'}
        controls=[json.loads(x) for x in (p/'control.jsonl').read_text().splitlines()]
        physical=[json.loads(x) for x in (p/'physics.jsonl').read_text().splitlines()]
        samples=[x for x in physical if x['kind']=='sample' and 0<=x['sim_time_s']-task['mpc_time_s']<=10.075]
        qa=np.array([model.jnt_qposadr[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'x5_joint'+str(k))] for k in range(1,7)])
        va=np.array([model.jnt_dofadr[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'x5_joint'+str(k))] for k in range(1,7)])
        act=np.array([int(np.flatnonzero(model.actuator_trnid[:,0]==mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'x5_joint'+str(k)))[0]) for k in range(1,7)])
        target=quat_to_matrix(task['initial_tcp'][3:]);site=mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,'x5_ee')
        values=[];times=[]
        for c in controls:
            t=c['sim_time_s'];elapsed=t-task['mpc_time_s'];key=round(t,9)
            if not 0<=elapsed<=10.075 or key not in measured or key not in odom:continue
            arm=measured[key];pose=odom[key]['pose']['pose'];rotation=quat_to_matrix([pose['orientation'][k] for k in ('x','y','z','w')])
            q=np.asarray(samples[0]['qpos']).copy();q[:3]=[pose['position'][k] for k in ('x','y','z')]-rotation@np.array([-.02557,0,.04232]);q[3:7]=np.roll(matrix_to_quat(rotation),1);q[qa]=arm['position']
            reference=np.array(c['q_target'][12:18]);record=[]
            for subset in ([],[2],[3],[2,3],list(range(6))):
                qc=q.copy();qc[qa[subset]]=reference[subset];data.qpos[:]=qc;mj.mj_kinematics(model,data)
                record.extend([np.linalg.norm(data.site_xpos[site]-task['initial_tcp'][:3]),np.linalg.norm(logrot(target.T@data.site_xmat[site].reshape(3,3)))])
            values.append(record);times.append(elapsed)
        a=np.array(values);late=np.array(times)>=8
        ctrl_vs_force=[];bias=[];actforce=[];post_phase_residual=[]
        for s in samples:
            ctrl=np.asarray(s['ctrl'])[act];expected=np.clip(ctrl,model.actuator_ctrlrange[act,0],model.actuator_ctrlrange[act,1]);actual=np.asarray(s['qfrc_actuator'])[va]
            ctrl_vs_force.append(actual-expected)
            if s['sim_time_s']-task['mpc_time_s']>=8:bias.append(np.asarray(s['qfrc_bias'])[va]);actforce.append(actual)
        result=dict(status='SAME_NATIVE_CONTROL_ARM_ODOMETRY_STAMP_COUNTERFACTUAL',samples=len(a),last2s_samples=int(late.sum()),
            columns=['actual_XYZ','actual_SO3','joint3_XYZ','joint3_SO3','joint4_XYZ','joint4_SO3','joint3_4_XYZ','joint3_4_SO3','all6_XYZ','all6_SO3'],
            whole=summary(a),last2s=summary(a[late]),
            timing='Native published sensor snapshots and control targets have identical simulation stamp; sensor evaluation may precede integration by2ms but all compared fields are from that same control cycle. This is an offline target-substitution, not an achieved trajectory.',
            actuator_force_minus_clipped_ctrl_Nm=summary(ctrl_vs_force),last2s_saved_bias_Nm=summary(bias),last2s_actual_actuator_force_Nm=summary(actforce),
            actual_consumed_policy_id='UNKNOWN',actual_applied_velocity='UNKNOWN_NOT_LOGGED',new_physics=0,new_model_queries=0,new_render=0)
        write(out/'STRICT_ALIGNMENT_AND_FORCE_CHECK.json',result);print(json.dumps(result))


def main(root,model_path):
    import mujoco as mj
    root=Path(root);validate_scope(root)
    with cpu_stage(root,'R4B_D0_SOURCE_CAUSAL_R1',cores=1,cpu_s_cap=1200):
        p=root/'fullbody/native-hold-01';out=root/'attribution/R4B_D0_CAUSAL_R1';out.mkdir(parents=True,exist_ok=False)
        original=read(p/'CPU_VERDICT.json');task=read(p/'TASK_START.json')
        for name,h in original['original_array_hashes'].items():
            if name in ('physics.jsonl','control.jsonl','ros.jsonl','planner.jsonl','TASK_START.json'):
                assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h
        assert mj.__version__=='3.2.7'
        assert hashlib.sha256(Path(model_path).read_bytes()).hexdigest()=='168b1232df2a50a163709c1c1e57d13d39aa3efb92a4c98005e797924142376d'
        def forbidden(*a,**kw):raise AssertionError('READ_ONLY_NO_INTEGRATION')
        mj.mj_step=mj.mj_step1=mj.mj_step2=forbidden
        model=mj.MjModel.from_binary_path(str(model_path));data=mj.MjData(model);grav=mj.MjData(model)
        site=mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,'x5_ee')
        joints=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'x5_joint'+str(k)) for k in range(1,7)]
        qa=model.jnt_qposadr[joints];va=model.jnt_dofadr[joints]
        actuators=[int(np.flatnonzero(model.actuator_trnid[:,0]==j)[0]) for j in joints]
        physics=[json.loads(x) for x in (p/'physics.jsonl').read_text().splitlines()]
        samples=[x for x in physics if x['kind']=='sample' and 0<=x['sim_time_s']-task['mpc_time_s']<=10.075]
        controls=[json.loads(x) for x in (p/'control.jsonl').read_text().splitlines()];ct=np.array([x['sim_time_s'] for x in controls])
        ros=[json.loads(x) for x in (p/'ros.jsonl').read_text().splitlines()]
        requests=[x for x in ros if x['kind']=='arm_request'];rt=np.array([stamp(x) for x in requests])
        policies=[x for x in ros if x['kind']=='policy'];pt=np.array([x['monotonic_s'] for x in policies])
        arm=[x for x in ros if x['kind']=='arm'];at=np.array([stamp(x) for x in arm])
        observations=[x for x in ros if x['kind']=='observation'];ee=[x for x in ros if x['kind']=='ee']
        et={round(x['msg']['time'],9):x['msg']['state']['value'] for x in ee}
        target=np.eye(4);target[:3,3]=task['initial_tcp'][:3];target[:3,:3]=quat_to_matrix(task['initial_tcp'][3:])
        def fk(q):
            data.qpos[:]=q;mj.mj_kinematics(model,data);mj.mj_comPos(model,data)
            t=np.eye(4);t[:3,3]=data.site_xpos[site];t[:3,:3]=data.site_xmat[site].reshape(3,3);return t
        def planned(q,state):
            x,y,z,yaw,pitch,roll=state[:6]
            cy,sy,cp,sp,cr,sr=np.cos(yaw),np.sin(yaw),np.cos(pitch),np.sin(pitch),np.cos(roll),np.sin(roll)
            rotation=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
            q=q.copy();q[:3]=[x,y,z]-rotation@np.array([.05,0,.1]);q[3:7]=np.roll(matrix_to_quat(rotation),1);q[qa]=state[6:12];return q
        closure=[]
        for obs in observations:
            key=round(obs['msg']['time'],9)
            if key not in et:continue
            t=fk(planned(np.array(samples[0]['qpos']),np.array(obs['msg']['state']['value'])))
            actual=et[key];closure.append([np.linalg.norm(t[:3,3]-actual[:3]),np.linalg.norm(logrot(quat_to_matrix(actual[3:]).T@t[:3,:3]))])
        assert closure and np.max(np.asarray(closure),axis=0).max()<1e-5,'NATIVE_MJCF_MODEL_FRAME_CLOSURE_FAILED'
        # Numerical Jacobian check on a private state; no live state exists here.
        q0=np.array(samples[-1]['qpos']);r0=fk(q0)[:3,:3].copy();jp=np.zeros((3,model.nv));jr=jp.copy();mj.mj_jacSite(model,data,jp,jr,site)
        jac_error=[]
        for k in range(6):
            q=q0.copy();q[qa[k]]+=1e-6;rot=fk(q)[:3,:3];jac_error.append(np.linalg.norm(r0@logrot(r0.T@rot)/1e-6-jr[:,va[k]]))
        assert max(jac_error)<1e-5
        chain={name:[] for name in ('R0_R1','R1_R2','R2_R3','R3_R4','R4_R5','R0_R5')}
        levels={str(k):[] for k in range(1,6)};cf={name:[] for name in ('joint3','joint4','joint3_4','other4','all6')}
        jac_res=[];composite=[];ages=[];plan_matches=[];rows=[];gravity=[];force_phase=[];saturation=[]
        stream=(out/'POSE_CHAIN_ROWS.private.jsonl').open('w')
        for s in samples:
            t=s['sim_time_s'];q=np.array(s['qpos']);ci=np.searchsorted(ct,t+1e-10,side='right')-1;ri=np.searchsorted(rt,t+1e-10,side='right')-1;pi=np.searchsorted(pt,s['monotonic_s'],side='right')-1
            while ci>=0 and controls[ci]['monotonic_s']>s['monotonic_s']:ci-=1
            while ri>=0 and requests[ri]['monotonic_s']>s['monotonic_s']:ri-=1
            if min(ci,ri,pi)<0:continue
            c=controls[ci];request=requests[ri]
            pi=np.searchsorted(pt,request['monotonic_s'],side='right')-1
            if pi<0:continue
            rq=np.array(request['msg']['points'][-1]['positions']);ap=np.array(c['q_target'][12:18]);policy=policies[pi]['msg'];preview=stamp(request)+.15
            try:state=interp(policy,min(preview,policy['time_trajectory'][-1]))
            except ValueError:continue
            predicted=state[6:12];match=float(np.max(np.abs(predicted-rq)));plan_matches.append(match)
            r1=fk(planned(q,state));q2=q.copy();q2[qa]=predicted;r2=fk(q2);q3=q.copy();q3[qa]=rq;r3=fk(q3);q4=q.copy();q4[qa]=ap;r4=fk(q4);r5=fk(q)
            rs=[target,r1,r2,r3,r4,r5];product=np.eye(3)
            for k in range(5):
                relative=rs[k][:3,:3].T@rs[k+1][:3,:3];product=product@relative
                common=target[:3,:3].T@rs[k][:3,:3]@logrot(relative)
                chain['R'+str(k)+'_R'+str(k+1)].append([np.linalg.norm(rs[k+1][:3,3]-rs[k][:3,3]),np.linalg.norm(logrot(relative)),*common])
            total=target[:3,:3].T@r5[:3,:3];composite.append(np.linalg.norm(product-total));chain['R0_R5'].append([np.linalg.norm(r5[:3,3]-target[:3,3]),np.linalg.norm(logrot(total)),*logrot(total)])
            for k in range(1,6):levels[str(k)].append([np.linalg.norm(rs[k][:3,3]-target[:3,3]),np.linalg.norm(logrot(target[:3,:3].T@rs[k][:3,:3]))])
            fk(q);mj.mj_jacSite(model,data,jp,jr,site)
            for name,which in (('joint3',[2]),('joint4',[3]),('joint3_4',[2,3]),('other4',[0,1,4,5]),('all6',list(range(6)))):
                qc=q.copy();qc[qa[which]]=ap[which];pose=fk(qc);cf[name].append([np.linalg.norm(pose[:3,3]-target[:3,3]),np.linalg.norm(logrot(target[:3,:3].T@pose[:3,:3]))])
                if name=='joint3_4':
                    dq=np.zeros(6);dq[which]=ap[which]-q[qa[which]]
                    local=target[:3,:3].T@r5[:3,:3]@logrot(r5[:3,:3].T@pose[:3,:3]);linear=target[:3,:3].T@jr[:,va]@dq;jac_res.append(np.linalg.norm(local-linear))
            grav.qpos[:]=q;grav.qvel[:]=0;grav.ctrl[:]=0;mj.mj_forward(model,grav);gravity.append(grav.qfrc_bias[va].copy())
            saturation.append(np.minimum(np.array(s['ctrl'])[actuators]-model.actuator_ctrlrange[actuators,0],model.actuator_ctrlrange[actuators,1]-np.array(s['ctrl'])[actuators]))
            # Saved dynamics is pre-integration; M at saved post-integration q is diagnostic only.
            data.qpos[:]=q;data.qvel[:]=s['qvel'];data.ctrl[:]=s['ctrl'];mj.mj_forward(model,data);mass=np.empty((model.nv,model.nv));mj.mj_fullM(model,mass,data.qM)
            residual=mass@np.array(s['qacc'])+np.array(s['qfrc_bias'])-np.array(s['qfrc_actuator'])-np.array(s['qfrc_passive'])-np.array(s['qfrc_constraint']);force_phase.append(residual[va])
            ages.append([t-c['sim_time_s'],t-stamp(request),s['monotonic_s']-policies[pi]['monotonic_s']])
            entry=dict(sim_s=t,relative_s=t-task['mpc_time_s'],control_age_s=ages[-1][0],request_age_s=ages[-1][1],policy_owner_receipt_age_s=ages[-1][2],candidate_arm_plan_match_max_rad=match,
                candidate_plan_observation_s=policy['init_observation']['time'],candidate_valid_horizon=[policy['time_trajectory'][0],policy['time_trajectory'][-1]],arm_preview_time_s=preview,
                actual_base_with_actual_arm=r5.tolist(),candidate_levels=[r.tolist() for r in rs],applied_velocity='UNKNOWN_NOT_LOGGED',exact_consumed_policy_id='UNKNOWN_NOT_LOGGED')
            stream.write(json.dumps(entry)+'\n');rows.append(entry)
        stream.close()
        # Align original native control records with the actually published measured arm state.
        torque=[];torque_rows=[];ctrl_arm_matches=[]
        for c in controls:
            t=c['sim_time_s'];elapsed=t-task['mpc_time_s']
            if not 0<=elapsed<=10.075:continue
            inds=np.flatnonzero(np.abs(at-t)<1e-8)
            if not len(inds):continue
            measured=arm[int(inds[-1])]['msg'];q=np.array(measured['position']);qd=np.array(measured['velocity']);kp=np.array(c['kp'][12:18]);kd=np.array(c['kd'][12:18]);ff=np.array(c['tau'][12:18]);u=np.array(c['ctrl'])[actuators]
            P=kp*(np.array(c['q_target'][12:18])-q);D=u-ff-P
            ri=np.searchsorted(rt,t+1e-10,side='right')-1
            if ri<0:continue
            dq_requested=np.array(requests[ri]['msg']['points'][-1]['velocities']);D_from_request=kd*(dq_requested-qd)
            torque.append(np.r_[P,D,ff,u,D-D_from_request]);torque_rows.append(dict(sim_s=t,elapsed_s=elapsed,P_from_native_float_sensor=P.tolist(),D_residual_inferred_not_recorded=D.tolist(),tau_ff_recorded=ff.tolist(),ctrl_recorded=u.tolist(),requested_dq=dq_requested.tolist(),applied_dq='UNKNOWN_NOT_LOGGED'))
        (out/'TORQUE_ROWS.private.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in torque_rows))
        times=np.array([x['relative_s'] for x in rows]);late=times>=8
        def windows(values):
            a=np.array(values);return dict(whole=summary(a),last2s=summary(a[late]))
        coverage=dict(status='READ_ONLY_SAVED_SIGNAL_AUDIT',reference='OBSERVED_INSTALLED_CONSTANT',MPC_trajectories='OBSERVED',consumed_policy_identity='UNKNOWN_NOT_RECORDED',
            R1_R2='CAUSAL_OWNER_RECEIVED_POLICY_CANDIDATE_AT_ARM_PREVIEW_TIME_NOT_PROVEN_CONSUMED',R3='OBSERVED_REQUEST_ZOH',R4='OBSERVED_NATIVE_TARGET_ZOH_WITH_REPORTED_AGE',
            native_applied_dq='UNKNOWN_NOT_RECORDED',native_control_cycle_state='PUBLISHED_FLOAT_SENSOR_AT_MATCHING_SIM_STAMP',
            dynamics_phase='FORCES_AND_QACC_PRE_INTEGRATION_QPOS_POST_INTEGRATION_BY_0.002s',exact_dynamics_balance='UNKNOWN_PRESTEP_QPOS_MISSING',
            alignment='SOURCE_TIMESTAMPS_AND_CAUSAL_OWNER_RECEIPT_FOR_CONTROL_REQUEST_POLICY_ZOH_NO_FITTED_TIME_SHIFT',ages_control_request_policy_s=summary(ages),
            source_model_frame_closure=summary(closure),plan_request_match_abs_rad=summary(plan_matches),candidate_plan_matches_2e_6_rad=int(np.sum(np.array(plan_matches)<=2e-6)),
            total_candidate_plan_samples=len(plan_matches),native_gravity='RECONSTRUCTED_ZERO_VELOCITY_AT_SAVED_POST_STATE',new_physics=0,new_render=0,new_model_queries=0)
        write(out/'SIGNAL_COVERAGE_AND_ALIGNMENT.json',coverage)
        pose=dict(status='OFFLINE_COUNTERFACTUAL_NOT_EXECUTED',levels_vs_constant_reference={k:windows(x) for k,x in levels.items()},
            relative_segments={k:windows(x) for k,x in chain.items()},segment_columns=['XYZ_norm_m','SO3_angle_rad','target_frame_rotvec_x','target_frame_rotvec_y','target_frame_rotvec_z'],
            joint_counterfactual={k:windows(x) for k,x in cf.items()},FK_jacobian_max_finite_difference_error=max(jac_error),joint3_4_Jacobian_nonlinear_residual_rad=summary(jac_res),
            exact_SO3_product_max_Frobenius_residual=max(composite),scalar_RMS_angles_additive=False,actual_consumed_plan_attribution='UNKNOWN_NO_POLICY_CONSUMER_LOG',
            actual_native_observation_FK_model_closure=summary(closure),original_FAIL_unchanged=True,new_physics=0)
        write(out/'POSE_CHAIN_DECOMPOSITION.json',pose)
        torque_np=np.array(torque);late_t=np.array([x['elapsed_s'] for x in torque_rows])>=8
        force=dict(status='OBSERVED_COMMAND_PLUS_EXPLICIT_RECONSTRUCTIONS',joint_order=['x5_joint'+str(k) for k in range(1,7)],
            control_samples=len(torque),P_Dresidual_FF_ctrl_Dresidual_minus_requestedD_columns=5,
            whole=summary(torque_np.reshape(-1,5,6)),last2s=summary(torque_np[late_t].reshape(-1,5,6)),
            D_measured=False,D_inference='ctrl - recorded_FF - Kp*(native_target-native_published_float_q); float-vs-double sensor residual retained',
            requested_dq_is_proven_applied=False,zero_velocity_gravity=windows(gravity),
            actuator_metadata=[dict(joint='x5_joint'+str(k+1),actuator_id=a,gear=model.actuator_gear[a].tolist(),ctrl_limited=bool(model.actuator_ctrllimited[a]),ctrlrange=model.actuator_ctrlrange[a].tolist(),force_limited=bool(model.actuator_forcelimited[a]),forcerange=model.actuator_forcerange[a].tolist()) for k,a in enumerate(actuators)],
            ctrlrange_margin=summary(saturation),ctrlrange_margin_only_applies_to_limited_actuators=True,
            external_generalized_force='UNKNOWN_NOT_RECORDED',
            post_M_times_saved_pre_qacc_balance_diagnostic_only=windows(force_phase),exact_actual_M_vdot_balance='UNKNOWN_PRESTEP_QPOS_AND_EXTERNAL_FORCE_NOT_SAVED',
            qfrc_bias_is_pure_gravity=False,new_physics=0)
        write(out/'JOINT_TORQUE_ACCOUNTING.json',force)
        write(out/'MODE_AND_PATCH_RESULTS.json',dict(status='NOT_RUN_NO_NEW_EXECUTION_AUTHORITY',physics_starts=0,integrations=0,controller_changes=0,
            D1_authorizes_execution=False,source_release='134389406df96711c97b8557d50ee620f741b3df',old_native_QUALITY_FAIL_retained=True))
        print(json.dumps(dict(coverage=coverage,late_pose={k:x['last2s'] for k,x in pose['levels_vs_constant_reference'].items()},late_counterfactual={k:x['last2s'] for k,x in pose['joint_counterfactual'].items()},late_torque=force['last2s'])))


if __name__=='__main__':
    import sys
    main(sys.argv[1],sys.argv[2])
