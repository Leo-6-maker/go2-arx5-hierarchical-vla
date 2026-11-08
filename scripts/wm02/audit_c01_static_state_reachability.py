"""Finite FK/Jacobian search using saved C01 state; integration is forbidden.

Tests actual robot reachability of a few in-range source-coordinate poses.
It changes only private MjData trial coordinates. No reference, scene, safety,
controller, policy input, model checkpoint or running capsule is changed.
A static solution supplies neither a path certificate nor dynamic admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def pinned(path, pin):
    path=Path(path)
    if not(path.is_absolute()and path.resolve()==path and not path.is_symlink()):
        raise PermissionError('CANONICAL_STATIC_INPUT')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('PINNED_STATIC_INPUT')
    return raw


def audit(native_path,gpu_path,scope_path,scope_sha,*,source_targets=None):
    import numpy as np
    import mujoco
    from scripts.pi0292.relay_protocol import unseal
    from scripts.pi0294.task_state import parse_state
    from scripts.pi0294.task_contacts import allowed
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from go2arx5_vla.world_model.fusion_scope import binding_contract
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,matrix_to_quat
    from go2arx5_vla.task_data.codec import pose10
    start,cpu=time.monotonic(),time.process_time()
    scope=json.loads(pinned(scope_path,scope_sha));binding_contract(scope,time.time())
    if (os.uname().nodename!=scope['host']or os.getuid()!=scope['uid']or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
            or json.loads((Path(scope_path).parent/'LEASE.json').read_bytes())['status']!='ACTIVE'):
        raise PermissionError('EXISTING_FINITE_OWNER_CPU_ANALYSIS_ONLY')
    n=json.loads(pinned(native_path,'5369da7902f730138c96718920a3ac3392f8cc55524b64df71afaf66d1fc092a'))
    g=json.loads(pinned(gpu_path,'fed445c120457536793da4736712f52aa15b1234c7fd3c8d87c0224af2b83a1d'))
    scene='24fc70f0b9911312f17a083cd2f254252e0003583aac3520a4e3ce642253b45f'
    pinned(n['model_path'],scene)
    record=json.loads(pinned(g['recorded_observation'],'67fbdadffeaf23e7a911beccbaaf9550fcc7afc2fbe569fa4dbbc8f03c1ba5b8'))
    body=unseal(record);packet=parse_state(pinned(Path(g['recorded_observation']).parent/'STATE.private.bin',body['files']['STATE.private.bin']))
    source=unseal(json.loads(pinned(n['task_phase_source_observation'],n['task_phase_source_observation_sha256'])))['payload']
    stats=json.loads(pinned(Path(g['checkpoint'])/'processor/statistics.json',g['checkpoint_files']['processor/statistics.json']))['new_embodiment']['state']['eef_9d']
    frame=np.asarray(n['paused_source_from_target']);inverse=np.linalg.inv(frame)
    if (mujoco.__version__!='3.2.7' or mujoco.get_mjcb_control()is not None or packet['step']!=34303):
        raise PermissionError('EXACT_OFFLINE_MODEL_NO_CONTROL_CALLBACK')
    model=mujoco.MjModel.from_binary_path(n['model_path']);data=mujoco.MjData(model)
    if (model.nq,model.nv,model.nu)!=(34,32,20)or model.opt.timestep!=.002:raise ValueError('FROZEN_ROBOT_ABI_AND_DT')
    joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'x5_joint'+str(k))for k in range(1,7)]
    if any(j<0 for j in joints):raise ValueError('ACTUAL_SIX_ARM_JOINTS')
    qa=np.asarray(model.jnt_qposadr[joints]);va=np.asarray(model.jnt_dofadr[joints]);bounds=np.asarray(model.jnt_range[joints]);lo=bounds[:,0]+.03;hi=bounds[:,1]-.03
    site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'x5_ee')
    qsaved=packet['qpos'].copy();qinitial=qsaved[qa].copy();rng=np.random.default_rng(366107)
    seeds=[np.clip(qinitial,lo,hi),(lo+hi)/2]+[rng.uniform(lo,hi)for _ in range(10)]
    fk_calls=0;rows=[]
    def current(q):
        nonlocal fk_calls
        data.qpos[:]=qsaved;data.qpos[qa]=q;data.qvel[:]=0;data.ctrl[:]=0;data.time=packet['time_s']
        mujoco.mj_forward(model,data);fk_calls+=1
        value=np.eye(4);value[:3,:3]=data.site_xmat[site].reshape(3,3);value[:3,3]=data.site_xpos[site]
        return value
    def error(actual,target):
        quaternion=matrix_to_quat(target[:3,:3]@actual[:3,:3].T)
        if quaternion[3]<0:quaternion=-quaternion
        size=np.linalg.norm(quaternion[:3]);rotation=quaternion[:3]*(2*np.arctan2(size,quaternion[3])/size if size>1e-12 else 2.)
        return np.r_[target[:3,3]-actual[:3,3],.1*rotation]
    with forbid_integration(mujoco)as prohibited:
        actual=current(qinitial);mapped=frame@actual
        source_R=rot6d_rows_to_matrix(np.asarray(source['state_eef10'])[3:9])
        targets=([[float(mapped[0,3]),float(mapped[1,3]),z]for z in (.50,.54,.58)]
                 if source_targets is None else np.asarray(source_targets,float))
        if (np.asarray(targets).ndim!=2 or np.asarray(targets).shape[1]!=3 or not 1<=len(targets)<=30
                or not np.isfinite(targets).all() or np.any(np.asarray(targets)<np.asarray(stats['min'])[:3])
                or np.any(np.asarray(targets)>np.asarray(stats['max'])[:3])):
            raise ValueError('FINITE_PREDECLARED_SOURCE_POSITION_DESIGN_WITHIN_TRAIN_RANGE')
        for xyz in targets:
            goal=np.eye(4);goal[:3,:3]=source_R;goal[:3,3]=xyz;target=inverse@goal
            best=None
            for seed_index,seed in enumerate(seeds):
                q=seed.copy()
                for iteration in range(150):
                    if time.monotonic()-start>90:raise TimeoutError('FINITE_STATIC_90S_ENDPOINT')
                    actual=current(q);e=error(actual,target);loss=float(e@e)
                    if best is None or loss<best['loss']:best=dict(loss=loss,q=q.copy(),actual=actual.copy(),seed=seed_index,iteration=iteration)
                    if np.linalg.norm(e[:3])<1e-5 and np.linalg.norm(e[3:])<1e-5:break
                    jp=np.empty((3,model.nv));jr=np.empty_like(jp);mujoco.mj_jacSite(model,data,jp,jr,site)
                    J=np.vstack((jp[:,va],.1*jr[:,va]));delta=J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),e)
                    delta*=min(1.,.15/max(float(np.max(np.abs(delta))),1e-12));improved=False
                    for factor in (1.,.5,.2,.1):
                        trial=np.clip(q+factor*delta,lo,hi);te=error(current(trial),target)
                        if float(te@te)<loss:q=trial;improved=True;break
                    if not improved:break
                if best['loss']<2e-10:break
            achieved=current(best['q']);mapped_achieved=frame@achieved;vector=pose10(mapped_achieved,source['state_eef10'][9])[:9]
            e=error(achieved,target);bad=np.flatnonzero((vector<np.asarray(stats['min'])-1e-7)|(vector>np.asarray(stats['max'])+1e-7))
            contacts=[]
            for index in range(data.ncon):
                c=data.contact[index];names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(j))for j in(c.geom1,c.geom2)]
                if not allowed(*names):contacts.append(dict(geoms=names,distance_m=float(c.dist)))
            rows.append(dict(requested_source_pose=goal.tolist(),achieved_source_pose=mapped_achieved.tolist(),arm_q=best['q'].tolist(),
                position_error_m=float(np.linalg.norm(e[:3])),rotation_error_rad=float(np.linalg.norm(e[3:])/.1),
                outside_training_minmax_components=bad.tolist(),minimum_model_joint_limit_margin_rad=float(np.minimum(best['q']-bounds[:,0],bounds[:,1]-best['q']).min()),
                masked_static_contacts_not_allowed_by_original_classifier=contacts,seed_index=best['seed'],iteration=best['iteration'],
                static_pose_match=bool(np.linalg.norm(e[:3])<.001 and np.linalg.norm(e[3:])/.1<.01),dynamic_qualification=False,path_certificate=False,
                collision_check_limit='ORIGINAL_MASKED_MJ_FORWARD_CONTACTS_ONLY_NOT_DENSE_WHOLE_PATH_OR_NATIVE_CERTIFICATE'))
        if prohibited:raise ValueError('STATIC_ANALYSIS_CALLED_INTEGRATION')
    return dict(schema='WM02_C01_STATIC_SOURCE_STATE_REACHABILITY_ACTUAL_V1',scene_sha256=scene,MuJoCo_version=mujoco.__version__,
        saved_native_step=packet['step'],saved_native_time_s=packet['time_s'],actual_initial_mapped_pose=mapped.tolist(),
        targets=rows,static_FK_calls=fk_calls,prohibited_integration_calls=len(prohibited),native_integrations=0,model_queries=0,model_updates=0,
        base_and_other_saved_qpos_fixed_for_FK_only=True,robot_dynamics_or_scene_changed=False,WBC_changed=False,
        not_a_hidden_state_restore=True,not_an_execution_grant=True,source_inputs_changed=False,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-spec',type=Path,required=True);p.add_argument('--gpu-spec',type=Path,required=True);p.add_argument('--scope',type=Path,required=True);p.add_argument('--scope-sha256',required=True);a=p.parse_args()
    print(json.dumps(audit(a.native_spec,a.gpu_spec,a.scope,a.scope_sha256)))
    return 0
if __name__=='__main__':raise SystemExit(main())
