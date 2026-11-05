"""Zero-integration redundant-state FK and teacher-forced diagnostic."""
def audit(spec):
    from pathlib import Path
    import numpy as np,mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.target_arx import CausalARX
    from go2arx5_vla.world_model.storage import digest
    if digest(spec['predictions'])!=spec['predictions_sha256']:raise ValueError('PREDICTION_IDENTITY_CHANGED')
    data,meta=load(spec['replay']);sl=CausalARX(meta).layout;model=mujoco.MjModel.from_binary_path(spec['model_path']);d=mujoco.MjData(model);site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'x5_ee');result={}
    with np.load(spec['predictions'],allow_pickle=False)as z,forbid_integration(mujoco):
        for label in z.files:
            p=z[label];rows={}
            for h in [1,8,16]:
                a=p[:,h-1];positions=[];quat=[];rot=[];ctrl=[]
                for s in a:
                    q=s[sl['qpos']];d.qpos[:]=q;d.qvel[:]=s[sl['qvel']];d.ctrl[:]=s[sl['ctrl']];mujoco.mj_forward(model,d);positions.append(d.site_xpos[site].copy());quat.append(abs(np.linalg.norm(q[3:7])-1));r=s[sl['tcp_rotation6d']].reshape(2,3).T;rot.append(float(np.linalg.norm(r.T@r-np.eye(2))));ctrl.append(float(np.linalg.norm(s[sl['ctrl']])))
                rows[str(h)]=dict(FK_vs_independent_TCP_RMSE_m=float(np.sqrt(np.mean((np.asarray(positions)-a[:,sl['tcp_position_m']])**2))),root_quaternion_norm_error_max=float(max(quat)),TCP_rotation6d_orthogonality_error_max=max(rot),ctrl_norm_max=max(ctrl),finite=bool(np.isfinite(a).all()),predicted_states_not_repaired=True)
            result[label]=rows
    return dict(schema='WM02_NEXT2_REDUNDANT_STATE_CONSISTENCY_V1',results=result,new_integrations=0,new_optimizer_updates=0,compact_response='Only TCP/BASE/BASE-relative TCP, not full safety state; consistency audit applies to106D predictors')

def main(argv=None):
    import argparse,json
    from pathlib import Path
    from go2arx5_vla.world_model.storage import write_json
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);v=audit(json.loads(a.spec.read_text()));write_json(a.output,v);print(json.dumps(v));return 0
if __name__=='__main__':raise SystemExit(main())
