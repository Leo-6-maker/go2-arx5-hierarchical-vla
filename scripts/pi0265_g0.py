"""Actual native CPU contract checks: no simulator construction or actor forwards."""
import isaacgym
import ast
import hashlib
import json
import math
import textwrap
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pi0265_source import ROOT, BASE, sha, write
from pi0265_metrics import self_check, so3

WBC=Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare')


def decode(value,version):
    d=np.asarray(value)
    pair=d.reshape(-1,3,2).transpose(0,2,1) if version=='LEGACY' else d.reshape(-1,2,3)
    a=pair[:,0];a=a/np.linalg.norm(a,axis=-1)[:,None]
    b=pair[:,1]-(a*pair[:,1]).sum(-1)[:,None]*a;b=b/np.linalg.norm(b,axis=-1)[:,None]
    matrix=np.stack((a,b,np.cross(a,b)),axis=1)
    return matrix.transpose(0,2,1) if version=='LEGACY' else matrix


def main():
    from go1_gym.envs.config import RoboDuetRuntimeOptions,build_roboduet_config
    from stage21_native_receding_horizon import _configure_stage21
    import pytorch3d.transforms as transforms
    from modules.trajectory import to_canonical,quat_to_mat,TrajectoryBatch
    from go1_gym.utils.math_utils import pose_world_to_body_9d,quat_xyzw_to_rot6d
    from isaacgym.torch_utils import quat_conjugate,quat_mul,quat_apply
    source=json.loads((ROOT/'evidence/SOURCE_IDENTITY.json').read_text())
    for row in source['files']:assert sha(row['path'])==row['sha256'],row['name']
    cfg=build_roboduet_config(options=RoboDuetRuntimeOptions(num_envs=1,robot='go2_x5',use_rot6d=True,
        dyna_gait=True,goal_reaching=True,traj_tracking=True,arm_action_mode='end_to_end',reach_table=True),debug=True)
    _configure_stage21(cfg)
    dt=float(cfg.sim.dt);decimation=int(cfg.control.decimation);control_dt=dt*decimation
    assert abs(control_dt-.02)<1e-8
    stage=Path('/var/cache/users/liuyu-nfs/stage41_corelab_long_horizon_20260908/chain/M2')
    seal=json.loads((stage/'M2_result.seal.json').read_text());assert seal['result']['status']=='PASS'
    records=seal['result']['evidence_manifest']['records']
    for item in records:assert sha(stage/item['relative_path'])==item['sha256'],item['relative_path']
    stage_receipt=dict(seal_sha256=sha(stage/'M2_result.seal.json'),files=len(records),all_readbacks_match=True,
        classification=seal['classification'],counters=seal['result']['counters'],forbidden_operations=seal['result']['forbidden_operations'])
    run=WBC/'runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018'
    metadata=[]
    for p in [run/'params.txt']+sorted((run/'scripts').rglob('*.py')):
        text=p.read_text();hits=[dict(line=i+1,text=s.strip()) for i,s in enumerate(text.splitlines())
            if any(x in s for x in ('matrix_to_rotation_6d','rotation_6d_to_matrix','pytorch3d','use_rot6d'))]
        metadata.append(dict(path=str(p),sha256=sha(p),import_or_packing_lines=hits))
    # Archived imports identify PyTorch3D use, but do not establish its installed training version.
    convention='UNKNOWN_INSTALLED_TRAINING_PYTORCH3D_NOT_RECORDED'
    matrices=[np.eye(3)]+[Rotation.from_rotvec(np.eye(3)[i]*np.pi/2).as_matrix() for i in range(3)]
    matrices+=[Rotation.from_euler('xyz',[.31,-.47,.83]).as_matrix(),Rotation.from_euler('zyx',[-.7,.4,.2]).as_matrix()]
    reference=BASE/'PI0252/pre_full_sim/controls/A_attempt2/actual_expert_reference.npz'
    recorded=dict(np.load(reference,allow_pickle=False))
    qkey=next(k for k in recorded if 'quaternion' in k or k=='quaternion_xyzw')
    recorded_q=recorded[qkey].reshape(-1,4)
    matrices+=list(Rotation.from_quat(recorded_q[[0,-1]]).as_matrix())
    matrix=np.asarray(matrices);assert np.max(abs(np.linalg.det(matrix)-1))<1e-10
    actual=torch.tensor(matrix,dtype=torch.float32);legacy=transforms.matrix_to_rotation_6d
    rotation_results=[]
    for version in ('LEGACY','ROW6D'):
        transforms.matrix_to_rotation_6d=legacy if version=='LEGACY' else lambda m:m[..., :2,:].clone().reshape(m.shape[:-2]+(6,))
        packed=transforms.matrix_to_rotation_6d(actual).numpy();decoded=decode(packed,version)
        residual=float(np.max(so3(decoded,matrix)));assert residual<=1e-4
        assert np.isfinite(packed).all() and np.max(abs(decoded@decoded.transpose(0,2,1)-np.eye(3)))<1e-5
        q=torch.tensor(Rotation.from_matrix(matrix).as_quat(),dtype=torch.float32)
        native_matrix=quat_to_mat(q).numpy();assert np.max(so3(native_matrix,matrix))<1e-4
        # Nonidentity root/body/current and preview frames must agree independently.
        root=matrix[4];heading=Rotation.from_euler('z',.63).as_matrix();position=np.array([.29,-.13,.61]);base=np.array([.1,.07,.34])
        for frame in (root,heading):
            local=(position-base)@frame;assert np.linalg.norm(frame@local+base-position)<=1e-5
            local_r=frame.T@matrix
            d=decode(transforms.matrix_to_rotation_6d(torch.tensor(local_r,dtype=torch.float32)).numpy(),version)
            assert np.max(so3(frame@d,matrix))<=1e-4
        native_body=pose_world_to_body_9d(torch.tensor(np.tile(position,(len(matrix),1)),dtype=torch.float32),q,
            torch.tensor(np.tile(base,(len(matrix),1)),dtype=torch.float32),
            torch.tensor(np.tile(Rotation.from_matrix(root).as_quat(),(len(matrix),1)),dtype=torch.float32)).numpy()
        assert np.max(np.linalg.norm(native_body[:,:3]@root.T+base-position,axis=-1))<=1e-5
        assert np.max(so3(root@decode(native_body[:,3:],version),matrix))<=1e-4
        rotation_results.append(dict(version=version,max_roundtrip_so3_rad=residual,nonidentity_cases=len(matrix),
            current_body_and_future_heading_frames=True,physical_matrix_quaternion_unchanged=True))
    transforms.matrix_to_rotation_6d=legacy
    t=np.arange(401)*.05;p=np.column_stack((.1*np.clip(t/6,0,1),.02*np.clip(t/6,0,1),np.zeros(len(t))))
    q=np.repeat(recorded_q[:1],len(t),axis=0)
    gamma,law=to_canonical(torch.tensor(p,dtype=torch.float32),quat_to_mat(torch.tensor(q,dtype=torch.float32)),.05)
    for obj in (gamma,law):
        for value in vars(obj).values():
            if hasattr(value,'isfinite'):assert torch.isfinite(value).all()
    assert cfg.wbc.goal_reaching.trajectory.max_tl_points>=len(t)
    batch=TrajectoryBatch(1,1536,device='cpu',max_tl_points=512);batch.load([0],[gamma],[law])
    wbc_path=next(Path(row['path']) for row in source['files'] if row['name']=='wbc_env')
    wbc_text=wbc_path.read_text();preview_node=next(n for n in ast.walk(ast.parse(wbc_text))
        if isinstance(n,ast.FunctionDef) and n.name=='_arm_traj_obs_terms')
    scope=dict(torch=torch,np=np,quat_conjugate=quat_conjugate,quat_mul=quat_mul,quat_apply=quat_apply,
        quat_xyzw_to_rot6d=quat_xyzw_to_rot6d)
    exec(compile(textwrap.dedent(ast.get_source_segment(wbc_text,preview_node)),str(wbc_path),'exec'),scope)
    heading=Rotation.from_euler('z',.63).as_matrix();heading_q=torch.tensor(Rotation.from_matrix(heading).as_quat()[None],dtype=torch.float32)
    zero=torch.zeros(1);preview=SimpleNamespace(cfg=cfg,num_envs=1,traj_batch=batch,traj_s=zero,
        traj_timing_err=zero,traj_sdot_meas=zero,traj_sdot_ref=zero,base_pos=torch.tensor([[.1,.07,.34]]),
        _heading_quat=lambda:heading_q,
        _reach_query=lambda p:(torch.ones(p.shape[:2]),torch.zeros_like(p),torch.ones(p.shape[:2])))
    previews=[]
    for version in ('LEGACY','ROW6D'):
        transforms.matrix_to_rotation_6d=legacy if version=='LEGACY' else lambda m:m[..., :2,:].clone().reshape(m.shape[:-2]+(6,))
        actual=scope['_arm_traj_obs_terms'](preview).numpy();k=int(cfg.wbc.goal_reaching.trajectory.preview_points)
        stored=actual[:,8:].reshape(1,k,14);s,world,quaternion,_=batch.sample_preview(zero,float(cfg.wbc.goal_reaching.trajectory.preview_horizon),k)
        assert np.isfinite(actual).all()
        xyz=float(np.max(np.linalg.norm(stored[0,:,:3]@heading.T+preview.base_pos.numpy()-world.numpy()[0],axis=-1)))
        rot=float(np.max(so3(heading@decode(stored[0,:,3:9],version),Rotation.from_quat(quaternion.numpy()[0]).as_matrix())))
        assert xyz<=1e-5 and rot<=1e-4
        previews.append(dict(version=version,actual_source_function=True,position_roundtrip_m=xyz,SO3_roundtrip_rad=rot,
            reach_urgency_stubbed_only_for_coordinate_fixture=True,preview_points=k))
    transforms.matrix_to_rotation_6d=legacy
    math_paths=[p for p in (WBC/'go1_gym').rglob('*.py') if 'def quat_error_axis_angle' in p.read_text()]
    assert len(math_paths)==1
    text=math_paths[0].read_text();node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=='quat_error_axis_angle')
    native_error=ast.get_source_segment(text,node)
    assert 'sign' in native_error and 'acos' not in native_error and 'atan' not in native_error
    budget=dict(planned_starts=344,hard_starts=352,hard_simulated_seconds=7392.,sim_dt=dt,decimation=decimation,
        control_dt=control_dt,sim_substeps=int(cfg.sim.substeps),hard_control_ticks=math.ceil(7392/control_dt),
        hard_simulate_calls=math.ceil(7392/dt),hard_integration_substeps=math.ceil(7392/dt)*int(cfg.sim.substeps),
        hard_actor_calls=2*math.ceil(7392/control_dt),implicit_reset_steps_included=True,
        GR00T_queries=11000,S4_heads=44000,training_updates=8480,training_head_cap=210000)
    write(ROOT/'design/BUDGET_CARD.json',budget)
    result=dict(status='PASS_NATIVE_CPU_CONTRACT',source_sha256=sha(ROOT/'evidence/SOURCE_IDENTITY.json'),
        stage41=stage_receipt,training_convention=convention,training_metadata=metadata,rotation_checks=rotation_results,actual_preview_checks=previews,
        native_error_proxy=dict(path=str(math_paths[0]),sha256=sha(math_paths[0]),unchanged=True,feature='bounded signed quaternion vector'),
        full_20s_hold_canonical=dict(points=len(t),finite=True),metrics=self_check(),budget=budget,
        new_physics_starts=0,GR00T_queries=0,WBC_actor_calls=0)
    write(ROOT/'evidence/G0_CPU.json',result);print(json.dumps(result))


if __name__=='__main__':main()
