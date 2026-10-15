"""Source-bound zero-integration compilation of the actual task overlay."""
import argparse
import json
from pathlib import Path
import numpy as np
from .task_capture_runtime import cpu_stage,sha,read
from .task_plant import xml_scene,check_frozen_robot,PARAMETERS,VERSION


def run(root,asset_root,reference,stage,render=False):
    import mujoco
    from scripts.pi0292.durable import write,OwnerLock
    with cpu_stage(root,stage,cores=1,cpu_s_cap=900):
        root=Path(root);out=root/'runtime'/stage;out.mkdir(exist_ok=False)
        card=dict(id='TASK-PILOT-W0-T1',task='T1',split='PILOT',source_xy=[.45,-.08],
            goal_xy=[.45,.08],cube_yaw=0.,group_id='TASK-PILOT-W0',scene_class='W0_FREE_STANDING_NEAR',
            instruction='Pick up the red cube, place it in the blue target, release and withdraw.')
        text=xml_scene(card,asset_root);(out/'task-scene.private.xml').write_text(text)
        original=mujoco.MjModel.from_binary_path(str(reference))
        model=mujoco.MjModel.from_xml_string(text)
        frozen=check_frozen_robot(original,model)
        for field in ('timestep','integrator','solver','cone','impratio','tolerance','iterations','gravity'):
            np.testing.assert_array_equal(getattr(original.opt,field),getattr(model.opt,field))
        data=mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home'))
        # Exact already-adopted native PI0289ResetStartup semantics; no IK/pose search.
        for name in ('x5_joint2','x5_joint3'):
            joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
            data.qpos[model.jnt_qposadr[joint]]=.9
        mujoco.mj_forward(model,data)
        joints=[]
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                q=float(data.qpos[model.jnt_qposadr[j]])
                lo,hi=map(float,model.jnt_range[j])
                if not lo<=q<=hi:raise ValueError('INITIAL_RAW_JOINT_LIMIT')
                joints.append(dict(name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j),q=q,lower_margin=q-lo,upper_margin=hi-q))
        cameras={}
        for name,body in (('body_rgb','base_link'),('wrist_rgb','x5_link6')):
            c=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,name)
            b=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body)
            if model.cam_bodyid[c]!=b:raise ValueError('ACTUAL_MOVING_CAMERA_REQUIRED')
            cameras[name]=dict(mounted_body=body,position_world=data.cam_xpos[c].tolist(),rotation_world=data.cam_xmat[c].reshape(3,3).tolist())
        images={}
        if render:
            from PIL import Image
            renderer=mujoco.Renderer(model,height=256,width=256)
            try:
                for name in cameras:
                    renderer.update_scene(data,camera=name);rgb=renderer.render().copy()
                    Image.fromarray(rgb).save(out/(name+'.png'))
                    red=(rgb[:,:,0]>80)&(rgb[:,:,0].astype(float)>1.5*rgb[:,:,1])&(rgb[:,:,0].astype(float)>1.5*rgb[:,:,2])
                    images[name]=dict(shape=list(rgb.shape),dtype=str(rgb.dtype),standard_deviation=float(rgb.std()),red_pixels=int(red.sum()),sha256=sha(out/(name+'.png')))
            finally:renderer.close()
            # Two actual renders comprise one pair, even if visibility fails.
            parent=Path(read(root/'OUTPUT_BINDING.private.json')['parent_root'])
            owner=OwnerLock(parent/'R4_RESOURCE_OWNER.lock');ledger=read(parent/'R4_RESOURCE_LEDGER.json')
            if ledger['RGB_pairs']+1>102560:raise PermissionError('CUMULATIVE_RGB_CAP')
            ledger['RGB_pairs']+=1;write(parent/'R4_RESOURCE_LEDGER.json',ledger);owner.close()
        path=out/'task-scene.private.mjb';mujoco.mj_saveModel(model,str(path),None)
        tcp=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'x5_ee')
        receipt=dict(status='COMPILED_FULLBODY_TASK_OVERLAY_ZERO_PHYSICS',version=VERSION,
            model_sha256=sha(path),source_reference_model_sha256=sha(reference),frozen_robot=frozen,
            nq=model.nq,nv=model.nv,nu=model.nu,parameters=PARAMETERS,card=card,joint_margins=joints,
            cameras=cameras,initial_images=images,initial_TCP_xyz=data.site_xpos[tcp].tolist(),
            initial_TCP_R=data.site_xmat[tcp].reshape(3,3).tolist(),initial_contacts=[dict(
                geom1=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),geom2=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),distance=float(c.dist)) for c in data.contact],
            time_s=data.time,physics_steps=0,new_model_queries=0,physical_task_qualification=False)
        write(out/'CPU_RECEIPT.json',receipt)
        print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','asset-root','reference','stage'):p.add_argument('--'+name,required=True)
    p.add_argument('--render',action='store_true');a=p.parse_args()
    run(a.root,a.asset_root,a.reference,a.stage,a.render)
