"""Compile an explicit new ARM-home initial condition, with zero integration.

The original native ResetHome consumes this keyframe. This is not a reached
pose, complete controller restore, C02 repair, or dynamic qualification.
"""
import argparse,ctypes,json,time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def replace_arm_home(original,indices,values,bounds):
    q=np.asarray(original,float);i=np.asarray(indices);v=np.asarray(values,float);b=np.asarray(bounds,float)
    if(q.shape!=(34,)or i.shape!=(6,)or i.dtype.kind not in 'iu'or len(set(i.tolist()))!=6
       or np.any(i<7)or np.any(i>=27)or v.shape!=(6,)or b.shape!=(6,2)
       or not np.isfinite(v).all()or np.any(v<b[:,0]+.03)or np.any(v>b[:,1]-.03)):
        raise ValueError('EXACT_SIX_ARM_INITIAL_JOINTS_WITH_RAW_LIMIT_MARGIN')
    out=q.copy();out[i]=v;return out


def run(spec,output):
    import mujoco
    from go2arx5_vla.world_model.storage import digest,write_json
    from go2arx5_vla.world_model.pilot1_curriculum import robot_contract
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0294.task_contacts import allowed
    start,cpu=time.monotonic(),time.process_time();output=Path(output)
    for k in('model','xml','static_proof','reset_library'):
        if digest(spec[k])!=spec[k+'_sha256']:raise ValueError('PINNED_PREPOSE_INPUT_'+k)
    proof=json.loads(Path(spec['static_proof']).read_bytes())
    if(proof.get('schema')!='WM02_C03_NOMINAL_SOURCE_POSE_PREPOSE_STATIC_V1'or proof.get('static_passed')is not True
       or proof.get('command_compensation_applied')is not False or proof.get('new_integrations')!=0):raise ValueError('ACTUAL_NOMINAL_STATIC_POSE_NOT_COMPENSATION')
    original=mujoco.MjModel.from_binary_path(spec['model']);key=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_KEY,'home')
    joints=[mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_JOINT,'x5_joint'+str(k))for k in range(1,7)]
    if key<0 or min(joints)<0 or mujoco.__version__!='3.2.7':raise ValueError('ORIGINAL_HOME_AND_NATIVE_MUJOCO')
    indices=np.asarray(original.jnt_qposadr[joints]);home=replace_arm_home(original.key_qpos[key],indices,proof['arm_q'],original.jnt_range[joints])
    xml=ET.fromstring(Path(spec['xml']).read_bytes());node=xml.find("./keyframe/key[@name='home']")
    if node is None or not np.array_equal(np.fromstring(node.get('qpos'),sep=' '),original.key_qpos[key]):raise ValueError('EXACT_ORIGINAL_XML_HOME')
    node.set('qpos',' '.join(format(float(x),'.17g')for x in home));candidate=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    new_keys=candidate.key_qpos.copy();expected=original.key_qpos.copy();expected[key]=home
    if not np.array_equal(new_keys,expected):raise ValueError('ONLY_DECLARED_HOME_ARM_JOINTS_CHANGED')
    # Audit all original dynamics and other initial fields; the sole declared
    # exception is recorded separately and restored before saving the model.
    candidate.key_qpos[:]=original.key_qpos
    try:baseline_contract=robot_contract(original,candidate)
    finally:candidate.key_qpos[:]=new_keys
    output.mkdir(exist_ok=False);(output/'SCENE.private.xml').write_text(ET.tostring(xml,encoding='unicode'));mujoco.mj_saveModel(candidate,str(output/'SCENE.private.mjb'),None)
    model=mujoco.MjModel.from_binary_path(str(output/'SCENE.private.mjb'));data=mujoco.MjData(model)
    library=ctypes.CDLL(spec['reset_library']);reset=library.WM02ResetAudit;reset.argtypes=[ctypes.c_void_p,ctypes.c_void_p];reset.restype=ctypes.c_int
    with forbid_integration(mujoco)as calls:
        if reset(model._address,data._address)!=0 or not np.array_equal(data.qpos,home):raise ValueError('ORIGINAL_NATIVE_RESETHOME_MUST_CONSUME_NEW_KEY_EXACTLY')
        unsafe=[]
        for c in data.contact:
            names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(i))for i in(c.geom1,c.geom2)]
            if not allowed(*names):unsafe.append(dict(geoms=names,distance_m=float(c.dist)))
        if calls or unsafe:raise ValueError('PREPOSE_INITIAL_SAFETY_OR_FORBIDDEN_INTEGRATION')
    result=dict(schema='WM02_C03_PREPOSED_HOME_STATIC_V1',status='PREPOSE_HOME_STATIC_PASSED_NOT_NATIVE_QUALIFIED',
        scene_sha256=digest(output/'SCENE.private.mjb'),scene_xml_sha256=digest(output/'SCENE.private.xml'),
        parent_scene_sha256=spec['model_sha256'],static_proof_sha256=spec['static_proof_sha256'],source_sha=spec['source_sha'],
        original_dynamics_contract=baseline_contract,original_C02_exact_home_contract_is_not_claimed=True,
        declared_change='ONLY_SIX_ARM_HOME_KEYFRAME_JOINT_POSITIONS_NEW_INITIAL_CONDITION',home_key_id=key,
        changed_qpos_indices=indices.tolist(),original_home_arm_q=original.key_qpos[key,indices].tolist(),new_home_arm_q=home[indices].tolist(),
        original_native_ResetHome_exact=True,reset_is_not_midpoint_controller_restore=True,controlled_reaching_demonstrated=False,
        WBC_changes=False,safety_changes=False,physics_dt_s=.002,new_integrations=0,new_queries=0,new_training_labels=0,
        wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)
    write_json(output/'RESULT.private.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    print(json.dumps(run(json.loads(a.spec.read_bytes()),a.output)));return 0
if __name__=='__main__':raise SystemExit(main())
