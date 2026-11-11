"""Recorded link6/cube pair sensitivity; isolated geometry, no robot rollout.

The diagnostic model can be a different scene only after the rigid TCP-to-geom
transform reproduces the recorded world center. Full robot/IK/reference safety
is still the original native certificate, never this two-geom reconstruction.
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .storage import digest,write_json


def pose(position,quaternion,mj):
    rotation=np.empty(9);mj.mju_quat2Mat(rotation,np.asarray(quaternion,float))
    result=np.eye(4);result[:3,:3]=rotation.reshape(3,3);result[:3,3]=position
    return result


def run(spec,output):
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    started=time.monotonic();cpu=time.process_time()
    model_file=Path(spec['rigid_geometry_model']['path']);before=digest(model_file)
    if before!=spec['rigid_geometry_model']['sha256']or mj.__version__!='3.2.7':raise ValueError('Pinned geometry and MuJoCo3.2.7 required')
    model=mj.MjModel.from_binary_path(str(model_file));record=spec['recorded_pair'];gid,cube_id=record['geom_ids'];site=model.site('x5_ee').id
    if not(int(model.geom_bodyid[gid])==int(model.site_bodyid[site])and int(model.geom_type[gid])==3
        and int(model.geom_type[cube_id])==6 and np.array_equal(model.geom_size[gid],record['link6_size'])
        and np.array_equal(model.geom_size[cube_id],record['cube_size'])):raise ValueError('Same rigid native capsule/box properties required')
    target=np.asarray(record['planned_TCP'],float)
    geom=target@np.linalg.inv(pose(model.site_pos[site],model.site_quat[site],mj))@pose(model.geom_pos[gid],model.geom_quat[gid],mj)
    center_error=float(np.linalg.norm(geom[:3,3]-np.asarray(record['recorded_link6_world_center'])))
    if center_error>1e-9:raise ValueError('Diagnostic model does not reproduce the recorded rigid geometry')
    original_cube_rotation=np.asarray(record['cube_rotation_assumption'],float)
    rows=[];loads=1;forwards=0;distance_calls=0
    cases=[(0.,(0.,0.,0.))]
    # Candidate shift is fixed before the robustness check. The baseline is not
    # a second native qualification or an executed collision.
    angle=float(spec['cube_orientation_probe_rad'])
    cases.extend((spec['candidate_TCP_world_z_shift_m'],v)for v in itertools.product((-angle,0.,angle),repeat=3))
    for dz,rotation_vector in cases:
        q=np.empty(4);mj.mju_mat2Quat(q,geom[:3,:3].reshape(9))
        rotation=Rotation.from_rotvec(rotation_vector).as_matrix()@original_cube_rotation
        cube_q=np.empty(4);mj.mju_mat2Quat(cube_q,rotation.reshape(9))
        join=lambda value:' '.join(format(float(v),'.17g')for v in value)
        xml=('<mujoco><option timestep="0.002"/><worldbody><geom name="recorded_link6" type="capsule" size="'+join(model.geom_size[gid,:2])+
             '" pos="'+join(geom[:3,3]+[0.,0.,dz])+'" quat="'+join(q)+'"/><geom name="recorded_cube" type="box" size="'+join(model.geom_size[cube_id])+
             '" pos="'+join(record['cube_world_center'])+'" quat="'+join(cube_q)+'"/></worldbody></mujoco>')
        pair=mj.MjModel.from_xml_string(xml);loads+=1;data=mj.MjData(pair);mj.mj_forward(pair,data);forwards+=1
        endpoints=np.empty(6);distance=float(mj.mj_geomDistance(pair,data,0,1,1.,endpoints));distance_calls+=1
        rows.append(dict(TCP_world_z_shift_m=dz,cube_rotation_probe_vector_rad=list(rotation_vector),signed_distance_m=distance))
    error=abs(rows[0]['signed_distance_m']-record['recorded_signed_distance_m'])
    if error>1e-9:raise ValueError('Isolated baseline does not reproduce the saved signed distance')
    if digest(model_file)!=before:raise ValueError('Original model changed')
    result=dict(schema='TEACHER_V8_RECORDED_PAIR_VERTICAL_CLEARANCE_SENSITIVITY_V1',source_sha=spec['source_sha'],
        actual_site='LOCAL_PC_CPU_OFFLINE_GEOMETRY',rigid_geometry_model_SHA256=before,
        recorded_native_model_SHA256=record['recorded_native_model_SHA256'],different_scene_model_used_only_for_rigid_geometry=True,
        recorded_state_SHA256=record['recorded_state_SHA256'],recorded_geometry_proof_SHA256=record['geometry_proof_SHA256'],
        rigid_world_center_error_m=center_error,baseline_signed_distance_error_m=error,
        cube_rotation_is_explicit_assumption_validated_only_by_saved_pair_scalar=True,
        candidate_TCP_world_z_shift_m=spec['candidate_TCP_world_z_shift_m'],candidate_grasp_cube_z_offset_m=.012+spec['candidate_TCP_world_z_shift_m'],
        minimum_candidate_pair_clearance_m=min(r['signed_distance_m']for r in rows[1:]),
        cube_orientation_probe_rad=angle,rows=rows,requested_pair_cases=len(cases),completed_pair_cases=len(rows),
        MuJoCo_model_loads=loads,cloned_forward_calls=forwards,geometry_distance_calls=distance_calls,
        physics_integrations=0,GR00T_model_loads=0,GR00T_queries=0,optimizer_updates=0,
        original_model_unchanged=True,original_collision_threshold_unchanged=True,whole_robot_collision_certificate=False,
        native_reference_IK_checked=False,Teacher_qualification_requests=0,wall_s=time.monotonic()-started,CPU_s=time.process_time()-cpu,
        meaning='One saved veto pair and isolated perturbations only. Candidate must still pass the original full native reference and actual task qualification.')
    write_json(output,result);return result
