"""Rebuild frozen Teacher decisions and probe paired observed pose errors without physics."""

from dataclasses import replace
from pathlib import Path
import argparse
import json

import numpy as np

from go2arx5_vla.trajectory.so3 import quat_to_matrix
from go2arx5_vla.world_model.fusion_trajectory_adapter import MECHANICAL_TRAVEL_M
from go2arx5_vla.world_model.mobile_continuous_teacher_v2 import rotation_increment, rotation_vector
from go2arx5_vla.world_model.multitask_placement_geometry_d2 import POSITION_STEP_M, ANGLE_STEP_RAD
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, read_state
from go2arx5_vla.world_model.multitask_teacher_d2 import PlacementTeacher, PlacementUnavailable
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.eval_contract import matrix10


def read(path):
    return json.loads(Path(path).read_bytes())


def reference_pose(row):
    xyz, xyzw = row['accepted_pose7'][:3], row['accepted_pose7'][3:]
    if not row['exact_receiver_knot']:
        raise ValueError('EXACT_ARCHIVED_RECEIVER_BOUNDARY_REQUIRED')
    pose = np.eye(4)
    pose[:3,3], pose[:3,:3] = xyz, quat_to_matrix(np.asarray(xyzw))
    return pose


def measured(row):
    sample = dict(row)
    for key in ('tcp','base_pose','object_pose','qpos','qvel','ctrl','finger_q'):
        sample[key] = np.asarray(sample[key])
    return sample


def paired_error(before, observed, target_pose):
    root0, root1 = before['qpos'], observed['qpos']
    dq0, dq1 = quat_to_matrix(root0[3:7][[1,2,3,0]]), quat_to_matrix(root1[3:7][[1,2,3,0]])
    actual_in_root = dq1.T@(observed['tcp'][:3,3]-root1[:3])
    expected_in_root = dq0.T@(target_pose[:3,3]-root0[:3])
    actual_rotation_in_root = dq1.T@observed['tcp'][:3,:3]
    expected_rotation_in_root = dq0.T@target_pose[:3,:3]
    return dict(tick=observed['native_step'],
        translation=observed['tcp'][:3,3]-target_pose[:3,3],
        rotation=rotation_vector(observed['tcp'][:3,:3]@target_pose[:3,:3].T),
        root_shift=root1[:2]-root0[:2], root_rise_m=float(root1[2]-root0[2]),
        root_angle_rad=float(np.linalg.norm(rotation_vector(dq1@dq0.T))),
        relative_translation_m=float(np.linalg.norm(actual_in_root-expected_in_root)),
        relative_rotation_rad=float(np.linalg.norm(rotation_vector(
            actual_rotation_in_root@expected_rotation_in_root.T))))


def perturbed_prefix(knots, hypothesis, attached):
    """Offline translation and SO(3) sensitivity with matched planar root motion."""
    attachment = np.linalg.solve(knots[0].tcp, knots[0].object_pose)
    shifted = []
    for k, point in enumerate(knots):
        # The explicit measured anchor and decoded knot 0 both occur at t=0.
        weight = max(0,k-1)/(len(knots)-2)
        tcp = point.tcp.copy()
        tcp[:3,3] += weight*hypothesis['translation']
        tcp[:3,:3] = rotation_increment(weight*hypothesis['rotation'])@tcp[:3,:3]
        shifted.append(replace(point,tcp=tcp, object_pose=tcp@attachment if attached else point.object_pose,
                               root_xy=point.root_xy+weight*hypothesis['root_shift']))
    return shifted


def opening_probes(geometry, actual, intended, program, color, margin):
    """Test the jaw-space projection while retaining the rejected full margin."""
    fingers = actual['finger_q']
    native_setpoints = program.desired_path[:11,9]*MECHANICAL_TRAVEL_M
    supported = actual['objects'][color]['table_contact']
    increasing = bool((np.diff(native_setpoints)>=-1e-12).all())
    if not increasing or not supported:
        return dict(supported=supported, opening_monotone=increasing, checked=False)
    calibrated = [intended[0]]+[replace(point,finger_q=np.full(2,width))
                                for point,width in zip(intended[1:],native_setpoints,strict=True)]
    opening = [intended[0]]+[replace(point,finger_q=np.maximum(fingers,width))
                             for point,width in zip(intended[1:],native_setpoints,strict=True)]
    return dict(supported=True, opening_monotone=True, checked=True, actual_joint_openings_m=fingers.tolist(),
                old_shadow_first_joint_setpoint_m=intended[1].finger_q.tolist(),
                first_wire_working_setpoint_m=float(native_setpoints[0]),
                last_wire_working_setpoint_m=float(native_setpoints[-1]),
                full_original_margin_retained_m=margin,
                calibrated_absolute=geometry.check(actual,calibrated,margin),
                supported_start=geometry.check(actual,opening,margin))


def clearance_by_pair_role(geometry, actual, intended, errors, sample_margin, radius, full_margin):
    """Keep the full world margin on pairs with any external obstacle."""
    co_moving_margin = sample_margin+max(e['relative_translation_m']+radius*e['relative_rotation_rad']
                                         for e in errors)
    all_pairs = geometry.pairs
    arm_target = [(a,b) for a,b in all_pairs if a in geometry.arm and b==geometry.target]
    try:
        geometry.pairs = [pair for pair in all_pairs if pair not in arm_target]
        world_verdict = geometry.check(actual,intended,full_margin)
        geometry.pairs = arm_target
        relative_verdict = geometry.check(actual,intended,co_moving_margin)
    finally:
        geometry.pairs = all_pairs
    return dict(world_obstacle_pairs=world_verdict, grasp_attached_robot_pairs=relative_verdict,
                original_native_contact_predicates_on_all_pairs=True)


def lift_to_carry(observations):
    """Check the archived boundary the Teacher used to begin carrying."""
    route = observations[0]['d2_placement_corridor_audit_only'].get('d2_full_route')
    lift = next((row for row in observations if row['phase'] == 'LIFT'), None)
    carry = next((row for row in observations if row['phase'] == 'CARRY'), None)
    if route is None or lift is None or carry is None:
        return None
    accepted = next(item for item in route['attempts'] if item['passed'])
    before, after = (measured(row['actual_state']) for row in (lift, carry))
    displacement = after['tcp'][:3, 3] - before['tcp'][:3, 3]
    nominal = np.asarray(carry['nominal_world_H40'])
    return dict(lift_tick=before['native_step'], carry_tick=after['native_step'],
                planned_rise_m=accepted['transit_raise_m'], measured_rise_m=float(displacement[2]),
                measured_horizontal_drift_m=float(np.linalg.norm(displacement[:2])),
                first_carry_nominal_P10_displacement_m=(nominal[10, :3]-nominal[0, :3]).tolist())


def verify_case(case, expected_sha):
    import mujoco

    state = read(case/'control/LEASE.json')
    if state['status']!='CLOSED' or digest(case/'RESULT.private.json')!=expected_sha:
        raise PermissionError('EXACT_COLD_PLACEMENT_RESULT_REQUIRED')
    spec, result = (read(case/name) for name in ('EXECUTION_SPEC.json','RESULT.private.json'))
    native = case/'native-01'
    detail = result['detail']
    failure = detail['teacher_placement_failure']
    if failure['reason']!='D2_FILTERED_NEXT_P10_CLEARANCE':
        raise ValueError('EXACT_ARCHIVED_D2_PREFIX_REJECTION_REQUIRED')
    pinned = read(native/'LIVE_DATA/MANIFEST.private.json')['labels']
    if digest(pinned['path'])!=pinned['sha256'] or digest(spec['model_path'])!=spec['frozen_files'][spec['model_path']]:
        raise ValueError('ARCHIVED_D2_LABEL_OR_SCENE_CHANGED')
    observations = read(pinned['path'])
    model = mujoco.MjModel.from_binary_path(spec['model_path'])
    initial = measured(observations[0]['actual_state'])
    teacher_type = PlacementTeacher
    if spec.get('d2_teacher_candidate') is not None:
        from go2arx5_vla.world_model.multitask_scope_d2 import STAGED_TEACHER
        from go2arx5_vla.world_model.multitask_staged_placement_d2 import StagedPlacementTeacher

        if spec['d2_teacher_candidate'] != STAGED_TEACHER:
            raise ValueError('EXACT_FROZEN_D2_REPLAY_TEACHER_REQUIRED')
        teacher_type = StagedPlacementTeacher
    teacher = teacher_type(initial, spec['mobile_condition'], spec['multitask_geometry'],
                           spec['task_instruction_id'], model)
    geometry = teacher.placement_geometry
    base = initial['qpos'][[0,1,2,4,5,6,3]]
    peak_error = 0.
    with forbid_integration(mujoco) as attempted:
        for row in observations:
            sample = measured(row['actual_state'])
            if row['recent_reference']['native_step']!=sample['native_step']:
                raise ValueError('SHARED_MEASURED_AND_ACCEPTED_QUERY_CLOCK_REQUIRED')
            program, _ = teacher.decide(sample,reference_pose(row['recent_reference']),base)
            delta = max(np.abs(program.decoded_H40-row['nominal_world_H40']).max(),
                        np.abs(program.reference_path-row['installed_reference_H40']).max())
            peak_error = max(peak_error,float(delta))
            if teacher.stage!=row['phase'] or delta>1e-9 or row['consumed_mask']!=[True]*10+[False]*30:
                raise ValueError('FROZEN_TEACHER_H40_OR_CONSUMPTION_MISMATCH')
            teacher.commit(program,sample['native_step']+250)
        actual = TwoObjectState(model,teacher.task.color).update(read_state(native/'state.bin'))
        try:
            teacher.decide(actual,reference_pose(observations[-1]['next_query_boundary']),base)
        except PlacementUnavailable as err:
            if str(err)!=failure['reason'] or err.evidence['sample_count']!=failure['geometry_evidence']['sample_count']:
                raise
            rejected = err.evidence
        else:
            raise ValueError('FROZEN_PREFIX_REJECTION_NOT_REPRODUCED')
        if not np.isclose(rejected['margin_m'],failure['geometry_evidence']['margin_m'],atol=1e-10):
            raise ValueError('FROZEN_PREFIX_MARGIN_NOT_REPRODUCED')
        intended = teacher.planned_prefix(actual,teacher.pending[0])
        attached = actual['native_grasp'] and teacher.stage in ('LIFT','HOLD','CARRY','LOWER')
        radius = np.linalg.norm(actual['tcp'][:3,3]-actual['object_pose'][:3,3])+.025
        sample_margin = POSITION_STEP_M/2+radius*ANGLE_STEP_RAD/2
        jaw = (opening_probes(geometry,actual,intended,teacher.pending[0],teacher.task.color,rejected['margin_m'])
               if teacher.stage=='OPEN' else None)
        checks = [dict(name='observed_anchor_no_future_perturbation',varied=False,
                       verdict=geometry.check(actual,intended,sample_margin))]
        errors = []
        for i in range(len(observations)-4,len(observations)):
            before = measured(observations[i]['actual_state'])
            after = actual if i==len(observations)-1 else measured(observations[i+1]['actual_state'])
            error = paired_error(before,after,matrix10(np.asarray(observations[i]['nominal_world_H40'][10])))
            errors.append(error)
            paths = perturbed_prefix(intended,error,attached)
            checks.append(dict(name='paired_TCP_and_root_XY',varied=True,
                paired_observed_error={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in error.items()},
                verdict=geometry.check(actual,paths,sample_margin)))
        pair_roles = (clearance_by_pair_role(geometry,actual,intended,errors,sample_margin,radius,rejected['margin_m'])
                      if attached else None)
    if attempted:
        raise ValueError('D2_SAME_CLOCK_REPLAY_MUST_NOT_INTEGRATE')
    return dict(case=case.name,source_sha=spec['source_sha'],result_sha256=expected_sha,
        teacher_candidate=spec.get('d2_teacher_candidate'),
        exact_query_count=len(observations),max_archived_H40_reproduction_error=peak_error,
        measured_lift_to_carry=lift_to_carry(observations),
        observed_terminal_tick=actual['native_step'],original_replayed_rejection=rejected,
        fixed_vertical_and_angular_root_state=True,
        existing_spatial_sampling_margin_m=sample_margin,sensitivity_checks=checks,
        supported_opening_projection=jaw, clearance_by_pair_role=pair_roles,
        selected_controller_reference='UNKNOWN',new_native_requests=0,new_physics_ticks=0,
        modified_live_source=False,admission_claim=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case',type=Path)
    parser.add_argument('result_sha256')
    args = parser.parse_args()
    print(json.dumps(verify_case(args.case,args.result_sha256),allow_nan=False))


if __name__=='__main__':
    main()
