"""Independently spawned live worker. Shared accounting belongs to coordinator."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback
import numpy as np
from PIL import Image
from scripts.pi0291_arx5_data.config import PARAMETERS as P
from go2arx5_vla.task_data.codec import pose10
from go2arx5_vla.task_data.io import file_hash, jsonable, manifest, write
from go2arx5_vla.trajectory.so3 import matrix_to_quat
from .scene_cards import digest
from .scene import make_engine, qualify_engine
from .score import BOOL_FIELDS, saved_score
from .teacher import make_teacher


def validate_reservation(reservation, card, output_dir, source_sha):
    r = reservation
    if r.get('schema') != 'PI0294_LIVE_RESERVATION_V1' or r.get('fixture') is not False:
        raise ValueError('Live coordinator reservation required')
    if r['id'] != card['id'] or r['card_sha256'] != digest(card) or r['card'] != card:
        raise ValueError('Card reservation mismatch')
    if r['source_sha'] != source_sha or len(source_sha) != 40:
        raise ValueError('Source reservation mismatch')
    output = Path(output_dir).resolve()
    root = Path(r['root']).resolve()
    if output != Path(r['output']).resolve() or output.parent != root/'episodes':
        raise ValueError('Worker output outside exact reserved episode')
    if r['deadline_epoch'] <= time.time():
        raise RuntimeError('Worker deadline already reached')
    for field, minimum in (('physics_starts', 1), ('physics_steps', 16000), ('rgb_pairs', 641)):
        if r['reservation'][field] != minimum:
            raise ValueError('Worker requires one full 32s episode reservation')
    if not 1 <= len(r['cpus']) <= 3 or len(set(r['cpus'])) != len(r['cpus']):
        raise ValueError('Invalid worker CPU assignment')
    if r['reservation']['output_bytes'] < 512*1024**2:
        raise ValueError('Insufficient raw evidence space reservation')
    return root, output


def _robot_contact(engine, state):
    import mujoco
    cube = engine.geoms['cube_geom']
    for contact in state['contacts']:
        geoms = [contact['geom0'], contact['geom1']]
        if cube not in geoms or contact['normal_force_N'] < P['minimum_contact_force_N']:
            continue
        other = geoms[1] if geoms[0] == cube else geoms[0]
        body = int(engine.model.geom_bodyid[other])
        name = mujoco.mj_id2name(engine.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if name and name.startswith('x5_'):
            return True
    return False


def run_episode(card, output_dir, asset_root, source_sha, reservation, progress=None):
    root, folder = validate_reservation(reservation, card, output_dir, source_sha)
    os.sched_setaffinity(0, set(reservation['cpus']))
    if os.environ.get('LIBGL_ALWAYS_SOFTWARE') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Worker requires reserved Mesa CPU renderer')
    if os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES') != '/usr/share/glvnd/egl_vendor.d/50_mesa.json':
        raise RuntimeError('Mesa vendor binding missing')
    if 'MUJOCO_EGL_DEVICE_ID' in os.environ:
        raise RuntimeError('Unexpected GPU EGL binding')
    folder.mkdir(parents=False, exist_ok=False)
    write(folder/'CARD.json', card)
    write(folder/'RESERVATION.json', reservation)
    began, cpu_began = time.monotonic(), time.process_time()
    engine, teacher = None, None
    samples, physical, refs, highrate, contact_rows, contact_times, external_force_max = [], [], [], [], [], [], []
    status, error, starts = 'FAILED_PRESERVED', None, 0
    rgb_pairs = 0
    plan = ref_id = None
    static_qualification = None

    def guard():
        if time.time() >= reservation['deadline_epoch']:
            raise RuntimeError('RESERVED_DEADLINE')
        if time.process_time()-cpu_began > reservation['reservation']['cpu_s']:
            raise RuntimeError('RESERVED_CPU_SECONDS')
        memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if memory > reservation['reservation']['memory_bytes']:
            raise RuntimeError('RESERVED_MEMORY_BYTES')

    def record_tick():
        state = engine.snapshot()
        contact_times.append(state['time_s'])
        contact_rows.append([float(_robot_contact(engine, state)), float(state['native_grasp']),
                             float(np.linalg.norm(state['object_pose'][:2, 3]-initial[:2, 3])),
                             float(state['object_pose'][2, 3]), float(not state['safety_valid'])])
        highrate.append(np.r_[engine.data.time, engine.data.qpos, engine.data.qvel, engine.data.ctrl,
                             engine.data.xfrc_applied[engine.ids['tool']],
                             float(not state['safety_valid'])])
        external_force_max.append([float(np.max(np.abs(engine.data.xfrc_applied))),
                                   float(np.max(np.abs(engine.data.qfrc_applied)))])

    try:
        guard()
        starts = 1
        engine = make_engine(card, asset_root)
        first = engine.snapshot()
        initial = first['object_pose'].copy()
        teacher = make_teacher(card, initial, engine)
        if 'relay_request_relative' in card:
            from .relay_teacher import RelayTeacher
            teacher = RelayTeacher(teacher, root, folder, card)
        write(folder/'INITIAL_STATE.json', dict(qpos=first['qpos'], qvel=first['qvel'],
                                                object_pose=initial, tcp=first['tcp'],
                                                model_xml_sha256=hashlib.sha256(engine.xml.encode()).hexdigest(),
                                                paired_scene_sha256=card['initial_state_sha256']))
        if hasattr(teacher, 'geometry'):
            write(folder/'PUSH_GEOMETRY.json', teacher.geometry)
        for step in range(641):
            guard()
            state = engine.snapshot()
            state['robot_object_contact'] = _robot_contact(engine, state)
            if abs(state['time_s']-step/20) > 1e-8 or not state['safety_valid']:
                raise RuntimeError('REAL_CLOCK_OR_SAFETY_FAIL')
            images, geometry = engine.render(state)
            rgb_pairs += 1
            if step == 0:
                from OpenGL import GL
                vendor = GL.glGetString(GL.GL_VENDOR).decode()
                renderer = GL.glGetString(GL.GL_RENDERER).decode()
                if vendor != 'Mesa' or 'llvmpipe' not in renderer:
                    raise RuntimeError('Unexpected live rendering backend')
                write(folder/'RENDER_PROCESS.json', dict(pid=os.getpid(), backend='mesa_cpu', GL_VENDOR=vendor,
                                                          GL_RENDERER=renderer, observed_own_rows=[], requested_EGL_device=None))
            snapshot = f"{card['id']}:{engine.steps}"
            obs = dict(instruction=card['instruction'], actual_tcp=state['tcp'], actual_gripper=state['gripper_fraction'],
                       actual_gripper_speed=state['gripper_speed'], timestamp_s=state['time_s'], valid=True)
            for camera, image in images.items():
                if image.shape != (256, 256, 3):
                    raise RuntimeError('RGB_SHAPE')
                path = folder/camera/f'{step:04d}.png'
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(image).save(path)
                obs[camera] = dict(path=path.relative_to(folder).as_posix(), sha256=file_hash(path), frame_index=step,
                                   timestamp_s=state['time_s'], snapshot_id=snapshot, source='actual_mujoco_renderer', valid=True)
            if step == 0:
                static_qualification = qualify_engine(engine, rendered=(images, geometry), guard=guard)
                write(folder/'STATIC_QUALIFICATION.json', static_qualification)
            static_failed = static_qualification['status'] != 'STATIC_PASS_PENDING_LIVE_CONTACT'
            teacher.observe(state)
            supervisor = None
            if step < 640 and not static_failed:
                if step % 8 == 0:
                    plan = teacher.plan(state)
                    ref_id = plan.get('reference_uuid', f"{card['id']}:ref{step//8:03d}")
                    refs.append(dict(ref_id=ref_id, decision_index=step//8, decision_time_s=state['time_s'],
                                     observation_snapshot=snapshot, prior_committed_count=step, anchor_actual_tcp=state['tcp'],
                                     planned_H40=plan['poses'], gripper_H40=plan['opening'], offsets_s=plan['offset_s'],
                                     rows10=np.stack([pose10(a, b) for a, b in zip(plan['poses'], plan['opening'])]),
                                     raw_quaternion_xyzw=np.stack([matrix_to_quat(a[:3, :3]) for a in plan['poses']]),
                                     privileged_teacher_stage=plan['stage'], source=plan.get('source','ACTUAL_FEEDBACK_TEACHER'), committed_prefix_count=0))
                k = step % 8
                target, opening = plan['poses'][k], float(plan['opening'][k])
                desired, desired_gripper = target.copy(), opening
                if hasattr(teacher, 'reference_action'):
                    # Preserve original raw H40; resolve mapped action before labels.
                    target, opening = teacher.reference_action(k)
                supervisor = dict(ref_id=ref_id, sequence=step, point_index=k, command_time_s=state['time_s'],
                                  desired=desired, accepted=target.copy(), installed=target.copy(), committed=target.copy(),
                                  desired_gripper=desired_gripper, committed_gripper=opening, valid=True, actually_committed=True)
                refs[-1]['committed_prefix_count'] += 1
            physical.append({k: state[k] for k in ('time_s', 'object_pose', 'tcp', 'object_linear_velocity',
                                                   'object_angular_velocity', 'robot_object_contact', *BOOL_FIELDS)})
            samples.append(dict(snapshot_id=snapshot, policy_obs=obs,
                causal_history=dict(episode_id=card['id'], segment_id=0, previous_samples=step, previous_committed=step,
                                    reset=False, abort=False, takeover=False), supervision=supervisor,
                execution=dict(actual_tcp=state['tcp'], actual_finger_q=state['finger_q'], actual_finger_qd=state['finger_qd'],
                               actual_gripper_gap_m=state['gripper_gap_m'], actuator_force=state['actuator_force'],
                               actual_joint_q=state['arm_q'], actual_joint_qd=state['arm_qd'],
                               previous_finger_targets=engine.finger_targets.copy(),
                               reference_pose_error=None if supervisor is None else float(np.linalg.norm(target[:3, 3]-state['tcp'][:3, 3]))),
                privileged=dict(object_pose=state['object_pose'], object_linear_velocity=state['object_linear_velocity'],
                                object_angular_velocity=state['object_angular_velocity'], current_grasped=state['native_grasp'],
                                ever_lifted=teacher.lifted, ever_carried=teacher.carried, teacher_stage=teacher.stage,
                                contacts=state['contacts'], opposing_contact=state['opposing_contact'],
                                relative_motion_valid=state['relative_motion_valid'], object_points=state['object_points']), geometry=geometry))
            if progress:
                progress(dict(id=card['id'], physics_steps=engine.steps, rgb_pairs=rgb_pairs))
            if static_failed:
                raise RuntimeError('STATIC_QUALIFICATION_FAILED_NO_INTEGRATION')
            if step < 640:
                for tick in range(25):
                    if hasattr(teacher, 'before_native_tick'):
                        # Existing outer loop is the sole live Engine.step owner.
                        teacher.before_native_tick(engine, target, opening)
                        native_error = None
                        try:
                            engine.step(target, opening)
                        except Exception as caught:
                            native_error = caught
                        try:
                            record_tick()
                        except Exception as record_error:
                            teacher.after_native_tick(engine, native_error, record_error)
                            raise
                        teacher.after_native_tick(engine, native_error, None)
                        if native_error is not None:
                            raise native_error
                    else:
                        if hasattr(teacher, 'guard_prefix'):teacher.guard_prefix()
                        try:
                            engine.step(target, opening)
                        except Exception:
                            record_tick()
                            raise
                        record_tick()
                if step % 8 == 7 and hasattr(teacher, 'acknowledge'):
                    teacher.acknowledge(engine, highrate[-200:])
        status = 'VALID_COMPLETE'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        if engine is not None and engine.safety:
            status = 'SAFETY_STOP_PRESERVED'
        write(folder/'ERROR.json', dict(error=error, traceback=traceback.format_exc(), steps=0 if engine is None else engine.steps))
    finally:
        if teacher is not None and hasattr(teacher, 'finish'):teacher.finish(status,error)
        if engine is not None:
            engine.close()
        if physical:
            arrays = {k: np.asarray([s[k] for s in physical]) for k in physical[0]}
            arrays['tcp_world'] = arrays.pop('tcp')
            arrays['contact_time_s'] = np.asarray(contact_times)
            arrays['contact_audit'] = np.asarray(contact_rows).reshape(-1, 5)
            np.savez_compressed(folder/'physics.npz', **arrays)
        if engine is not None:
            high_columns = 1+engine.model.nq+engine.model.nv+engine.model.nu+6+1
            np.savez_compressed(folder/'highrate.npz', rows=np.asarray(highrate).reshape(-1, high_columns), nq=engine.model.nq, nv=engine.model.nv,
                                external_force_max=np.asarray(external_force_max).reshape(-1, 2),
                                layout=np.asarray(['time', 'qpos', 'qvel', 'finite_joint_motor_ctrl',
                                                   'zero_external_wrench', 'safety_flag']))
        write(folder/'samples.json', samples)
        write(folder/'references.json', refs)
        write(folder/'TEACHER_EVENTS.json', [] if teacher is None else teacher.events)
        if engine is not None:
            write(folder/'scene.xml.json', dict(xml=engine.xml, sha256=hashlib.sha256(engine.xml.encode()).hexdigest()))
        result = dict(id=card['id'], status=status, error=error, source_sha=source_sha, fixture=False,
                      physics_starts=starts, physics_steps=0 if engine is None else engine.steps,
                      paired_samples=len(samples), rgb_pairs=rgb_pairs, wall_s=time.monotonic()-began,
                      cpu_s=time.process_time()-cpu_began, memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      model_calls=0, render_backend='mesa_cpu', GPU_service_s=0., domain='FIXED_BASE_ARX5_TASK_SIM',
                      card_sha256=reservation['card_sha256'], samples_sha256=file_hash(folder/'samples.json'),
                      static_qualification_status=None if static_qualification is None else static_qualification['status'],
                      selected='NONE', deployment='NONE')
        if physical:
            try:
                result['score'] = saved_score(folder)
            except Exception as exc:
                result['score_error'] = f'{type(exc).__name__}: {exc}'
                result['status'] = 'FAILED_PRESERVED'
        write(folder/'RESULT.json', result)
        write(folder/'FILE_MANIFEST.json', manifest(folder))
        result['cpu_s'] = time.process_time()-cpu_began
        result['wall_s'] = time.monotonic()-began
        base_bytes = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        result['output_bytes'] = result['owned_bytes'] = base_bytes
        for _ in range(4):
            receipt_bytes = len((json.dumps(jsonable(result), sort_keys=True, indent=2, allow_nan=False)+'\n').encode())
            result['output_bytes'] = result['owned_bytes'] = base_bytes+receipt_bytes
        if result['output_bytes'] > reservation['reservation']['output_bytes']:
            result.update(status='FAILED_PRESERVED', error='RESERVED_OUTPUT_BYTES')
            for _ in range(4):
                receipt_bytes = len((json.dumps(jsonable(result), sort_keys=True, indent=2, allow_nan=False)+'\n').encode())
                result['output_bytes'] = result['owned_bytes'] = base_bytes+receipt_bytes
        write(folder/'WORKER_RECEIPT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--reservation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reservation = json.loads(Path(args.reservation).read_text())
    if Path(args.root).resolve() != Path(reservation['root']).resolve():
        raise ValueError('Root reservation mismatch')
    print(json.dumps(jsonable(run_episode(reservation['card'], args.output, reservation['asset_root'],
                                          reservation['source_sha'], reservation)), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
