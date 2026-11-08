"""Replay three sealed T06 snapshots through original CPU dual-camera capture.

This is saved-state image reconstruction, not online feedback, a controller
restore, a source outcome, or new target physics. The existing owner must admit
the finite CPU role before this entry can create an EGL Mesa software renderer.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import uuid

from scripts.pi0294.task_native_worker import FrameRecorder
from scripts.pi0294.task_state import ActualState, parse_state
from go2arx5_vla.world_model.fusion_observation import TargetShadowBinding
from go2arx5_vla.world_model.native_replay import forbid_integration, validate_qualified_recording
from go2arx5_vla.world_model.storage import digest, write_json

PROFILE = 'wm02-fusion-target-observation-replay-cpu'
KIND = 'FUSION_TARGET_OBSERVATION_REPLAY'
STEPS = (3503, 3703, 4103)
TICKS = 4103
ORIGINAL_RUN = 'fusion-night1-refit-t06-prefix-01'
ORIGINAL_PARENT = 'FUSION_F2_T06_ROOT_XY_V1'


def require(ok, why):
    if not ok:
        raise ValueError('FUSION_CAPTURE_' + why)


def pinned(entry):
    require(type(entry) is dict and set(entry) == {'path', 'sha256'}, 'PINNED_PATH_AND_HASH')
    path = Path(entry['path'])
    require(path.is_file() and digest(path) == entry['sha256'], 'ORIGINAL_BYTES_CHANGED:' + str(path))
    return path


def original_packets(path):
    """Keep exact journal payload bytes, and require every original 2ms packet."""
    rows, selected = [], {}
    with Path(path).open('rb') as stream:
        while header := stream.read(4):
            require(len(header) == 4, 'TRUNCATED_PACKET_LENGTH')
            size, = struct.unpack('<I', header)
            require(40 <= size <= 2 * 1024**2, 'BOUNDED_PACKET_SIZE')
            raw = stream.read(size)
            require(len(raw) == size, 'TRUNCATED_ORIGINAL_PACKET')
            row = parse_state(raw)
            require(row['step'] == len(rows) + 1
                    and abs(row['time_s'] - row['step'] * .002) <= 1e-8,
                    'COMPLETE_CONTIGUOUS_ORIGINAL_2MS_TRACE')
            for contact in row['contacts']:
                import numpy as np
                require(all(np.isfinite(np.asarray(contact[k])).all() for k in
                            ('distance_m', 'wrench', 'normal', 'frame', 'point')), 'FINITE_ORIGINAL_CONTACTS')
            rows.append(row)
            if row['step'] in STEPS:
                selected[row['step']] = raw
    require(len(rows) == TICKS and tuple(selected) == STEPS, 'EXACT_COMPLETE_4103_TICKS_AND_THREE_SNAPSHOTS')
    return rows, selected


def validate_inputs(spec):
    require(spec.get('profile') == PROFILE and spec.get('kind') == KIND
            and spec.get('execution_site') == 'CORE_FUSION_TARGET_OBSERVATION_REPLAY_CPU'
            and spec.get('device') == 'cpu' and spec.get('physics') is False
            and spec.get('capture_steps') == list(STEPS) and spec.get('expected_native_ticks') == TICKS,
            'FINITE_THREE_FRAME_CPU_REPLAY_PROFILE')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('MUJOCO_GL') == 'egl'
            and os.environ.get('PYOPENGL_PLATFORM') == 'egl'
            and os.environ.get('LIBGL_ALWAYS_SOFTWARE') == '1'
            and os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES') == '/usr/share/glvnd/egl_vendor.d/50_mesa.json'
            and 'MUJOCO_EGL_DEVICE_ID' not in os.environ, 'EXPLICIT_CPU_EGL_MESA_ENVIRONMENT')
    original = spec['original']; root = Path(original['run_root'])
    require(root.name == original['run_id'] == ORIGINAL_RUN, 'EXACT_ORIGINAL_T06_QUALIFICATION')
    required = ('spec', 'episode', 'lease', 'worker_exit', 'native_result', 'journal', 'scene')
    paths = {key: pinned(original[key]) for key in required}
    expected = dict(spec=root/'EXECUTION_SPEC.json', episode=root/'EPISODE_MANIFEST.private.json',
        worker_exit=root/'WORKER_EXIT.private.json', native_result=root/'native-01/RESULT.json',
        journal=root/'native-01/state.bin.journal')
    require(all(paths[k].resolve() == p.resolve() for k, p in expected.items()), 'ORIGINAL_RUN_MEMBER_PATHS')
    native_spec = json.loads(paths['spec'].read_text())
    episode = json.loads(paths['episode'].read_text())
    lease = json.loads(paths['lease'].read_text())
    worker = json.loads(paths['worker_exit'].read_text())
    native = json.loads(paths['native_result'].read_text())
    require(Path(native_spec['registered_root']).resolve() == root.resolve()
            and Path(native_spec['lease']).resolve() == paths['lease'].resolve()
            and paths['lease'].resolve().is_relative_to(root.resolve())
            and Path(native_spec['episode_manifest']).resolve() == paths['episode'].resolve()
            and native_spec['episode_manifest_sha256'] == original['episode']['sha256']
            and Path(native_spec['model_path']).resolve() == paths['scene'].resolve(), 'ORIGINAL_NATIVE_SPEC_BINDING')
    require(episode.get('parent') == ORIGINAL_PARENT and episode.get('scene_id') == 'F2T06'
            and episode.get('purpose') == 'SCENE_QUALIFICATION' and episode.get('split') == 'TRAIN'
            and episode.get('inference_mode') == 'CALIBRATION_NO_VLA', 'ORIGINAL_T06_TRAIN_QUALIFICATION_ROLE')
    require(lease.get('status') == 'CLOSED' and lease.get('run_uuid') == root.name
            and lease.get('spec_sha256') == original['spec']['sha256']
            and lease.get('worker_exit_code') == 0 and worker.get('exit_code') == 0
            and worker.get('failure') is None, 'CLOSED_CLEAN_ORIGINAL_NATIVE_WORKER')
    require(native_spec['source_sha'] == original['source_sha'] == native.get('source_sha')
            and native.get('native_binary_sha256') == digest(native_spec['native_binary']), 'FROZEN_NATIVE_RUNTIME_IDENTITY')
    require(bool(native_spec.get('frozen_files')) and bool(native_spec.get('source_files')), 'ORIGINAL_FROZEN_RUNTIME_INVENTORIES')
    for path, pin in native_spec['frozen_files'].items():
        require(digest(path) == pin, 'ORIGINAL_FROZEN_RUNTIME_CHANGED:' + path)
    for name, pin in native_spec['source_files'].items():
        require(digest(Path(native_spec['source_root'])/name) == pin, 'ORIGINAL_SOURCE_CAPSULE_CHANGED:' + name)
    loaded = original['loaded_executor']
    require(type(loaded) is list and len(loaded) == 2, 'ORIGINAL_PRE_AND_POST_LOADED_IDENTITIES')
    for index, entry in enumerate(loaded):
        path = pinned(entry)
        require(path.resolve() == (root/'native-01'/f'LOADED_EXECUTOR_{index:02d}.private.json').resolve(),
                'EXACT_ORIGINAL_LOADED_IDENTITY_PATH')
        value = json.loads(path.read_text())
        require(value['sim']['executable_sha256'] == native['native_binary_sha256'], 'ACTUAL_LOADED_NATIVE_BINARY_HASH')
    rows, selected = original_packets(paths['journal'])
    validate_qualified_recording(dict(scene=original['scene'], original_native_source_sha=original['source_sha']), native, rows)
    require(native['actual_target_integrations'] == TICKS, 'ORIGINAL_4103_NATIVE_PASS')
    return paths, native_spec, selected


def software_renderer_identity():
    from OpenGL import GL
    vendor = GL.glGetString(GL.GL_VENDOR).decode()
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    require('mesa' in vendor.lower() and any(name in renderer.lower() for name in
            ('llvmpipe', 'softpipe', 'software rasterizer')), 'ACTUAL_MESA_SOFTWARE_RENDERER')
    return dict(GL_VENDOR=vendor, GL_RENDERER=renderer, backend='egl_mesa_cpu')


@contextmanager
def count_forward(mujoco):
    original = mujoco.mj_forward
    count = [0]
    def observed(*args, **kwargs):
        count[0] += 1
        return original(*args, **kwargs)
    mujoco.mj_forward = observed
    try:
        yield count
    finally:
        mujoco.mj_forward = original


def write_bytes(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    require(Path(path).read_bytes() == raw, 'RETAINED_PACKET_READBACK')


def run(spec, output):
    output = Path(output); start, cpu = time.monotonic(), time.process_time()
    paths, native_spec, selected = validate_inputs(spec)
    import mujoco
    require(mujoco.__version__ == '3.2.7' and mujoco.get_mjcb_control() is None,
            'ORIGINAL_MUJOCO_WITHOUT_CONTROLLER_CALLBACK')
    rows = []; recorders = []; renderer = None; forwards = [0]; calls = []
    save_attempts = completed_saves = 0
    failure = None
    try:
        with forbid_integration(mujoco) as calls, count_forward(mujoco) as forwards:
            model = mujoco.MjModel.from_binary_path(str(paths['scene']))
            require((model.nq, model.nv, model.nu, model.na, model.nplugin) == (34, 32, 20, 0, 0),
                    'ORIGINAL_TARGET_MODEL_ABI')
            state = ActualState(model)
            frame_map = spec.get('frame_map')
            binding = TargetShadowBinding(spec['target_world_frame'], spec['original']['scene']['sha256'],
                spec['original']['spec']['sha256'],
                None if frame_map is None else frame_map['T_source_from_target'],
                None if frame_map is None else frame_map['sha256'])
            recorder = FrameRecorder(paths['scene'], output)
            recorders.append(recorder); (output/'images').mkdir()
            for index, step in enumerate(STEPS):
                raw = selected[step]; packet = parse_state(raw)
                write_json(output/f'CAPTURE_INTENT_{step}.private.json', dict(native_step=step,
                    native_state_sha256=hashlib.sha256(raw).hexdigest(), renderer_save_requested=1,
                    input_mode='SEALED_TARGET_NATIVE_REPLAY_NOT_ONLINE'))
                save_attempts += 1
                saved = recorder.save(index, packet)  # Exactly one save per original packet.
                completed_saves += 1
                if renderer is None:
                    renderer = software_renderer_identity()
                require(saved == dict(index=index, native_step=step, time_s=packet['time_s']), 'ORIGINAL_RECORDER_SAME_PACKET')
                raw_path = output/f'STATE_{step}.private.bin'; write_bytes(raw_path, raw)
                calibration_path = output/'images'/f'{index:06d}-calibration.json'
                image_paths = {camera: output/'images'/f'{index:06d}-{camera}.png' for camera in ('body_rgb', 'wrist_rgb')}
                image_hashes = {camera: digest(path) for camera, path in image_paths.items()}
                expected_state = digest(raw_path); expected_calibration = digest(calibration_path)
                bound = binding.bind(actual_state=state, native_state_bytes=raw_path.read_bytes(),
                    expected_state_sha256=expected_state, capture=saved,
                    calibration_bytes=calibration_path.read_bytes(), expected_calibration_sha256=expected_calibration,
                    image_bytes={camera: path.read_bytes() for camera, path in image_paths.items()},
                    expected_image_sha256=image_hashes, language=spec['language'], snapshot_uuid=str(uuid.uuid4()))
                bound_path = output/f'BOUND_OBSERVATION_{step}.private.json'; write_json(bound_path, bound)
                retained = {path.relative_to(output).as_posix(): digest(path)
                    for path in [raw_path, calibration_path, bound_path, *image_paths.values()]}
                row = dict(native_step=step, native_time_s=packet['time_s'], capture=saved,
                    native_state_sha256=expected_state, calibration_sha256=expected_calibration,
                    image_sha256=image_hashes, files=retained,
                    original_native_result_sha256=spec['original']['native_result']['sha256'],
                    frame_relation='SEMANTICS_UNRESOLVED' if frame_map is None else 'EXPLICIT_RIGID_MAP')
                write_json(output/f'CAPTURE_{step}.private.json', row)
                require(all(digest(output/name) == pin for name, pin in retained.items()), 'CAPTURE_ORIGINAL_BYTE_READBACK')
                rows.append(row)
            require(len(rows) == 3 and len(list((output/'images').glob('*.png'))) == 6
                    and len(list((output/'images').glob('*-calibration.json'))) == 3 and not calls,
                    'EXACT_THREE_DUAL_CAMERA_SAVES_ZERO_INTEGRATIONS')
        # Original native bytes and every runtime binding remain read-only.
        validate_inputs(spec)
    except BaseException as error:
        failure = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        close_failure = None
        for recorder in recorders:
            try:
                recorder.close()
            except BaseException as error:
                close_failure = type(error).__name__ + ': ' + str(error)
        if close_failure is not None and failure is None:
            failure = 'RECORDER_CLOSE_FAILED:' + close_failure
        result = dict(schema='WM02_FUSION_TARGET_OBSERVATION_REPLAY_CAPTURE_V1',
            status='SAVED_TARGET_OBSERVATIONS_CAPTURED_SHADOW_ONLY' if failure is None else 'CAPTURE_FAILED_PRESERVED_NO_RETRY',
            failure=failure, recorder_close_failure=close_failure,
            source_sha=spec['source_sha'], original_source_sha=spec['original']['source_sha'],
            original_native_result_sha256=spec['original']['native_result']['sha256'],
            original_journal_sha256=spec['original']['journal']['sha256'], scene_sha256=spec['original']['scene']['sha256'],
            original_native_ticks=TICKS, renderer_save_attempts=save_attempts,
            actual_renderer_saves=completed_saves, bound_observations=len(rows),
            actual_dual_camera_PNGs=len(list((output/'images').glob('*.png'))),
            actual_python_forward_calls=forwards[0], prohibited_integration_calls=len(calls),
            native_integrations=0, VLA_inferences=0, model_loads=0, optimizer_updates=0,
            input_mode='SEALED_TARGET_NATIVE_REPLAY_NOT_ONLINE', online_closed_loop=False,
            exact_controller_restore=False, source_outcomes_supplied=False, target_embodiment_qualified=False,
            renderer=renderer, captures=rows, wall_s=time.monotonic()-start, CPU_s=time.process_time()-cpu)
        write_json(output/'RESULT.private.json', result)
        if close_failure is not None:
            raise RuntimeError('FUSION_CAPTURE_RECORDER_CLOSE_FAILED:' + close_failure)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(args.spec, args.output.name, PROFILE, args.output, physics=False, gpu=False) as spec:
        value = run(spec, args.output)
    print(json.dumps(value)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
