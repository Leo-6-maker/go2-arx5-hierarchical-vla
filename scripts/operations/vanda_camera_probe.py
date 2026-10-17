"""Native EGL/dual-camera qualification, explicitly without physical execution."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time


AUTHORITY = 'c8cdf60e41e31da5796b44a064ef2f22d594a491'
SCOPE = 'MP1_VANDA_EGL_TASK_CAMERAS_NO_MODEL_WBC_OR_INTEGRATION'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validate(path, *, compute=False):
    m = read(path)
    root = Path(m['output_root']).resolve()
    assert Path(path).resolve().parent == root, 'MANIFEST_ROOT'
    assert root.is_relative_to(Path(m['registered_runs_root']).resolve()), 'OUTPUT_ESCAPE'
    assert m['owner_uid'] == os.getuid() and m['owner'] == 'EXISTING_LOCAL_PC', 'OWNER'
    assert m['scope'] == SCOPE and m['authority']['protocol_ref'] == AUTHORITY, 'SCOPE'
    assert m['execution']['profile'] == 'vanda-camera-qualification', 'PROFILE'
    assert time.time() < m['deadline_epoch'], 'DEADLINE'
    assert m['limits'] == dict(ngpus=1, active_cpus=4, walltime_s=900,
                              RGB_images=4, physics_steps=0, WBC_launches=0,
                              model_queries=0, optimizer_updates=0), 'LIMITS'
    binding_path = root / 'OUTPUT_BINDING.private.json'
    assert sha(binding_path) == m['execution']['output_binding_sha256'], 'BINDING_HASH'
    binding = read(binding_path)
    assert Path(binding['run_root']).resolve() == root and binding['run_uuid'] == m['run_uuid'], 'BINDING'
    assert any(x['role'] == binding['role'] and Path(x['path']).resolve() == root
               for x in read(binding['registry'])['entries']), 'REGISTRY'
    lease = read(binding['lease'])
    assert lease['status'] == 'ACTIVE' and lease['run_uuid'] == m['run_uuid'], 'LEASE'
    parent = read(m['parent_adoption'])
    assert parent['run_uuid'] == m['authority']['parent_run'], 'PARENT'
    assert parent['authority_release'] == AUTHORITY, 'PARENT_AUTHORITY'
    assert time.time() < parent['new_experiment_cutoff_epoch'], 'PARENT_CUTOFF'
    for name, expected in m['sealed_files'].items():
        assert sha(name) == expected, 'SEALED_INPUT:' + name
    if compute:
        assert socket.gethostname().split('.')[0].lower().startswith('gn-a40-'), 'A40_HOST'
        jobid = os.environ['PBS_JOBID']
        assert read(root / 'SUBMISSION.private.json')['job_id'] == jobid, 'SUBMISSION'
        native = subprocess.run(['/opt/pbs/bin/qstat', '-f', '-F', 'json', jobid],
                                capture_output=True, text=True, timeout=20, check=True)
        job = json.loads(native.stdout)['Jobs'][jobid]
        assert job['job_state'] == 'R' and job['project'] == m['project'], 'PBS_ALLOCATION'
        assert job['Job_Owner'].split('@')[0] == m['owner_user'], 'PBS_OWNER'
        assert int(job['Resource_List']['ngpus']) == 1, 'ONE_GPU'
        assert int(job['Resource_List']['ncpus']) == 4, 'FOUR_CPU'
        h, minute, second = map(int, job['Resource_List']['walltime'].split(':'))
        assert h * 3600 + minute * 60 + second <= 900, 'PBS_WALLTIME'
    return m


def require_render_allocation(m):
    assert socket.gethostname().split('.')[0].lower().startswith('gn-a40-'), 'A40_HOST'
    jobid = os.environ['PBS_JOBID']
    root = Path(m['output_root'])
    assert read(root / 'SUBMISSION.private.json')['job_id'] == jobid, 'SUBMISSION'
    admission = read(root / 'NATIVE_ALLOCATION_ADMISSION.private.json')
    assert admission['job_id'] == jobid and admission['project'] == m['project'], 'ALLOCATION_BINDING'
    assert admission['run_uuid'] == m['run_uuid'], 'ALLOCATION_RUN'
    assert 0 <= time.time() - admission['epoch'] < 120, 'FRESH_NATIVE_ADMISSION'


def render(m):
    import mujoco
    import numpy as np
    from PIL import Image
    from scripts.pi0294.night_geometry import xml_scene, initial_data, initial_audit
    from scripts.pi0294.task_plant import check_frozen_robot

    assert os.environ.get('MUJOCO_GL') == 'egl', 'EGL_REQUIRED'
    from OpenGL import GL

    root = Path(m['output_root'])
    assets = Path(m['assets'])
    card = read(m['card'])
    original = mujoco.MjModel.from_xml_path(str(assets / 'scene.xml'))
    text = xml_scene(card, assets)
    model = mujoco.MjModel.from_xml_string(text)
    frozen = check_frozen_robot(original, model)
    data = initial_data(model)
    initial = initial_audit(model, data, card)
    before = tuple(x.copy() for x in (data.qpos, data.qvel, data.ctrl)) + (data.time,)
    frames = []
    renderer = mujoco.Renderer(model, height=256, width=256)
    try:
        gl = {name: GL.glGetString(key).decode() for name, key in
              [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)]}
        assert 'NVIDIA' in gl['vendor'] and 'A40' in gl['renderer'], 'HARDWARE_EGL_NOT_SOFTWARE'
        for repeat in range(2):
            images = []
            for camera in ('body_rgb', 'wrist_rgb'):
                renderer.update_scene(data, camera=camera)
                pixels = renderer.render().copy()
                assert pixels.shape == (256, 256, 3) and pixels.dtype == np.uint8, 'RGB_SCHEMA'
                assert float(pixels.std()) > 5 and int(pixels.max()) - int(pixels.min()) > 30, 'BLANK_RGB'
                output = root / ('camera-' + str(repeat) + '-' + camera + '.png')
                Image.fromarray(pixels).save(output)
                c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
                transform = np.eye(4)
                transform[:3, :3] = data.cam_xmat[c].reshape(3, 3) @ np.diag([1, -1, -1])
                transform[:3, 3] = data.cam_xpos[c]
                focal = 128 / np.tan(np.deg2rad(model.cam_fovy[c]) / 2)
                frames.append(dict(camera=camera, repeat=repeat, time_s=float(data.time),
                                   png_sha256=sha(output), pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                                   std=float(pixels.std()), K=[[focal, 0, 128], [0, focal, 128], [0, 0, 1]],
                                   T_world_optical=transform.tolist()))
                images.append(pixels)
            assert not np.array_equal(*images), 'CAMERAS_MUST_BE_DISTINCT'
        for old, actual in zip(before[:3], (data.qpos, data.qvel, data.ctrl)):
            np.testing.assert_array_equal(old, actual)
        assert data.time == before[3] == 0, 'NO_PHYSICS_CLOCK_ADVANCE'
    finally:
        renderer.close()
    return dict(status='PASS_NATIVE_A40_EGL_AND_TASK_CAMERAS_NO_DYNAMIC_QUALIFICATION',
                mujoco=mujoco.__version__, OpenGL=gl, frozen_robot=frozen, initial_state=initial,
                scene_xml_sha256=hashlib.sha256(text.encode()).hexdigest(), frames=frames,
                repeated_initial_state_not_fresh_action_observations=True,
                physics_steps=0, WBC_launches=0, model_queries=0, optimizer_updates=0,
                TF_R3_STOP='PRESERVED_NOT_REOPENED', target_closed_loop_qualified=False,
                native_WBC_startup_qualified=False, source_processor_qualified=False,
                selected='NONE', deployment='NONE')


def execute(path):
    m = validate(path, compute=True)
    root = Path(m['output_root'])
    with (root / 'CAMERA_OWNER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root / 'RESULT.private.json').exists(), 'FIRST_RESULT_PRESERVED'
        started = time.monotonic()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = dict(status='ENGINEERING_INCOMPLETE', run_uuid=m['run_uuid'], job_id=os.environ['PBS_JOBID'],
                      physics_steps=0, WBC_launches=0, model_queries=0, optimizer_updates=0)
        try:
            with (root / 'NATIVE_ALLOCATION_ADMISSION.private.json').open('x') as output:
                json.dump(dict(job_id=os.environ['PBS_JOBID'], project=m['project'],
                               run_uuid=m['run_uuid'], epoch=time.time()), output)
            command = [m['apptainer'], 'exec', '--cleanenv', '--nv',
                       '--bind', str(root) + ':' + str(root),
                       '--env', 'MUJOCO_GL=egl,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1',
                       '--env', 'PBS_JOBID=' + os.environ['PBS_JOBID'],
                       m['image'], '/opt/wbc-python/bin/python', str(Path(__file__).resolve()),
                       '--manifest', str(Path(path).resolve()), '--render-only']
            with (root / 'renderer.log').open('x') as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=700)
            child.check_returncode()
            result.update(read(root / 'CAMERA_RESULT.private.json'))
            result['command_exit_status'] = child.returncode
        except BaseException as error:
            result.update(error_type=type(error).__name__)
            raise
        finally:
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            result.update(elapsed_s=time.monotonic() - started, completed_epoch=time.time(),
                          child_CPU_s=after.ru_utime + after.ru_stime - usage.ru_utime - usage.ru_stime)
            with (root / 'RESULT.private.json').open('x') as output:
                json.dump(result, output, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.render_only:
        # The allocation is checked by the outer launcher immediately before
        # entering this container; file seals and run binding are rechecked here.
        m = validate(args.manifest)
        require_render_allocation(m)
        import sys
        sys.path[:0] = [m['capsule']]
        value = render(m)
        with (Path(m['output_root']) / 'CAMERA_RESULT.private.json').open('x') as output:
            json.dump(value, output, indent=2)
    elif args.execute:
        value = execute(args.manifest)
    else:
        value = dict(status='MANIFEST_VALIDATED_NOT_EXECUTED', run_uuid=validate(args.manifest)['run_uuid'])
    print(json.dumps(value))


if __name__ == '__main__':
    main()
