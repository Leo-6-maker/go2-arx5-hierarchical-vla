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
N1_RELEASE = '926f9ef69042bb9f3b7b336a076d2be8fa0afbb9'
N1_DELTA_SHA = '91d41f1f810f97caf0cb9be14882d4b94fd13e5af3bedadb3505277693281cce'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_n1_window(m, lease, parent, now):
    root = Path(m['output_root']).resolve()
    window = m['execution'].get('n1_camera_window')
    if window is None:
        assert root.name != 'vanda-egl-cameras-n1-03', 'N1_WINDOW_REQUIRED_FOR_NEW_CHILD'
        return
    parent_root = Path(m['parent_adoption']).resolve().parent
    first = parent_root / 'authority/N1/ADOPTION_DELTA.json'
    card_path = root / 'N1_QUALIFICATION_CARD.json'
    assert root.parent.parent == parent_root, 'N1_REGISTERED_PARENT'
    assert parent['run_uuid'] == '20261016T191445Z-localpc-mp1' and parent['T0_epoch'] == 1792178085, 'N1_ORIGINAL_PARENT'
    assert first.resolve().is_relative_to(parent_root) and not first.is_symlink() and not first.stat().st_mode & 0o222, 'N1_IMMUTABLE_FIRST_ADOPTION'
    assert sha(first) == N1_DELTA_SHA and m['sealed_files'].get(str(first)) == N1_DELTA_SHA, 'N1_FIRST_ADOPTION_HASH'
    delta = read(first)
    assert delta['authority_release'] == N1_RELEASE and delta['parent_authority_release'] == AUTHORITY, 'N1_AUTHORITY'
    assert delta['reset_on_resume'] is False and delta['new_competing_owner'] is False, 'N1_NO_RESET_OR_OWNER'
    assert set(window) == {'release', 'adoption_delta_sha256', 'created_epoch', 'deadline_epoch',
                           'maximum_queue_and_execution_window_s', 'original_N1_T0_epoch'}, 'N1_WINDOW_FIELDS'
    assert window['release'] == N1_RELEASE and window['adoption_delta_sha256'] == N1_DELTA_SHA, 'N1_WINDOW_AUTHORITY'
    assert window['original_N1_T0_epoch'] == delta['autonomy_started_epoch'], 'N1_ORIGINAL_T0'
    assert window['maximum_queue_and_execution_window_s'] == 43200, 'N1_BOUNDED_QUEUE_AND_EXECUTION_WINDOW'
    assert delta['autonomy_started_epoch'] <= window['created_epoch'] <= now, 'N1_ACTUAL_BINDING_TIME'
    assert window['created_epoch'] < delta['new_submission_and_candidate_cutoff_epoch'], 'N1_NEW_BINDING_CUTOFF'
    assert now < window['deadline_epoch'] <= min(window['created_epoch'] + 43200,
                                                delta['new_experiment_cutoff_epoch']), 'N1_ABSOLUTE_CUTOFF'
    assert window['deadline_epoch'] == m['deadline_epoch'] == lease['deadline_epoch'], 'N1_EXACT_NEW_LEASE_DEADLINE'
    assert card_path.resolve().is_relative_to(root) and not card_path.is_symlink() and not card_path.stat().st_mode & 0o222, 'N1_IMMUTABLE_CAMERA_CARD'
    assert m['sealed_files'].get(str(card_path)) == sha(card_path), 'N1_CAMERA_CARD_HASH'
    card = read(card_path)
    assert card['schema'] == 'MP1_N1_VANDA_CAMERA_ONLY_NEW_WINDOW_V1' and card['authority_release'] == N1_RELEASE, 'N1_CAMERA_CARD'
    assert root.name == card['child'] == 'vanda-egl-cameras-n1-03', 'N1_NEW_CHILD_NOT_OLD_LEASE'
    assert card['maximum_new_PBS'] == 1 and card['maximum_queue_and_execution_window_s'] == 43200, 'N1_ONE_CAMERA_SUBMISSION'
    assert card['limits'] == m['limits'], 'N1_CAMERA_LIMITS'
    old = parent_root / 'children/vanda-egl-cameras-02/OUTPUT_BINDING.private.json'
    assert read(read(old)['lease'])['status'] == 'CLOSED', 'N1_OLD_CAMERA_LEASE_STAYS_CLOSED'


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
    validate_n1_window(m, lease, parent, time.time())
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
        # Vanda auto expands a GPU request to its native 36-CPU bundle.
        # Active work remains pinned to four CPUs, below both MP1 and PBS caps.
        assert 4 <= int(job['Resource_List']['ncpus']) <= 36, 'NATIVE_CPU_BUNDLE'
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


def constrain_cpu():
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) >= 4, 'FOUR_AVAILABLE_CPUS'
    os.sched_setaffinity(0, set(cpus[:4]))
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))


def container_binds(m):
    root = Path(m['output_root']).resolve()
    parent = Path(m['parent_adoption']).resolve().parent
    binding = read(root / 'OUTPUT_BINDING.private.json')
    readonly = [parent, Path(m['assets']), Path(m['image']),
                Path(binding['registry']), Path(binding['lease'])]
    if 'n1_camera_window' in m['execution']:
        old = read(parent / 'children/vanda-egl-cameras-02/OUTPUT_BINDING.private.json')
        readonly.append(Path(old['lease']))
    paths = list(dict.fromkeys(str(p.resolve()) for p in readonly))
    assert all(':' not in p for p in paths + [str(root)]), 'EXACT_CONTAINER_BIND_PATHS'
    return [p + ':' + p + ':ro' for p in paths] + [str(root) + ':' + str(root) + ':rw']


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
    constrain_cpu()
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
            command = [m['apptainer'], 'exec', '--cleanenv', '--nv']
            for binding in container_binds(m):
                command.extend(['--bind', binding])
            command.extend([
                       '--env', 'MUJOCO_GL=egl,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1',
                       '--env', 'PBS_JOBID=' + os.environ['PBS_JOBID'],
                       m['image'], '/opt/wbc-python/bin/python', str(Path(__file__).resolve()),
                       '--manifest', str(Path(path).resolve()), '--render-only'])
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
