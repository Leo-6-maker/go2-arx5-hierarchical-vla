"""Raw ABI/identity checks and synthetic capture doubles; no renderer/native run."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.wm02 import capture_fusion_observation as unit
from scripts.pi0294.task_state import ActualState
from scripts.pi0292.relay_protocol import unseal
from go2arx5_vla.world_model.fusion_observation import shadow_inference_payload
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_fusion_observation import fixture as observation_fixture

ENV = dict(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', LIBGL_ALWAYS_SOFTWARE='1',
           __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=digest(path))


def packet(step, *, stamp=None):
    values = np.zeros(86, dtype='<f8'); values[0] = -0.; values[3] = 1.; values[-1] = .123
    return (struct.pack('<8sQdIII', b'GRTSTATE', step, step * .002 if stamp is None else stamp, 34, 32, 20)
            + values.tobytes() + struct.pack('<I', 0))


def fixture(tmp):
    root = Path(tmp)/unit.ORIGINAL_RUN; (root/'native-01').mkdir(parents=True)
    old_source = Path(tmp)/'original-capsule'; old_source.mkdir()
    source_file = old_source/'frozen.py'; source_file.write_text('frozen-original-source')
    binary = Path(tmp)/'native-binary'; binary.write_bytes(b'NOT_AN_EXECUTABLE_FIXTURE')
    scene = Path(tmp)/'scene.mjb'; scene.write_bytes(b'NOT_A_MUJOCO_MODEL_FIXTURE')
    source = '0' * 40
    ep = write(root/'EPISODE_MANIFEST.private.json', dict(parent=unit.ORIGINAL_PARENT, scene_id='F2T06',
        purpose='SCENE_QUALIFICATION', split='TRAIN', inference_mode='CALIBRATION_NO_VLA'))
    lease_path = root/'control/LEASE.json'
    old = write(root/'EXECUTION_SPEC.json', dict(registered_root=str(root), lease=str(lease_path),
        episode_manifest=ep['path'], episode_manifest_sha256=ep['sha256'], model_path=str(scene),
        source_sha=source, source_root=str(old_source), native_binary=str(binary),
        source_files={'frozen.py': digest(source_file)}, frozen_files={str(binary): digest(binary)}))
    lease = write(lease_path, dict(status='CLOSED', run_uuid=root.name, spec_sha256=old['sha256'], worker_exit_code=0))
    worker = write(root/'WORKER_EXIT.private.json', dict(exit_code=0, failure=None))
    native = write(root/'native-01/RESULT.json', dict(status='NATIVE_STAGE_PASSED', shutdown_clean=True,
        physics_stops=[], original_joint_or_warning_samples=[], owned_process_exits=[dict(exit_code=0)],
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_PASS'), scene_sha256=digest(scene),
        source_sha=source, native_binary_sha256=digest(binary), actual_target_integrations=unit.TICKS))
    journal = root/'native-01/state.bin.journal'
    with journal.open('wb') as stream:
        for step in range(1, unit.TICKS + 1):
            raw = packet(step); stream.write(struct.pack('<I', len(raw))); stream.write(raw)
    loaded = [write(root/'native-01'/f'LOADED_EXECUTOR_{i:02d}.private.json',
              dict(sim=dict(executable_sha256=digest(binary)))) for i in range(2)]
    original = dict(run_root=str(root), run_id=root.name, source_sha=source, spec=old, episode=ep,
        lease=lease, worker_exit=worker, native_result=native, journal=dict(path=str(journal), sha256=digest(journal)),
        scene=dict(path=str(scene), sha256=digest(scene)), loaded_executor=loaded)
    return dict(profile=unit.PROFILE, kind=unit.KIND, execution_site='CORE_FUSION_TARGET_OBSERVATION_REPLAY_CPU',
        device='cpu', physics=False, capture_steps=list(unit.STEPS), expected_native_ticks=unit.TICKS,
        original=original, source_sha='1' * 40, target_world_frame='ACTUAL_TARGET_WORLD', language='Original task instruction')


@contextmanager
def capture_doubles(spec, *, integrate=False):
    binding, args, measured, transform = observation_fixture()
    reader = args['actual_state']; model = reader.model
    for k, v in dict(nq=34, nv=32, nu=20, na=0, nplugin=0).items(): setattr(model, k, v)
    state_calls = []; saves = []; closed = []
    mj = SimpleNamespace(__version__='3.2.7', get_mjcb_control=lambda: None,
        MjModel=SimpleNamespace(from_binary_path=lambda _: model), mj_forward=lambda *a: None,
        mj_step=lambda *a: (_ for _ in ()).throw(AssertionError('MUST_BE_GUARDED')),
        mj_step1=lambda *a: None, mj_step2=lambda *a: None)
    class Recorder:
        def __init__(self, model_path, folder): self.folder = folder
        def save(self, index, row):
            saves.append((index, row['step']))
            if integrate: mj.mj_step(None, None)
            calibration = json.loads(args['calibration_bytes'])
            for value in calibration.values(): value['timestamp_s'] = row['time_s']
            for camera, raw in args['image_bytes'].items():
                (self.folder/'images'/f'{index:06d}-{camera}.png').write_bytes(raw)
            (self.folder/'images'/f'{index:06d}-calibration.json').write_text(json.dumps(calibration))
            return dict(index=index, native_step=row['step'], time_s=row['time_s'])
        def close(self): closed.append(True)
    def update(row):
        state_calls.append(row['step'])
        return dict(measured, native_step=row['step'], time_s=row['time_s'])
    with patch.dict(sys.modules, mujoco=mj), patch.object(unit, 'FrameRecorder', Recorder), \
         patch.object(unit, 'ActualState', return_value=reader), patch.object(ActualState, 'update', side_effect=update), \
         patch.object(unit, 'software_renderer_identity', return_value=dict(GL_VENDOR='Mesa', GL_RENDERER='llvmpipe CPU FIXTURE ONLY')):
        yield SimpleNamespace(saves=saves, state_calls=state_calls, closed=closed, transform=transform,
                              transform_sha256=binding.expected_transform_sha256)


class CaptureFusionObservation(unittest.TestCase):
    def test_complete_original_packets_keep_exact_bytes_and_no_missing_native_ticks(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = fixture(tmp)
            rows, selected = unit.original_packets(spec['original']['journal']['path'])
            self.assertEqual(len(rows), 4103)
            self.assertEqual(tuple(selected), unit.STEPS)
            for step, raw in selected.items(): self.assertEqual(raw, packet(step))
            self.assertEqual(selected[3703][36:44], struct.pack('<d', -0.))

    def test_replay_captures_only_three_snapshots_and_retains_original_hashes_without_frame_map(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV):
            spec = fixture(tmp); output = Path(tmp)/'captured'; output.mkdir()
            with capture_doubles(spec) as observed:
                result = unit.run(spec, output)
            self.assertEqual(observed.saves, [(0, 3503), (1, 3703), (2, 4103)])
            self.assertEqual(observed.state_calls, list(unit.STEPS)); self.assertEqual(observed.closed, [True])
            self.assertEqual(result['actual_renderer_saves'], 3)
            self.assertEqual(result['actual_dual_camera_PNGs'], 6)
            self.assertEqual(result['native_integrations'], 0)
            self.assertFalse(result['online_closed_loop']); self.assertFalse(result['exact_controller_restore'])
            for row in result['captures']:
                step = row['native_step']
                self.assertEqual((output/f'STATE_{step}.private.bin').read_bytes(), packet(step))
                bound = json.loads((output/f'BOUND_OBSERVATION_{step}.private.json').read_text())
                self.assertIsNone(unseal(bound)['source_policy_view'])
                self.assertEqual(unseal(bound)['native_time_s'], step * .002)
                self.assertTrue(all(digest(output/name) == pin for name, pin in row['files'].items()))
            self.assertEqual(digest(spec['original']['journal']['path']), spec['original']['journal']['sha256'])

    def test_explicit_pinned_map_produces_shadow_policy_view_without_source_outcome(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV):
            spec = fixture(tmp); output = Path(tmp)/'captured'; output.mkdir()
            with capture_doubles(spec) as observed:
                spec['frame_map'] = dict(T_source_from_target=observed.transform.tolist(), sha256=observed.transform_sha256)
                unit.run(spec, output)
            bound = json.loads((output/'BOUND_OBSERVATION_3703.private.json').read_text())
            payload = shadow_inference_payload(bound)
            self.assertEqual([i['camera'] for i in payload['images']], ['external_rgb', 'wrist_rgb'])
            self.assertEqual(unseal(bound)['provenance']['desired_outcome_provenance'], 'SEMANTICS_UNRESOLVED')
            self.assertFalse(unseal(bound)['provenance']['source_outcomes_supplied'])

    def test_closed_native_pass_and_frozen_runtime_are_checked_before_renderer(self):
        for fault in ('active', 'native_fail', 'runtime', 'native_binary', 'task_role', 'loaded', 'trace_audit'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV):
                spec = fixture(tmp); original = spec['original']
                key = {'active':'lease', 'native_fail':'native_result', 'runtime':'native_result',
                       'native_binary':'native_result', 'task_role':'episode', 'trace_audit':'native_result'}.get(fault)
                if fault == 'loaded':
                    entry = original['loaded_executor'][0]; value = dict(sim=dict(executable_sha256='f' * 64))
                    original['loaded_executor'][0] = write(Path(entry['path']), value)
                else:
                    entry = original[key]; value = json.loads(Path(entry['path']).read_text())
                    if fault == 'active': value['status'] = 'ACTIVE'
                    elif fault == 'native_fail': value['status'] = 'FAILED'
                    elif fault == 'runtime': value['source_sha'] = 'f' * 40
                    elif fault == 'native_binary': value['native_binary_sha256'] = 'f' * 64
                    elif fault == 'task_role': value['purpose'] = 'EVALUATION'
                    else: value['trace_audit']['status'] = 'FAILED'
                    original[key] = write(Path(entry['path']), value)
                with patch.object(unit, 'FrameRecorder') as recorder:
                    with self.assertRaises(ValueError): unit.validate_inputs(spec)
                    recorder.assert_not_called()

    def test_missing_reordered_or_retimed_packet_cannot_be_a_complete_trace(self):
        for fault in ('gap', 'reordered', 'retimed', 'partial'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'journal'
                with path.open('wb') as stream:
                    for step in range(1, 4104):
                        if fault == 'gap' and step == 5: continue
                        if fault == 'partial' and step == 4103: break
                        raw = packet(5 if fault == 'reordered' and step == 6 else step,
                                     stamp=0. if fault == 'retimed' and step == 3703 else None)
                        stream.write(struct.pack('<I', len(raw))); stream.write(raw)
                with self.assertRaises(ValueError): unit.original_packets(path)

    def test_integration_is_denied_and_failure_keeps_first_intent_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV):
            spec = fixture(tmp); output = Path(tmp)/'captured'; output.mkdir()
            with capture_doubles(spec, integrate=True) as observed:
                with self.assertRaisesRegex(RuntimeError, 'REPLAY_FORBIDS_INTEGRATION'): unit.run(spec, output)
            self.assertEqual(observed.saves, [(0, 3503)]); self.assertEqual(observed.closed, [True])
            result = json.loads((output/'RESULT.private.json').read_text())
            self.assertEqual(result['renderer_save_attempts'], 1)
            self.assertEqual(result['actual_renderer_saves'], 0)
            self.assertEqual(result['prohibited_integration_calls'], 1)
            self.assertEqual(result['status'], 'CAPTURE_FAILED_PRESERVED_NO_RETRY')
            self.assertTrue((output/'CAPTURE_INTENT_3503.private.json').exists())

    def test_failed_binding_still_reports_actual_completed_capture_and_retains_pixels(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV):
            spec = fixture(tmp); output = Path(tmp)/'captured'; output.mkdir()
            with capture_doubles(spec) as observed, patch.object(unit.TargetShadowBinding, 'bind', side_effect=ValueError('BAD_CAPTURE')):
                with self.assertRaisesRegex(ValueError, 'BAD_CAPTURE'): unit.run(spec, output)
            result = json.loads((output/'RESULT.private.json').read_text())
            self.assertEqual(observed.saves, [(0, 3503)])
            self.assertEqual(result['actual_renderer_saves'], 1)
            self.assertEqual(result['actual_dual_camera_PNGs'], 2)
            self.assertEqual(result['bound_observations'], 0)
            self.assertEqual((output/'STATE_3503.private.bin').read_bytes(), packet(3503))

    def test_profile_env_and_frame_count_cannot_admit_other_work(self):
        for change in (dict(physics=True), dict(device='cuda'), dict(capture_steps=[3503]),
                       dict(expected_native_ticks=4104), dict(kind='OTHER')):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV):
                spec = fixture(tmp); spec.update(change)
                with self.assertRaises(ValueError): unit.validate_inputs(spec)
        for environment in (dict(ENV, MUJOCO_GL='osmesa'), dict(ENV, PYOPENGL_PLATFORM='glx'),
                            dict(ENV, LIBGL_ALWAYS_SOFTWARE='0'), dict(ENV, CUDA_VISIBLE_DEVICES='0'),
                            dict(ENV, __EGL_VENDOR_LIBRARY_FILENAMES='/other/vendor.json'),
                            dict(ENV, MUJOCO_EGL_DEVICE_ID='0'), dict(ENV, MUJOCO_EGL_DEVICE_ID='')):
            with self.subTest(environment=environment), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, environment):
                with self.assertRaises(ValueError): unit.validate_inputs(fixture(tmp))

    def test_actual_gl_identity_requires_mesa_software_renderer_without_gpu_fallback(self):
        for vendor, renderer, allowed in (('Mesa/X.org', 'llvmpipe (LLVM)', True),
                                         ('Mesa', 'softpipe', True),
                                         ('NVIDIA Corporation', 'NVIDIA GPU', False),
                                         ('Mesa/X.org', 'AMD GPU', False)):
            gl = SimpleNamespace(GL_VENDOR=1, GL_RENDERER=2,
                glGetString=lambda key: (vendor if key == 1 else renderer).encode())
            with self.subTest(vendor=vendor, renderer=renderer), patch.dict(sys.modules, OpenGL=SimpleNamespace(GL=gl)):
                if allowed:
                    value = unit.software_renderer_identity()
                    self.assertEqual(value['backend'], 'egl_mesa_cpu')
                    self.assertEqual(value['GL_RENDERER'], renderer)
                else:
                    with self.assertRaisesRegex(ValueError, 'ACTUAL_MESA_SOFTWARE_RENDERER'):
                        unit.software_renderer_identity()

    def test_cli_uses_existing_cpu_admission_and_never_admits_physics_or_gpu(self):
        @contextmanager
        def admitted(*args, **kwargs):
            self.assertEqual(args[2], unit.PROFILE)
            self.assertEqual(kwargs, dict(physics=False, gpu=False))
            yield {'fixture': True}
        with patch('go2arx5_vla.world_model.runtime.admitted_run', side_effect=admitted) as admission, \
             patch.object(unit, 'run', return_value=dict(status='CPU_FIXTURE_ONLY')) as run, patch('builtins.print'):
            self.assertEqual(unit.main(['--spec', '/fixture/spec', '--output', '/fixture/output']), 0)
            admission.assert_called_once(); run.assert_called_once()

    def test_cli_import_does_not_load_neural_simulator_or_ros_runtime(self):
        code = ('import sys; import scripts.wm02.capture_fusion_observation; '
                'assert not any(k in sys.modules for k in ("torch", "mujoco", "rclpy"))')
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
