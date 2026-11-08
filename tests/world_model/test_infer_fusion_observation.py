"""Actual saved-target binding reader; synthetic captures, no GPU or model."""
import json
import struct
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_fusion_observation import fixture
from scripts.pi0292.relay_protocol import sealed, unseal
from scripts.pi0294.task_state import ActualState
from scripts.wm02.infer_fusion_vla import recorded_payload
from go2arx5_vla.world_model.storage import digest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def capture_fixture(root):
    output = root/'fit-01'
    output.mkdir()
    rows = []
    for index, step in enumerate((3503, 3703, 4103)):
        binding, args, measured, transform = fixture(step)
        args['capture']['index'] = index
        with patch.object(ActualState, 'update', return_value=measured):
            record = binding.bind(**args)
        state, cal = output/f'STATE_{step}.private.bin', output/f'images/{index:06d}-calibration.json'
        state.write_bytes(args['native_state_bytes'])
        cal.parent.mkdir(exist_ok=True)
        cal.write_bytes(args['calibration_bytes'])
        image_files = {}
        for camera, raw in args['image_bytes'].items():
            path = output/f'images/{index:06d}-{camera}.png'
            path.write_bytes(raw)
            image_files[camera] = path
        bound = output/f'BOUND_OBSERVATION_{step}.private.json'
        put(bound, record)
        files = {p.relative_to(output).as_posix():digest(p) for p in (state, cal, bound, *image_files.values())}
        rows.append(dict(native_step=step, native_time_s=step*.002, files=files,
            capture=args['capture'],
            native_state_sha256=digest(state), calibration_sha256=digest(cal),
            image_sha256={key:digest(path) for key,path in image_files.items()}))
    producer = dict(source_sha='SYNTHETIC_CAPTURE_SOURCE', task_prompt_matches_target_task=False,
        language_provenance='SYNTHETIC_SOURCE_PROMPT_ON_TARGET_CALIBRATION_UNQUALIFIED',
        original=dict(scene=dict(sha256='1'*64), spec=dict(sha256='2'*64),
            native_result=dict(sha256='3'*64), journal=dict(sha256='4'*64)),
        frame_map=dict(T_source_from_target=transform.tolist(), sha256=binding.expected_transform_sha256))
    put(root/'EXECUTION_SPEC.json', producer)
    put(root/'control/LEASE.json', dict(status='CLOSED', worker_exit_code=0,
        spec_sha256=digest(root/'EXECUTION_SPEC.json')))
    put(root/'WORKER_EXIT.private.json', dict(exit_code=0))
    result = dict(schema='WM02_FUSION_TARGET_OBSERVATION_REPLAY_CAPTURE_V1',
        status='SAVED_TARGET_OBSERVATIONS_CAPTURED_SHADOW_ONLY', failure=None,
        native_integrations=0, prohibited_integration_calls=0, actual_renderer_saves=3,
        bound_observations=3, actual_dual_camera_PNGs=6, captures=rows,
        source_sha=producer['source_sha'], scene_sha256='1'*64, original_native_result_sha256='3'*64,
        original_journal_sha256='4'*64, original_native_ticks=4103)
    put(output/'RESULT.private.json', result)
    return dict(recorded_observation=str(output/'BOUND_OBSERVATION_3703.private.json'),
        recorded_observation_sha256=digest(output/'BOUND_OBSERVATION_3703.private.json'),
        observation_mode='SAVED_TARGET_NATIVE_REPLAY_SHADOW', target_capture_step=3703,
        target_capture_result=str(output/'RESULT.private.json'),
        target_capture_result_sha256=digest(output/'RESULT.private.json'))


class SavedObservationReadback(unittest.TestCase):
    def test_target_reader_keeps_complete_provenance_and_original_pixels_without_model_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = capture_fixture(Path(tmp))
            payload, provenance = recorded_payload(spec)
            body = unseal(json.loads(Path(spec['recorded_observation']).read_text()))
            self.assertEqual(payload, body['source_policy_view']['payload'])
            self.assertTrue(provenance['actual_saved_target_images'])
            self.assertFalse(provenance['target_online_images'])
            self.assertFalse(provenance['task_prompt_matches_target_task'])
            self.assertEqual(provenance['target_provenance'], body['provenance'])
            self.assertEqual(provenance['native_time_s'], 3703*.002)
            self.assertFalse(provenance['target_embodiment_qualified'])
            self.assertNotIn('PRIVILEGED', json.dumps(payload))
            self.assertEqual(provenance['target_provenance']['camera_equivalence'], False)

    def test_source_packet_legacy_mode_unchanged_but_bound_target_cannot_drop_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = capture_fixture(Path(tmp))
            target = json.loads(Path(spec['recorded_observation']).read_text())
            payload = unseal(target)['source_policy_view']['payload']
            source = Path(tmp)/'SOURCE.json'
            put(source, sealed(dict(payload=payload)))
            actual, provenance = recorded_payload(dict(recorded_observation=str(source),
                recorded_observation_sha256=digest(source)))
            self.assertEqual(actual, payload)
            self.assertFalse(provenance['actual_saved_target_images'])
            with self.assertRaises(KeyError):
                recorded_payload(dict(spec, observation_mode='ARCHIVED_SOURCE_RELAY_PACKET'))
            with self.assertRaises(ValueError):
                recorded_payload(dict(spec, observation_mode='TARGET_ONLINE'))

    def test_changed_pixels_open_or_failed_producer_and_wrong_view_reject(self):
        changes = ('pixel', 'open', 'failed', 'wrong-view', 'result-hash', 'language-match',
                   'bound-hash', 'wrong-raw-tick', 'fully-rehashed-wrong-raw-tick',
                   'missing-file-binding', 'producer-scene', 'producer-map')
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                spec = capture_fixture(root)
                if change == 'pixel':
                    (root/'fit-01/images/000001-body_rgb.png').write_bytes(b'CHANGED')
                elif change == 'open':
                    p = root/'control/LEASE.json'; v = json.loads(p.read_text()); v['status'] = 'ACTIVE'; put(p,v)
                elif change == 'failed':
                    put(root/'WORKER_EXIT.private.json', dict(exit_code=1))
                elif change == 'wrong-view':
                    spec['target_capture_step'] = 3503
                elif change == 'result-hash':
                    spec['target_capture_result_sha256'] = '0'*64
                elif change == 'language-match':
                    p = root/'EXECUTION_SPEC.json'; v = json.loads(p.read_text()); v['task_prompt_matches_target_task'] = True; put(p,v)
                    p = root/'control/LEASE.json'; v = json.loads(p.read_text()); v['spec_sha256'] = digest(root/'EXECUTION_SPEC.json'); put(p,v)
                elif change == 'bound-hash':
                    spec['recorded_observation_sha256'] = '0'*64
                elif change in ('producer-scene', 'producer-map'):
                    p = root/'EXECUTION_SPEC.json'; v = json.loads(p.read_text())
                    if change == 'producer-scene':
                        v['original']['scene']['sha256'] = '9'*64
                    else:
                        v['frame_map']['T_source_from_target'][0][3] += 1.
                    put(p,v)
                    p = root/'control/LEASE.json'; v = json.loads(p.read_text()); v['spec_sha256'] = digest(root/'EXECUTION_SPEC.json'); put(p,v)
                else:
                    p = root/'fit-01/RESULT.private.json'; v = json.loads(p.read_text())
                    if change in ('wrong-raw-tick', 'fully-rehashed-wrong-raw-tick'):
                        rawpath = root/'fit-01/STATE_3703.private.bin'
                        raw = bytearray(rawpath.read_bytes()); struct.pack_into('<Q', raw, 8, 3503); rawpath.write_bytes(raw)
                        v['captures'][1]['files'][rawpath.name] = digest(rawpath)
                        if change == 'fully-rehashed-wrong-raw-tick':
                            v['captures'][1]['native_state_sha256'] = digest(rawpath)
                            boundpath = Path(spec['recorded_observation'])
                            body = unseal(json.loads(boundpath.read_text()))
                            body['target_observation']['native_state_sha256'] = digest(rawpath)
                            put(boundpath, sealed(body))
                            spec['recorded_observation_sha256'] = digest(boundpath)
                            v['captures'][1]['files'][boundpath.name] = digest(boundpath)
                    else:
                        del v['captures'][1]['files']['images/000001-body_rgb.png']
                    put(p,v); spec['target_capture_result_sha256'] = digest(p)
                with self.assertRaises((ValueError, PermissionError)):
                    recorded_payload(spec)


if __name__ == '__main__':
    unittest.main()
