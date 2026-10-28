import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import uuid

from PIL import Image

from .eval_contract import FRAME, TCP
from .relay_protocol import PROTOCOL, RelayExchange, sealed, unseal


class TestPausedRelayProtocol(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.identity = dict(profile='B0', model_revision='a' * 40, source_sha='b' * 40,
            config_sha256='c' * 64, processor_sha256='d' * 64, statistics_sha256='e' * 64)
        self.run, self.episode = 'synthetic-no-physics', str(uuid.uuid4())
        self.exchange = self.create(self.root / 'exchange')
        out = io.BytesIO()
        Image.new('RGB', (256, 256), (20, 40, 60)).save(out, format='PNG')
        raw = out.getvalue()
        self.images = [dict(camera=camera, png_base64=base64.b64encode(raw).decode(),
                            sha256=hashlib.sha256(raw).hexdigest())
                       for camera in ('external_rgb', 'wrist_rgb')]
        self.state = [.4, 0., .5, 1., 0., 0., 0., 1., 0., 1.]

    def tearDown(self):
        self.exchange.close()
        self.temporary.cleanup()

    def create(self, root):
        return RelayExchange(root, run_uuid=self.run, episode_uuid=self.episode,
                             identity=self.identity, now=100.)

    def packet(self, kind, payload, sequence=0, tick=0):
        return sealed(dict(protocol=PROTOCOL, kind=kind, run_uuid=self.run,
            episode_uuid=self.episode, sequence=sequence, tick=tick, payload=payload))

    def observation(self, sequence=0, tick=0, snapshot=None):
        return self.packet('observation', dict(snapshot_uuid=snapshot or str(uuid.uuid4()),
            language='Pick up the object and place it in the target tray.',
            state_eef10=self.state.copy(), images=copy.deepcopy(self.images), frame=FRAME, tcp=TCP), sequence, tick)

    def response(self, request):
        body = unseal(request)
        payload = {key:body[key] for key in ('query_uuid', 'reference_uuid', 'observation_sha256',
                                            'anchor_sha256', 'identity')}
        payload.update(trajectory_eef10=[self.state.copy() for _ in range(40)],
                       offsets_ns=[i * 50_000_000 for i in range(40)])
        return self.packet('action', payload, body['sequence'], body['tick'])

    def ack(self, command):
        body = unseal(command)
        value = {key:body[key] for key in ('query_uuid', 'reference_uuid', 'prefix_sha256')}
        value.update(command_sha256=command['sha256'], tick_after=body['tick'] + 200,
                     integrations=200, physics_receipt_sha256='f' * 64)
        return self.packet('ack', value, body['sequence'], body['tick'])

    def assert_stopped_without_progress(self):
        state = json.loads(self.exchange.path.read_text())
        self.assertEqual(state['phase'], 'STOPPED')
        self.assertEqual(state['tick'], 0)
        self.assertEqual(state['sequence'], 0)
        self.assertEqual(state['history'], [])

    def test_one_prefix_only_and_fresh_observation_after_ack(self):
        observation = self.observation()
        request = self.exchange.observe(observation, 101.)
        self.assertEqual(unseal(request)['payload'], unseal(observation)['payload'])
        command = self.exchange.action(self.response(request), 102.)
        self.assertEqual(len(unseal(command)['prefix_eef10']), 8)
        self.assertNotIn('trajectory_eef10', unseal(command))
        self.assertEqual(unseal(command)['advance_integrations'], 200)
        self.assertEqual(self.exchange.state['tick'], 0)
        self.exchange.acknowledge(self.ack(command), 103.)
        fresh = self.exchange.observe(self.observation(1, 200), 104.)
        self.assertNotEqual(unseal(fresh)['observation_sha256'], unseal(request)['observation_sha256'])
        self.assertEqual(unseal(fresh)['tick'], 200)

    def test_crc_corruption_never_advances(self):
        packet = self.observation()
        packet['body']['payload']['language'] = 'corrupted'
        with self.assertRaisesRegex(ValueError, 'HASH_MISMATCH'):
            self.exchange.observe(packet, 101.)
        self.assert_stopped_without_progress()

    def test_privileged_student_fields_are_rejected_even_with_valid_hash(self):
        body = unseal(self.observation())
        body['payload']['goal_position'] = [1., 2., 3.]
        with self.assertRaisesRegex(ValueError, 'UNEXPECTED_PACKET_FIELDS'):
            self.exchange.observe(sealed(body), 101.)
        self.assert_stopped_without_progress()

    def test_rgb_hash_and_camera_order_are_checked(self):
        for mode in ('hash', 'order', 'extra_metadata'):
            with self.subTest(mode=mode):
                exchange = self.create(self.root / mode)
                body = unseal(self.observation())
                if mode == 'hash':
                    body['payload']['images'][0]['sha256'] = '0' * 64
                elif mode == 'order':
                    body['payload']['images'].reverse()
                else:
                    body['payload']['images'][0]['teacher_stage'] = 'close'
                try:
                    with self.assertRaises(ValueError):
                        exchange.observe(sealed(body), 101.)
                    self.assertEqual(exchange.state['tick'], 0)
                finally:
                    exchange.close()

    def test_stale_and_reordered_observations_rejected(self):
        for sequence, tick in ((1, 0), (0, 200), (True, 0)):
            with self.subTest(sequence=sequence, tick=tick):
                exchange = self.create(self.root / (str(sequence) + '-' + str(tick)))
                try:
                    with self.assertRaises(ValueError):
                        exchange.observe(self.observation(sequence, tick), 101.)
                    self.assertEqual(exchange.state['phase'], 'STOPPED')
                finally:
                    exchange.close()

    def test_image_resolution_matches_frozen_renderer_without_resizing(self):
        from scripts.pi0291_arx5_data.config import PARAMETERS
        from .relay_protocol import RGB_SIZE
        self.assertEqual(RGB_SIZE, (PARAMETERS['pixels'], PARAMETERS['pixels']))
        body = unseal(self.observation())
        out = io.BytesIO()
        Image.new('RGB', (640, 480)).save(out, format='PNG')
        raw = out.getvalue()
        body['payload']['images'][0].update(png_base64=base64.b64encode(raw).decode(),
                                           sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError, 'ORIGINAL_RGB_FORMAT_REQUIRED'):
            self.exchange.observe(sealed(body), 101.)
        self.assert_stopped_without_progress()

    def test_duplicate_observation_cannot_query_twice(self):
        packet = self.observation()
        self.exchange.observe(packet, 101.)
        with self.assertRaisesRegex(ValueError, 'OBSERVATION_NOT_EXPECTED'):
            self.exchange.observe(packet, 102.)
        self.assert_stopped_without_progress()

    def test_wrong_action_anchor_or_identity_never_installs(self):
        for key in ('anchor_sha256', 'observation_sha256', 'query_uuid', 'identity'):
            with self.subTest(key=key):
                exchange = self.create(self.root / key)
                try:
                    request = exchange.observe(self.observation(), 101.)
                    body = unseal(self.response(request))
                    body['payload'][key] = 'wrong'
                    with self.assertRaises(ValueError):
                        exchange.action(sealed(body), 102.)
                    self.assertEqual(exchange.state['tick'], 0)
                    self.assertFalse((exchange.root / 'command-000.json').exists())
                finally:
                    exchange.close()

    def test_frozen_gate_reject_is_retained_not_repaired(self):
        request = self.exchange.observe(self.observation(), 101.)
        body = unseal(self.response(request))
        body['payload']['trajectory_eef10'][0][9] = 1.1
        with self.assertRaisesRegex(ValueError, 'FROZEN_SAFETY_GATE_REJECT'):
            self.exchange.action(sealed(body), 102.)
        evidence = json.loads((self.exchange.root / 'action-000.json').read_text())
        self.assertEqual(evidence['packet']['body']['payload']['trajectory_eef10'][0][9], 1.1)
        self.assertEqual(evidence['gate']['decision'], 'REJECT')
        self.assert_stopped_without_progress()

    def test_timeout_and_episode_deadline_stop_without_progress(self):
        self.exchange.observe(self.observation(), 101.)
        with self.assertRaisesRegex(ValueError, 'QUERY_TIMEOUT'):
            self.exchange.check_timeout(161.)
        self.assert_stopped_without_progress()
        other = self.create(self.root / 'deadline')
        try:
            with self.assertRaisesRegex(ValueError, 'EPISODE_TIMEOUT'):
                other.observe(self.observation(), 1000.)
            self.assertEqual(other.state['tick'], 0)
        finally:
            other.close()

    def test_restart_retains_reserved_action_and_cannot_execute_again(self):
        request = self.exchange.observe(self.observation(), 101.)
        response = self.response(request)
        self.exchange.action(response, 102.)
        self.exchange.close()
        self.exchange = RelayExchange(self.root / 'exchange')
        with self.assertRaisesRegex(ValueError, 'ALREADY_RESERVED'):
            self.exchange.action(response, 103.)
        self.assert_stopped_without_progress()

    def test_reconnect_may_read_original_ack_without_replaying_action(self):
        request = self.exchange.observe(self.observation(), 101.)
        command = self.exchange.action(self.response(request), 102.)
        self.exchange.close()
        self.exchange = RelayExchange(self.root / 'exchange')
        self.exchange.acknowledge(self.ack(command), 103.)
        self.assertEqual(self.exchange.state['tick'], 200)
        self.assertEqual(len(self.exchange.state['history']), 1)
        with self.assertRaisesRegex(ValueError, 'ACK_NOT_EXPECTED'):
            self.exchange.acknowledge(self.ack(command), 104.)
        self.assertEqual(self.exchange.state['tick'], 200)
        self.assertEqual(len(self.exchange.state['history']), 1)

    def test_wrong_ack_never_advances_owner_tick(self):
        request = self.exchange.observe(self.observation(), 101.)
        command = self.exchange.action(self.response(request), 102.)
        body = unseal(self.ack(command))
        body['payload']['integrations'] = 400
        with self.assertRaisesRegex(ValueError, 'PHYSICS_PROGRESS_MISMATCH'):
            self.exchange.acknowledge(sealed(body), 103.)
        self.assert_stopped_without_progress()

    def test_same_snapshot_after_ack_is_not_a_fresh_observation(self):
        observation = self.observation()
        request = self.exchange.observe(observation, 101.)
        command = self.exchange.action(self.response(request), 102.)
        self.exchange.acknowledge(self.ack(command), 103.)
        packet = self.observation(1, 200, unseal(observation)['payload']['snapshot_uuid'])
        with self.assertRaisesRegex(ValueError, 'SNAPSHOT_REUSE'):
            self.exchange.observe(packet, 104.)
        self.assertEqual(self.exchange.state['tick'], 200)

    def test_duplicate_owner_and_existing_episode_restart_refused(self):
        with self.assertRaises(BlockingIOError):
            RelayExchange(self.root / 'exchange')
        self.exchange.close()
        with self.assertRaises(FileExistsError):
            self.create(self.root / 'exchange')
        self.exchange = RelayExchange(self.root / 'exchange')

    def test_no_automatic_recovery_after_stop_or_clock_reversal(self):
        self.exchange.observe(self.observation(), 101.)
        with self.assertRaisesRegex(ValueError, 'WALL_CLOCK_REVERSED'):
            self.exchange.check_timeout(100.5)
        first = self.exchange.state['failure']
        with self.assertRaises(ValueError):
            self.exchange.observe(self.observation(), 102.)
        self.assertEqual(self.exchange.state['failure'], first)
        self.assert_stopped_without_progress()

    def test_full_synthetic_episode_is_exactly_80_prefixes_and_no_81st_query(self):
        for sequence in range(80):
            now = 101. + sequence * 3
            request = self.exchange.observe(self.observation(sequence, sequence * 200), now)
            command = self.exchange.action(self.response(request), now + 1)
            self.exchange.acknowledge(self.ack(command), now + 2)
        self.assertEqual(self.exchange.state['phase'], 'DONE')
        self.assertEqual(self.exchange.state['tick'], 16000)
        self.assertEqual(len(self.exchange.state['history']), 80)
        with self.assertRaisesRegex(ValueError, 'OBSERVATION_NOT_EXPECTED'):
            self.exchange.observe(self.observation(80, 16000), 342.)
        self.assertEqual(self.exchange.state['tick'], 16000)

    def test_lost_ack_records_unknown_physics_and_forbids_replay(self):
        request = self.exchange.observe(self.observation(), 101.)
        self.exchange.action(self.response(request), 102.)
        with self.assertRaisesRegex(ValueError, 'QUERY_TIMEOUT'):
            self.exchange.check_timeout(161.)
        self.assertEqual(self.exchange.state['unacknowledged_physics'], 'UNKNOWN_DO_NOT_RESEND_COMMAND')
        self.assertEqual(self.exchange.state['tick'], 0)  # Last acknowledged tick, not physical proof.

    def test_protocol_import_has_no_tensor_or_physics_engine(self):
        import subprocess
        import sys
        code = ('import sys; import scripts.pi0292.relay_protocol; '
                'assert not any(k == m or k.startswith(m + ".") '
                'for k in sys.modules for m in ("torch", "mujoco", "transformers", "gr00t"))')
        subprocess.run([sys.executable, '-c', code], check=True)


if __name__ == '__main__':
    unittest.main()
