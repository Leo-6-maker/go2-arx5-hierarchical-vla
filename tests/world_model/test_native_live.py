import base64
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest

import numpy as np

from go2arx5_vla.world_model.live_source import NativeSource, header
from go2arx5_vla.world_model.native_live import LiveReceiver, ObserverTransport


def packet(step=1, q=0.):
    values = np.zeros(86, dtype='<f8')
    values[0] = q
    return (struct.pack('<8sQdIII', b'GRTSTATE', step, step * .002, 34, 32, 20)
            + values.tobytes() + struct.pack('<I', 0))


class Live(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'episode'
        self.output = self.root / 'native-01'
        self.output.mkdir(parents=True)
        self.model = self.root / 'scene.mjb'
        self.model.write_bytes(b'test-model-not-a-native-model')
        self.binary = self.root / 'binary'
        self.binary.write_bytes(b'test-binary-never-executed')
        self.lease = self.root / 'LEASE.json'
        self.spec = self.root / 'EXECUTION_SPEC.json'
        self.spec.write_text(json.dumps(dict(registered_root=str(self.root), source_sha='a' * 40,
             model_path=str(self.model), native_binary=str(self.binary), lease=str(self.lease),
             frozen_files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.model, self.binary)},
             deadline_epoch=time.time() + 3600, phase='TEST_ONLY')))
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        self.remote = dict(execution_spec=dict(path=str(self.spec), sha256=sha(self.spec)),
                           native_output=str(self.output), episode_id='episode', source_sha='a' * 40,
                           scene_sha256=sha(self.model), native_binary_sha256=sha(self.binary))
        self.set_lease('ACTIVE')
        (self.output / 'state.bin').write_bytes(packet())
        self.config = dict(episode_id='episode', source_sha='a' * 40, ngeom=100,
                           scene=dict(path=str(self.model), sha256=sha(self.model)), remote=self.remote)

    def set_lease(self, status):
        self.lease.write_text(json.dumps(dict(status=status, run_uuid='episode',
                                spec_sha256=self.remote['execution_spec']['sha256'])))

    def receive(self, source, receiver, now):
        receiver.accept(json.dumps(source.snapshot()).encode(), now=now)

    def test_live_requires_actual_advance_and_detects_pause_disconnect_end(self):
        source, receiver = NativeSource(self.remote), LiveReceiver(self.config)
        self.receive(source, receiver, 0.)
        self.assertEqual(receiver.view(now=0.)['status'], 'WAITING_FOR_ADVANCE')
        (self.output / 'state.bin').write_bytes(packet(2))
        self.receive(source, receiver, .1)
        self.assertEqual(receiver.view(now=.1)['status'], 'LIVE')
        self.receive(source, receiver, 1.5)
        self.assertEqual(receiver.view(now=1.5)['status'], 'PAUSED')
        self.assertEqual(receiver.view(now=3.)['status'], 'DISCONNECTED')
        receiver.closed = True
        self.assertEqual(receiver.view(now=3.)['status'], 'STREAM_ENDED')

    def test_closed_lease_cannot_be_live_even_if_file_changes(self):
        self.set_lease('CLOSED')
        source, receiver = NativeSource(self.remote), LiveReceiver(self.config)
        self.receive(source, receiver, 0.)
        (self.output / 'state.bin').write_bytes(packet(2))
        self.receive(source, receiver, .1)
        self.assertEqual(receiver.view(now=.1)['status'], 'SEALED')

    def test_readonly_source_preserves_files_and_uses_acknowledged_goal(self):
        (self.output / 'COMMANDS.private.jsonl').write_text('\n'.join(json.dumps(r) for r in [
            dict(phase='proposed', wire_sha256='a', action_EEF10=[[1, 2, 3]]),
            dict(phase='installed', wire_sha256='a'),
            dict(phase='proposed', wire_sha256='b', action_EEF10=[[9, 9, 9]])]) + '\n')
        paths = list(self.root.rglob('*'))
        before = {p: p.read_bytes() for p in paths if p.is_file()}
        source = NativeSource(self.remote)
        value = source.snapshot()
        self.assertEqual(value['command']['installed_horizon_goal_m'], [1, 2, 3])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(set(paths), set(self.root.rglob('*')))

    def test_binding_and_state_replacement_fail_closed(self):
        with self.assertRaises(ValueError):
            NativeSource(dict(self.remote, native_output=str(self.root)))
        source = NativeSource(self.remote)
        source.snapshot()
        (self.output / 'state.bin').write_bytes(packet(1, q=1.))
        with self.assertRaises(ValueError):
            source.snapshot()
        (self.output / 'state.bin').write_bytes(packet(0))
        with self.assertRaises(ValueError):
            source.snapshot()
        self.spec.write_text('{}')
        with self.assertRaises(ValueError):
            NativeSource(self.remote)

    def test_rejects_corruption_wrong_identity_sequence_and_clock(self):
        source, receiver = NativeSource(self.remote), LiveReceiver(self.config)
        value = source.snapshot()
        for changes in (dict(scene_sha256='other'), dict(packet_sha256='bad'), dict(sequence=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                receiver.accept(json.dumps(dict(value, **changes)))
        receiver.accept(json.dumps(value), now=0.)
        with self.assertRaises(ValueError):
            receiver.accept(json.dumps(value), now=.1)
        for raw in (packet()[:-1], packet() + b'x', packet(q=float('nan'))):
            with self.assertRaises(ValueError):
                header(raw)
        raw = bytearray(packet())
        struct.pack_into('<d', raw, 16, .003)
        with self.assertRaises(ValueError):
            header(raw)

    def test_missing_startup_packet_and_missing_lease_are_explicit(self):
        (self.output / 'state.bin').unlink()
        self.lease.unlink()
        source, receiver = NativeSource(self.remote), LiveReceiver(self.config)
        self.receive(source, receiver, 0.)
        self.assertIsNone(receiver.view(now=0.)['row'])
        self.assertEqual(receiver.view(now=0.)['status'], 'OWNER_UNKNOWN')

    def test_real_subprocess_transport_observes_without_touching_run(self):
        # Exercise the same stdin bootstrap/stdout receiver used by SSH, without SSH in unit tests.
        self.set_lease('CLOSED')
        receiver = LiveReceiver(self.config)
        config = dict(self.config, observer_command=[sys.executable, '-'])
        transport = ObserverTransport(config, receiver, 1., 10.)
        try:
            deadline = time.monotonic() + 5
            while not receiver.closed and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertGreaterEqual(receiver.frames, 5)
            self.assertEqual(receiver.advances, 0)
            self.assertIsNone(receiver.error)
            self.assertEqual((self.output / 'state.bin').read_bytes(), packet())
            self.assertEqual(json.loads(self.lease.read_text())['status'], 'CLOSED')
        finally:
            transport.close()
        self.assertEqual(transport.process.returncode, 0)

    def test_binding_preparation_reads_original_spec_and_reuses_exact_asset_cache(self):
        from scripts.wm02.prepare_native_live import prepare
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        first = Path(self.tmp.name) / 'config1.json'
        cache = Path(self.tmp.name) / 'cache'
        result = prepare([sys.executable, '-'], self.spec, cache, first)
        self.assertGreater(result['compressed_transfer_bytes'], 0)
        config = json.loads(first.read_text())
        self.assertEqual(Path(config['scene']['path']).read_bytes(), self.model.read_bytes())
        self.assertEqual(config['remote']['execution_spec']['sha256'], self.remote['execution_spec']['sha256'])
        second = prepare([sys.executable, '-'], self.spec, cache, Path(self.tmp.name) / 'config2.json')
        self.assertEqual(second['compressed_transfer_bytes'], 0)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        with self.assertRaises(FileExistsError):
            prepare([sys.executable, '-'], self.spec, cache, first)


if __name__ == '__main__':
    unittest.main()
