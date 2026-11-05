import json
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
from go2arx5_vla.world_model.pilot1_contact import ContactReader, exit_score, validate, PHASE
from go2arx5_vla.world_model.next1_episode import RULE
from go2arx5_vla.world_model.storage import digest


def packet(step, force, *, other=False):
    raw = struct.pack('<8sQdIII', b'GRTSTATE', step, step*.002, 34, 32, 20)
    raw += np.zeros(86, dtype='<f8').tobytes() + struct.pack('<I', 1)
    raw += struct.pack('<IId', 99 if other else 7, 9, -.0001)
    rest = np.zeros(18, dtype='<f8'); rest[0] = force
    raw += rest.tobytes()
    return struct.pack('<I', len(raw)) + raw


class ActualContactIntervals(unittest.TestCase):
    def test_native_force_identity_and_2ms_timestamp_not_geometry_or_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'journal'
            path.write_bytes(b''.join(packet(s, .1 if s == 2 else .049, other=s == 3) for s in range(1, 26)))
            reader = ContactReader(path, [7, 8], 9)
            events = reader.interval(0, 25)
            self.assertEqual(sum(e['contact_positive'] for e in events), 1)
            self.assertFalse(events[-1]['contact_positive'])
            self.assertAlmostEqual(events[1]['contact_time_s'], .002)
            self.assertEqual(reader.offset, path.stat().st_size)

    def test_missing_tick_and_incomplete_tail_are_recording_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'journal'
            for data in [packet(1, .1)+packet(3, .1), packet(1, .1)+packet(2, .1)[:-3]]:
                path.write_bytes(data)
                with self.assertRaises(ValueError): ContactReader(path, [7, 8], 9).interval(0, 2)

    def test_exit_requires_actual_entry_100_native_ticks_away_motion_and_support(self):
        events = [dict(native_step=s, contact_positive=s <= 4103) for s in range(4080, 4204)]
        self.assertTrue(exit_score(events, [0, 0, 0], [0, .0002, 0], True)['qualified_contact_exit'])
        self.assertFalse(exit_score(events[:-1], [0, 0, 0], [0, .0002, 0], True)['qualified_contact_exit'])
        self.assertFalse(exit_score(events, [0, 0, 0], [0, .00019, 0], True)['qualified_contact_exit'])
        self.assertFalse(exit_score(events, [0, 0, 0], [0, .001, 0], False)['qualified_contact_exit'])
        self.assertFalse(exit_score(events[24:], [0, 0, 0], [0, .001, 0], True)['qualified_contact_exit'])
        events[-2]['contact_positive'] = True
        self.assertFalse(exit_score(events, [0, 0, 0], [0, .001, 0], True)['qualified_contact_exit'])

    def test_safety_pass_without_real_entry_cannot_admit_withdraw_curriculum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save(name, value):
                p = root/name; p.write_text(json.dumps(value)); return str(p)
            model = save('model', {}); binary = save('binary', {})
            native = save('RESULT.json', dict(status='NATIVE_STAGE_PASSED', shutdown_clean=True, physics_stops=[], scene_sha256=digest(model)))
            save('STEADY_PREFIX_RESULT.private.json', dict(qualified=True))
            contact = save('contact', dict(native_result_sha256=digest(native), scene_sha256=digest(model), contact_positive_native_ticks=0, contact_force_threshold_N=.05))
            assets = save('assets', dict(native_binary_sha256=digest(binary), scenes=[dict(scene_id='C', parent='P', split='TRAIN', model_path=model, model_sha256=digest(model))]))
            episode = save('episode', dict(schema='WM02_PILOT1_TOUCH_WITHDRAW_V1', phase=PHASE, reference_rule=RULE, prehold_atoms=40, screen_atom=48, touch_end_atom=64, task_atoms=96, withdrawal_xyz_m=[0., .001, 0.], contact_force_threshold_N=.05, required_exit_s=.2, minimum_away_Y_m=.0002, purpose='CONTACT_EXIT_COMMISSION', run=str(root), split='TRAIN', scene_id='C', parent='P'))
            spec = dict(episode_manifest=episode, episode_manifest_sha256=digest(episode), registered_root=str(root), pilot1_binding='fixture', scene_allowlist=assets, scene_allowlist_sha256=digest(assets), model_path=model, native_binary=binary, scene_qualification=native, scene_qualification_sha256=digest(native), contact_entry_evidence=contact, contact_entry_evidence_sha256=digest(contact))
            previous = dict(status='NATIVE_STAGE_PASSED', shutdown_clean=True, native_binary_sha256=digest(binary))
            with self.assertRaisesRegex(PermissionError, 'REAL_SAFE_NATIVE_CONTACT_ENTRY'): validate(spec, previous)
            value = json.loads(Path(contact).read_text()); value['contact_positive_native_ticks'] = 1
            Path(contact).write_text(json.dumps(value)); spec['contact_entry_evidence_sha256'] = digest(contact)
            self.assertEqual(validate(spec, previous), 4903)


if __name__ == '__main__': unittest.main()
