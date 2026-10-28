"""Owner binding unit fixtures only, never physical/dataset qualification."""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import finalize_raw_bank as finalizer


class FinalizerTests(unittest.TestCase):
    def manifest_fixture(self, root):
        (root/'samples.json').write_bytes(b'unit fixture, not episode data')
        (root/'FILE_MANIFEST.json').write_text(json.dumps([
            dict(path='samples.json', sha256=finalizer.sha(root/'samples.json'))]))

    def test_complete_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.manifest_fixture(root)
            self.assertEqual(finalizer.verified_manifest(root),finalizer.sha(root/'FILE_MANIFEST.json'))

    def test_corrupt_manifest_member_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.manifest_fixture(root)
            (root/'samples.json').write_bytes(b'changed fixture')
            with self.assertRaises(AssertionError):
                finalizer.verified_manifest(root)

    def test_unlisted_member_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.manifest_fixture(root)
            (root/'unexpected.json').write_text('{}')
            with self.assertRaises(AssertionError):
                finalizer.verified_manifest(root)

    def test_unattempted_preserves_uint64_and_denominator(self):
        seed = 9820627790027372738
        row = finalizer.bind_formal(Path('/unused-unit-fixture'),'T291000',
            dict(split='TRAIN',group_id='UNIT/T291000',seed=seed),None,{})
        self.assertEqual(row['seed'],seed)
        self.assertFalse(row['attempted'])
        self.assertFalse(row['admitted'])
        self.assertEqual(row['windows'],0)

    def test_boolean_seed_rejected(self):
        with self.assertRaises(AssertionError):
            finalizer.bind_formal(Path('/unused-unit-fixture'),'T291000',
                dict(split='TRAIN',group_id='UNIT/T291000',seed=True),None,{})

    def test_all80_unattempted_terminal_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'evidence').mkdir()
            (root/'code').mkdir()
            source = root/'code/unit_fixture.py'
            source.write_bytes(b'unit fixture; never executed')
            ids = [f'T291{i:03d}' for i in range(64)]+[f'D291{i:03d}' for i in range(16)]
            layouts = {identifier:dict(id=identifier,seed=2**63+i,group_id='UNIT/'+identifier,
                split='TRAIN' if identifier.startswith('T') else 'DEV') for i,identifier in enumerate(ids)}
            seal = dict(source_sha=finalizer.SOURCE,authority=finalizer.AUTHORITY,formal_order=ids,layouts=layouts,
                code_files={'unit_fixture.py':finalizer.sha(source)})
            (root/'TEACHER_AND_LAYOUT_SEAL.json').write_text(json.dumps(seal))
            seal_sha = finalizer.sha(root/'TEACHER_AND_LAYOUT_SEAL.json')
            (root/'RESOURCE_ACCOUNTING.json').write_text(json.dumps(dict(starts=[],CPU_s=0,unmetered_dependency_setup_CPU_s_upper=0)))
            (root/'evidence/OWNER_N3_QUEUE_STATE.json').write_text(json.dumps(dict(physics_running=False,current=None,source_sha=finalizer.SOURCE)))
            gate = dict(status='PASS',all_eight_independent_raw_audits_pass=True,source_sha=finalizer.SOURCE,
                teacher_layout_seal_sha256=seal_sha,denominator=8,physical_successes=8)
            (root/'PILOT_GATE.json').write_text(json.dumps(gate))
            with patch.object(finalizer,'SEAL_SHA',seal_sha), contextlib.redirect_stdout(io.StringIO()):
                finalizer.finalize(root)
            admission = finalizer.read(root/'DATASET_ADMISSION.json')
            self.assertEqual(len(admission['episodes']),80)
            self.assertEqual(admission['counts']['TRAIN']['unattempted'],64)
            self.assertEqual(admission['counts']['DEV']['unattempted'],16)
            self.assertTrue(all(not r['admitted'] for r in admission['episodes']))
            self.assertFalse(admission['complete_attempt_bank'])
            self.assertEqual(admission['dataset_status'],'PARTIAL')
            with patch.object(finalizer,'SEAL_SHA',seal_sha), contextlib.redirect_stdout(io.StringIO()):
                finalizer.finalize(root)

    def test_empty_failed_episode_without_score_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/'episodes/T291000'
            folder.mkdir(parents=True)
            layout = dict(id='T291000',seed=2**63+1,split='TRAIN',group_id='UNIT/T291000')
            card = dict(**layout,source_sha=finalizer.SOURCE,authority=finalizer.AUTHORITY,domain=finalizer.DOMAIN,mode='normal')
            result = dict(id='T291000',source_sha=finalizer.SOURCE,domain=finalizer.DOMAIN,
                status='ENGINEERING_INCOMPLETE',paired_samples=0,physics_steps=0,error='Unit fixture empty initialization failure')
            (folder/'CARD.json').write_text(json.dumps(card))
            (folder/'RESULT.json').write_text(json.dumps(result))
            attempt = dict(source_sha=finalizer.SOURCE,status=result['status'])
            row = finalizer.bind_formal(root,'T291000',layout,attempt,{})
            self.assertTrue(row['attempted'])
            self.assertFalse(row['admitted'])
            self.assertIsNone(row['physical_success'])

    def test_partial_plan_is_resumed_without_replacing_good_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'evidence').mkdir()
            plans = {'first.json':b'first unit fixture','second.json':b'second unit fixture'}
            finalizer.publish_plan(root,plans,[])
            (root/'second.json').unlink()
            old = finalizer.sha(root/'first.json')
            finalizer.publish_plan(root,plans,[])
            self.assertEqual(finalizer.sha(root/'first.json'),old)
            self.assertEqual((root/'second.json').read_bytes(),plans['second.json'])
            (root/'first.json').write_bytes(b'corrupt existing unit fixture')
            with self.assertRaises(AssertionError):
                finalizer.publish_plan(root,plans,[])


if __name__ == '__main__':
    unittest.main()
