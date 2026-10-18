"""Inventory fixtures, never a renderer, engine, upload or network call."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.pi0294.fd2 import pilot_plan
from scripts.pi0294.fd2_archive import inventory


class ArchiveTests(unittest.TestCase):
    def fixture(self,root):
        def put(path,v):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v))
        plan=pilot_plan([]);put(root/'plans/PILOT_PLAN.json',plan)
        put(root/'evidence/PILOT_RESULT.private.json',dict(status='TERMINAL_EIGHT_CELLS',rows=[{}]*8))
        for i,c in enumerate(plan['cards']):
            folder=root/'episodes'/c['id'];out=root/'evidence/audits'/c['id'];out.mkdir(parents=True)
            result=dict(status='VALID_COMPLETE' if i%2 else 'FAILED_PRESERVED',physics_steps=16000 if i%2 else 200,rgb_pairs=641 if i%2 else 9)
            put(folder/'RESULT.json',result);movie=out/'actual-body-wrist.mp4';movie.write_bytes(b'fixture'+str(i).encode())
            h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            put(out/'AUDIT.private.json',dict(status='PASS_INDEPENDENT_SAVED_BYTE_AUDIT',errors=[],checks={'fixture':True},
                result_sha256=h(folder/'RESULT.json'),score={'persistent_success':bool(i%2)},S1={'pre_execution_rejections':int(i%2==0),'live_native_failures':0},
                video=dict(relative_path=str(movie.relative_to(root)),bytes=movie.stat().st_size,sha256=h(movie),
                    decoded_frames=result['rgb_pairs'],actual_paired_frames=result['rgb_pairs'],padding_or_interpolation=False)))
        return plan

    def test_all_eight_keep_failures_and_partial_durations(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root);index,files,sources=inventory(root)
            self.assertEqual(len(index['attempts']),8)
            self.assertEqual(sum(r['task_success'] for r in index['attempts']),4)
            self.assertEqual(sum(r['S1_rejections'] for r in index['attempts']),4)
            self.assertEqual(len(files),8);self.assertEqual(set(files),set(sources))
            self.assertEqual(index['attempts'][0]['native_sim_s'],.4)
            self.assertFalse(index['GR00T_inference'])

    def test_incomplete_population_is_not_a_success_subset_archive(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root)
            p=root/'evidence/PILOT_RESULT.private.json';v=json.loads(p.read_text());v['rows'].pop();p.write_text(json.dumps(v))
            with self.assertRaises(PermissionError):inventory(root)

    def test_changed_video_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);plan=self.fixture(root)
            (root/'evidence/audits'/plan['cards'][0]['id']/'actual-body-wrist.mp4').write_bytes(b'changed')
            with self.assertRaises(ValueError):inventory(root)

    def test_missing_audit_check_is_not_qualified(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);plan=self.fixture(root);p=root/'evidence/audits'/plan['cards'][0]['id']/'AUDIT.private.json'
            v=json.loads(p.read_text());v['checks']['fixture']=False;p.write_text(json.dumps(v))
            with self.assertRaises(PermissionError):inventory(root)


if __name__=='__main__':unittest.main()
