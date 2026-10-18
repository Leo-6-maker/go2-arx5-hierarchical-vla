import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.native_dataset import sha
from scripts.pi0294 import test_confirmation_evidence as evidence_fixture
from scripts.pi0294.mp1.confirmation_archive import inventory


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class ConfirmationArchiveTests(unittest.TestCase):
    def setUp(self):
        bound=patch('scripts.pi0294.mp1.confirmation_archive.validate_plan');self.bound=bound.start()
        self.addCleanup(bound.stop)

    def fixture(self,root,*,status='SUCCESS',zero=False):
        factory=evidence_fixture.ConfirmationEvidenceTests();plan=factory.setup_plan();row=factory.row(plan,status=status)
        self.bound.return_value=plan
        if zero:row.update(main_integrations=0,duration_s=0,RGB_pairs=1)
        identifier=row['attempt_id'];raw=root/'episodes'/identifier
        put(raw/'RESULT.json',dict(unit_fixture_not_actual_science=True));row['result_sha256']=sha(raw/'RESULT.json')
        put(raw/'MP1_ADJUDICATION.json',row)
        put(root/'worker_processes'/(identifier+'.json'),dict(pid=999999999,proc_start='0'))
        put(root/'RESOURCE_LEDGER.json',dict(reservations={}))
        folder=root/'evidence'/('zero-step-audits' if zero else 'audits')/identifier
        folder.mkdir(parents=True);movie=folder/('actual-initial-body-wrist.mp4' if zero else 'actual-body-wrist.mp4')
        movie.write_bytes(b'unit-fixture-not-real-video');audit=factory.audit(row)
        audit.update(attempt_id=identifier,checks=dict(raw=True),errors=[],execution_source_sha='a'*40)
        audit['status']='PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE' if zero else audit['status']
        audit['video']=dict(sha256=sha(movie),bytes=movie.stat().st_size,fps=20,decoded_frames=row['RGB_pairs'],
            actual_paired_frames=row['RGB_pairs'],playback_duration_s=row['RGB_pairs']/20,
            native_duration_s=row['duration_s'],relative_path=str(movie.relative_to(root)),padding_or_interpolation=False)
        if zero:audit['video'].update(initial_frame_only=True,task_trajectory_video=False)
        else:audit['video']['initial_frame_included']=True
        put(folder/'AUDIT.private.json',audit)
        return plan,row,folder,audit

    def test_success_negative_and_engineering_clips_preserve_original_classes_without_writes(self):
        for status,zero in [('SUCCESS',False),('VALID_TASK_FAIL',False),('ENGINEERING_INCOMPLETE',False),('ENGINEERING_INCOMPLETE',True)]:
            with self.subTest(status=status,zero=zero),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan,row,_,_=self.fixture(root,status=status,zero=zero)
                before={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
                value,files,sources=inventory(root,plan,[row])
                self.assertEqual(value['actual_videos'],1);self.assertEqual(value['rows'][0]['status'],status)
                self.assertEqual(value['rows'][0]['task_success'],status=='SUCCESS')
                self.assertEqual(value['rows'][0]['initial_frame_only'],zero)
                self.assertEqual(len(files),len(sources));self.assertEqual(value['new_physics'],0)
                self.assertFalse(value['public_upload_performed'])
                self.assertEqual(before,{str(p):sha(p) for p in root.rglob('*') if p.is_file()})

    def test_missing_or_failed_audits_remain_gaps_not_successes(self):
        for defect in ['missing','checks','row','status','errors']:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan,row,folder,audit=self.fixture(root)
                if defect=='missing':(folder/'AUDIT.private.json').unlink()
                else:
                    if defect=='checks':audit['checks']['raw']=False
                    elif defect=='row':audit['adjudication']['duration_s']=1
                    elif defect=='status':audit['status']='ENGINEERING_AUDIT_INCOMPLETE'
                    else:audit['errors']=['raw']
                    put(folder/'AUDIT.private.json',audit)
                value,files,_=inventory(root,plan,[row]);self.assertEqual(value['actual_videos'],0)
                self.assertEqual(value['coverage']['tasks']['T1']['successes'],0)
                self.assertEqual(len(value['audit_gaps']),1);self.assertEqual(files,{})

    def test_active_physics_wrong_raw_and_changed_video_cannot_be_published(self):
        for defect in ['active','raw','video','relative','frames','clock','interpolation','initial']:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan,row,folder,audit=self.fixture(root)
                if defect=='active':put(root/'RESOURCE_LEDGER.json',dict(reservations={row['attempt_id']:dict(maximum=16000)}))
                elif defect=='raw':(root/'episodes'/row['attempt_id']/'RESULT.json').write_text('{}')
                elif defect=='video':(folder/'actual-body-wrist.mp4').write_bytes(b'changed')
                else:
                    if defect=='relative':audit['video']['relative_path']='../elsewhere.mp4'
                    elif defect=='frames':audit['video']['decoded_frames']=1
                    elif defect=='clock':audit['video']['playback_duration_s']=32
                    elif defect=='interpolation':audit['video']['padding_or_interpolation']=True
                    else:audit['video']['initial_frame_included']=False
                    put(folder/'AUDIT.private.json',audit)
                with self.assertRaises((ValueError,PermissionError)):inventory(root,plan,[row])

    def test_symlink_video_refuses_even_with_matching_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,row,folder,_=self.fixture(root);p=folder/'actual-body-wrist.mp4'
            alternate=folder/'alternate.mp4';p.rename(alternate);p.symlink_to(alternate.name)
            with self.assertRaisesRegex(PermissionError,'SYMLINK'):inventory(root,plan,[row])

    def test_unsealed_or_foreign_plan_cannot_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,row,_,_=self.fixture(root);self.bound.return_value={}
            with self.assertRaisesRegex(PermissionError,'SEALED'):inventory(root,plan,[row])
            self.bound.side_effect=PermissionError('native unsealed plan')
            with self.assertRaisesRegex(PermissionError,'unsealed'):inventory(root,plan,[row])


if __name__=='__main__':unittest.main()
