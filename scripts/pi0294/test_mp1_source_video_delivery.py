import json
from pathlib import Path
import tempfile
import unittest

from scripts.pi0294.native_dataset import sha
from .mp1.source_video_delivery import IDS,PANEL,audited_movies


def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))


def fixture(root):
    child=root/'children/source-pair-01'
    put(child/'plans/SOURCE_EVALUATION_PLAN.json',dict(schema='MP1_FIRST_SOURCE_PAIR_FIXED_DEVELOPMENT_V1',panels={PANEL:dict(attempt_ids=IDS,checkpoint_step=2048,split='TRAIN',teacher_takeover=False)}))
    put(child/'RESOURCE_LEDGER.json',dict(reservations={'UNRELATED_DEV_CASE':{'maximum':200}}))
    for i,name in enumerate(IDS):
        decision=dict(attempt_id=name,status='VALID_TASK_FAIL' if i==2 else 'SUCCESS',failure_stage='S1_PRE_EXECUTION_REJECT' if i==2 else 'COMPLETE_SUCCESS',model=PANEL,split='TRAIN',pipeline_trustworthy=True,teacher_takeover=False,live_native_failures=0,task='T1' if i%2==0 else 'T2',scene_group='TRAIN/F'+str(i//2)+'/000',duration_s=1.2 if i==2 else 32,RGB_pairs=25 if i==2 else 641,S1_rejections=1 if i==2 else 0)
        put(child/'episodes'/name/'MP1_ADJUDICATION.json',decision)
        put(child/'worker_processes'/(name+'.json'),dict(pid=999999999,proc_start='0'))
        folder=child/'evidence/audits'/name;folder.mkdir(parents=True)
        movie=folder/'actual-body-wrist.mp4';movie.write_bytes(b'unit-fixture-not-a-real-qualified-video')
        put(folder/'AUDIT.private.json',dict(status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT',attempt_id=name,adjudication=decision,checks=dict(raw=True),errors=[],video=dict(sha256=sha(movie),bytes=movie.stat().st_size,decoded_frames=decision['RGB_pairs'],actual_paired_frames=decision['RGB_pairs'],native_duration_s=decision['duration_s'],padding_or_interpolation=False)))
    return child


class SourceVideoTests(unittest.TestCase):
    def test_exact_four_successes_and_negative_while_unrelated_DEV_is_reserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);before={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
            rows,files,sources=audited_movies(root)
            self.assertEqual(len(rows),4);self.assertEqual(sum(x['task_success'] for x in rows),3)
            self.assertEqual(len(files),len(sources));self.assertEqual(rows[2]['actual_frames'],25)
            self.assertEqual(before,{str(p):sha(p) for p in root.rglob('*') if p.is_file()})

    def test_selected_case_reservation_blocks_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=fixture(root);put(child/'RESOURCE_LEDGER.json',dict(reservations={IDS[0]:{'maximum':200}}))
            with self.assertRaisesRegex(PermissionError,'STILL_RESERVED'):audited_movies(root)

    def test_changed_movie_or_missing_raw_check_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=fixture(root);audit=child/'evidence/audits'/IDS[0]/'AUDIT.private.json';v=json.loads(audit.read_text());v['checks']['raw']=False;put(audit,v)
            with self.assertRaises(PermissionError):audited_movies(root)
            v['checks']['raw']=True;put(audit,v);(audit.parent/'actual-body-wrist.mp4').write_bytes(b'changed')
            with self.assertRaises(ValueError):audited_movies(root)

    def test_DEV_relabel_or_untrusted_outcome_is_not_a_TRAIN_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=fixture(root);p=child/'episodes'/IDS[0]/'MP1_ADJUDICATION.json';v=json.loads(p.read_text());v['split']='DEV';put(p,v)
            with self.assertRaises(PermissionError):audited_movies(root)


if __name__=='__main__':unittest.main()
