import json
from pathlib import Path
import tempfile
import unittest

from scripts.pi0294.native_dataset import sha
from .mp1.source_video_delivery import IDS,PANEL,DEV_IDS,DEV_PANEL,CORR_IDS,CORR_PANEL,audited_movies,delivery_scope,CORR2048_TRAIN_PANEL,CORR2048_DEV_PANEL,CORR2048_DEV_IDS


def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))


def fixture(root,panel=PANEL):
    scope=delivery_scope(panel);ids=scope['ids'];split=scope['split']
    child=root/'children/source-pair-01'
    put(child/'plans/SOURCE_EVALUATION_PLAN.json',dict(schema='MP1_FIRST_SOURCE_PAIR_FIXED_DEVELOPMENT_V1',panels={panel:dict(attempt_ids=scope['plan_ids'],checkpoint_step=scope['step'],split=split,teacher_takeover=False)}))
    put(child/'RESOURCE_LEDGER.json',dict(reservations={'UNRELATED_DEV_CASE':{'maximum':200}}))
    for i,name in enumerate(ids):
        decision=dict(attempt_id=name,status='VALID_TASK_FAIL' if i==2 else 'SUCCESS',failure_stage='S1_PRE_EXECUTION_REJECT' if i==2 else 'COMPLETE_SUCCESS',model=panel,split=split,pipeline_trustworthy=True,teacher_takeover=False,live_native_failures=0,task='T1' if i%2==0 else 'T2',scene_group=split+'/F'+str(i//2)+'/000',duration_s=1.2 if i==2 else 32,RGB_pairs=25 if i==2 else 641,S1_rejections=1 if i==2 else 0)
        if panel==CORR_PANEL:
            decision.update(status='ENGINEERING_INCOMPLETE' if i>=2 else 'VALID_TASK_FAIL',
                pipeline_trustworthy=i<2,task='T1' if i!=1 else 'T2')
        put(child/'episodes'/name/'MP1_ADJUDICATION.json',decision)
        put(child/'worker_processes'/(name+'.json'),dict(pid=999999999,proc_start='0'))
        folder=child/'evidence/audits'/name;folder.mkdir(parents=True)
        movie=folder/'actual-body-wrist.mp4';movie.write_bytes(b'unit-fixture-not-a-real-qualified-video')
        put(folder/'AUDIT.private.json',dict(status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT',attempt_id=name,adjudication=decision,checks=dict(raw=True),errors=[],video=dict(sha256=sha(movie),bytes=movie.stat().st_size,decoded_frames=decision['RGB_pairs'],actual_paired_frames=decision['RGB_pairs'],native_duration_s=decision['duration_s'],padding_or_interpolation=False)))
        if panel==CORR2048_DEV_PANEL and name.endswith('-REPLACEMENT-01'):
            original=dict(decision,attempt_id=name.removesuffix('-REPLACEMENT-01'),status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False)
            put(child/'episodes'/original['attempt_id']/'MP1_ADJUDICATION.json',original)
            put(child/'worker_processes'/(original['attempt_id']+'.json'),dict(pid=999999999,proc_start='0'))
    return child


class SourceVideoTests(unittest.TestCase):
    def test_partial_CORR_snapshot_preserves_engineering_class_and_does_not_fill_missing_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root,CORR_PANEL);before={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
            rows,files,sources=audited_movies(root,panel=CORR_PANEL)
            self.assertEqual([r['id'] for r in rows],CORR_IDS)
            self.assertEqual(sum(r['qualified_policy_outcome'] for r in rows),2)
            self.assertEqual(sum(r['engineering_incomplete'] for r in rows),2)
            self.assertEqual(sum(r['independent_new_condition'] for r in rows),3)
            self.assertFalse(any(r['task_success'] for r in rows))
            self.assertFalse(delivery_scope(CORR_PANEL)['complete_panel'])
            self.assertNotIn(delivery_scope(CORR_PANEL)['plan_ids'][-1],[r['id'] for r in rows])
            self.assertEqual(len(files),len(sources));self.assertEqual(len(files),4)
            self.assertEqual(before,{str(p):sha(p) for p in root.rglob('*') if p.is_file()})

    def test_partial_CORR_snapshot_rejects_relabelled_outcome_fault_or_wrong_fixed_step(self):
        for defect in ['qualified_partial','policy_fail','native_fault','wrong_step','wrong_plan']:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);child=fixture(root,CORR_PANEL)
                if defect in ['wrong_step','wrong_plan']:
                    p=child/'plans/SOURCE_EVALUATION_PLAN.json';v=json.loads(p.read_text())
                    if defect=='wrong_step':v['panels'][CORR_PANEL]['checkpoint_step']=2048
                    else:v['panels'][CORR_PANEL]['attempt_ids']=CORR_IDS
                else:
                    p=child/'episodes'/CORR_IDS[2]/'MP1_ADJUDICATION.json';v=json.loads(p.read_text())
                    if defect=='qualified_partial':v['pipeline_trustworthy']=True
                    elif defect=='policy_fail':v.update(status='VALID_TASK_FAIL',pipeline_trustworthy=True)
                    else:v['live_native_failures']=1
                put(p,v)
                with self.assertRaises(PermissionError):audited_movies(root,panel=CORR_PANEL)

    def test_complete_exposed_DEV8_has_distinct_immutable_delivery_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root,DEV_PANEL);before={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
            rows,files,sources=audited_movies(root,panel=DEV_PANEL)
            self.assertEqual([r['id'] for r in rows],DEV_IDS);self.assertEqual(len(files),8)
            self.assertEqual(len(sources),8);self.assertEqual(rows[2]['status'],'VALID_TASK_FAIL')
            self.assertTrue(all(r['id'].startswith('PI0294-MP1-S-BC-2048-DEV8-') for r in rows))
            self.assertNotEqual(delivery_scope(PANEL)['name'],delivery_scope(DEV_PANEL)['name'])
            self.assertEqual(before,{str(p):sha(p) for p in root.rglob('*') if p.is_file()})

    def test_DEV_delivery_rejects_subset_wrong_split_and_unapproved_model(self):
        for defect in ['subset','split']:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);child=fixture(root,DEV_PANEL);p=child/'plans/SOURCE_EVALUATION_PLAN.json';v=json.loads(p.read_text())
                if defect=='subset':v['panels'][DEV_PANEL]['attempt_ids'].pop()
                else:v['panels'][DEV_PANEL]['split']='TRAIN'
                put(p,v)
                with self.assertRaises(PermissionError):audited_movies(root,panel=DEV_PANEL)
        for panel in ['S-CORR-1024-DEV8','SOURCE_CONFIRM_V1','S-BC-512-TRAIN4']:
            with self.assertRaises(PermissionError):delivery_scope(panel)

    def test_CORR2048_fixed_panels_publish_existing_audited_bytes_and_keep_replacement_denominator(self):
        for panel,count in [(CORR2048_TRAIN_PANEL,4),(CORR2048_DEV_PANEL,8)]:
            with self.subTest(panel=panel),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);fixture(root,panel);before={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
                rows,files,sources=audited_movies(root,panel=panel)
                self.assertEqual(len(rows),count);self.assertEqual(len(files),count);self.assertEqual(len(sources),count)
                self.assertTrue(all(r['qualified_policy_outcome'] for r in rows));self.assertTrue(delivery_scope(panel)['N1_required'])
                self.assertEqual(before,{str(p):sha(p) for p in root.rglob('*') if p.is_file()})
                if panel==CORR2048_DEV_PANEL:
                    self.assertEqual([r['id'] for r in rows],CORR2048_DEV_IDS)
                    self.assertEqual(sum(not r['independent_new_condition'] for r in rows),1)

    def test_CORR2048_DEV_replacement_requires_the_original_engineering_gap_not_valid_or_unsafe_case(self):
        for defect in ['valid','unsafe']:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);child=fixture(root,CORR2048_DEV_PANEL)
                p=child/'episodes'/CORR2048_DEV_IDS[-2].removesuffix('-REPLACEMENT-01')/'MP1_ADJUDICATION.json';v=json.loads(p.read_text())
                if defect=='valid':v.update(status='VALID_TASK_FAIL',pipeline_trustworthy=True)
                else:v['live_native_failures']=1
                put(p,v)
                with self.assertRaises(PermissionError):audited_movies(root,panel=CORR2048_DEV_PANEL)

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
