"""Metadata-only report fixtures; success claims still require native receipts."""
from copy import deepcopy
import hashlib
import unittest

from scripts.pi0294.data_cov1.plan import candidate,slots,capture_cards,NEAR_REVISION
from scripts.pi0294.data_cov1.reports import stage_reports,joined_attempt,geometry_fingerprint
from scripts.pi0294.scene_cards import digest


def fixtures():
    groups=[candidate(**s,index=0,geometry_revision=NEAR_REVISION) for s in slots()]
    cards=capture_cards(groups,'P24');captures={};exports={}
    for card in cards:
        h=hashlib.sha256(card['id'].encode()).hexdigest()
        captures[card['id']]=dict(id=card['id'],scene_group=card['group_id'],task=card['task'],
            status='RAW_CLOSED_PENDING_CODEC',task_success=True,raw_admitted=True,admitted=False,
            codec_admitted=False,native_faults=0,unknown_effects=[],auditable_identity_and_clock=True,
            raw_manifest_sha256=h,raw_audit_sha256='b'*64)
        exports[card['id']]=dict(id=card['id'],scene_group=card['group_id'],task=card['task'],split='TRAIN',
            nominal_clearance_m=.004,source_card_sha256=digest(card),source_raw_manifest_sha256=h,
            source_full_audit_sha256='b'*64,full_task_success=True,independent_raw_admitted=True,windows=601,
            codec=dict(H40_windows_actually_checked=601,statistics_refit=False,encode_clip_components=0,
                status='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'),admitted_for_training=True,
            status='PASS_SAVED_TEACHER_EXPORT_AND_OFFICIAL_CHAIN',official_chain=dict(actual_anchors=[0,80,160,320,480,600],
                deterministic_repeat=True,full_641_two_camera_raw_RGB256_verified=True,model_forward=False))
    return groups,cards,captures,exports


class DataCov1Reports(unittest.TestCase):
    def test_complete_reports_have_12_crossed_layers_and_24_actual_geometry_pairs(self):
        groups,cards,captures,exports=fixtures();report=stage_reports(groups,'P24',captures,exports)
        self.assertEqual(report['all_attempts']['planned'],48)
        self.assertEqual(len(report['coverage_and_events']['strata']),12)
        self.assertEqual(report['coverage_and_events']['paired_admitted_same_geometry_groups'],24)
        self.assertTrue(report['stage_gate']['count_gate_pass']);self.assertTrue(report['first_four_gate']['count_gate_pass'])
        self.assertFalse(report['coverage_and_events']['PV1_bank_changed'])

    def test_preserved_unscorable_first_attempt_is_never_a_scientific_success_or_valid_failure(self):
        groups,cards,captures,exports=fixtures();c=cards[0];exports.pop(c['id'])
        captures[c['id']].update(status='ENGINEERING_INCOMPLETE',task_success=False,task_success_known=False,
            raw_admitted=False,valid_task_failure=False)
        report=stage_reports(groups,'P24',captures,exports)
        self.assertEqual(report['all_attempts']['statuses']['ENGINEERING_INCOMPLETE'],1)
        self.assertEqual(report['all_attempts']['unknown_task_results'],1)
        self.assertEqual(report['coverage_and_events']['paired_admitted_same_geometry_groups'],23)
        self.assertTrue(report['first_four_gate']['count_gate_pass'])
        self.assertEqual(report['all_attempts']['valid_task_failures'],0)

    def test_unexecuted_and_unexported_are_not_admitted_and_never_made_ready(self):
        groups,cards,captures,exports=fixtures()
        report=stage_reports(groups,'P24',{},{});self.assertEqual(report['all_attempts']['statuses'],dict(NOT_RUN=48))
        self.assertFalse(report['first_four_gate']['count_gate_pass']);self.assertFalse(report['stage_gate']['count_gate_pass'])
        row=joined_attempt(cards[0],captures[cards[0]['id']]);self.assertEqual(row['status'],'ENGINEERING_INCOMPLETE')
        self.assertFalse(row['admitted'])

    def test_bad_export_identity_unproven_reader_or_mismatched_executed_geometry_rejected(self):
        groups,cards,captures,exports=fixtures();card=cards[0];row=captures[card['id']]
        for key,value in [('source_card_sha256','changed'),('source_raw_manifest_sha256','changed'),('windows',600)]:
            bad=deepcopy(exports[card['id']]);bad[key]=value
            with self.assertRaises(PermissionError):joined_attempt(card,row,bad)
        bad=deepcopy(exports[card['id']]);bad['official_chain']['deterministic_repeat']=False
        with self.assertRaises(PermissionError):joined_attempt(card,row,bad)
        actual=deepcopy(card);actual['source_xy'][0]+=.001
        self.assertNotEqual(geometry_fingerprint(actual),geometry_fingerprint(card))
        with self.assertRaises(PermissionError):joined_attempt(card,row,exports[card['id']],actual_card=actual)
        other=cards[1]['id'];exports[other]['source_raw_manifest_sha256']=exports[card['id']]['source_raw_manifest_sha256']
        captures[other]['raw_manifest_sha256']=captures[card['id']]['raw_manifest_sha256']
        with self.assertRaises(PermissionError):stage_reports(groups,'P24',captures,exports)


if __name__=='__main__':unittest.main()
