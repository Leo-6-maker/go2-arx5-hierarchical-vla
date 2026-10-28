"""Metadata-only fixtures; never import a model or simulator."""
from copy import deepcopy
import unittest

from scripts.pi0294.pv1.future_data_inventory import reconcile, PASS, BLOCKED, UNREVIEWED, GATES


def fixture():
    rows = []
    for i, status in enumerate((PASS, UNREVIEWED, UNREVIEWED, BLOCKED)):
        rows.append(dict(id=str(i), scene_group='group'+str(i//2), task='T1' if i%2 == 0 else 'T2',
            family='F2', direction='+Y', split='TRAIN', nominal_clearance_m=.004,
            fixed_bank_member=i==0, full_raw_audit_admitted=i!=2,
            raw_manifest_sha256='raw'+str(i), full_raw_audit_sha256='audit'+str(i), official_codec_row_status=status))
    def receipt(i):
        row=rows[i];passed=i==1
        return dict(row, status=PASS if passed else BLOCKED, admitted_for_training=passed,
            source_raw_manifest_sha256=row['raw_manifest_sha256'], source_full_audit_sha256=row['full_raw_audit_sha256'],
            codec=dict(status='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP' if passed else BLOCKED,
                H40_windows_actually_checked=601,statistics_refit=False,native_decode_unchanged=True,
                maximum_error=dict(XYZ_m=0. if passed else .001,SO3_rad=0.,gripper=0.,relative_SE3_max_abs=0.),
                frozen_numeric_gates=GATES.copy(),
                inverse_clip_by_EEF10_channel=[0,0,1,0,0,0,0,0,0,0],
                action_preclip_components=1,native_inverse_clip_components_float32=1,state_preclip_components=0))
    return dict(attempt_rows=rows,population=dict(actual_first_attempts=4)),[receipt(3)],[receipt(1)]


class FutureDataInventory(unittest.TestCase):
    def test_raw_failure_is_excluded_not_assumed_codec_eligible(self):
        card,rejected,fresh=fixture();original=deepcopy(card)
        out=reconcile(card,rejected,fresh)
        self.assertEqual(out['raw_audit_failure_excluded'],['2'])
        self.assertEqual(out['completed_offline_reviews'],1)
        self.assertEqual(out['future_original_reference_union']['paired_groups'],1)
        self.assertEqual(card,original)
        self.assertFalse(out['PV1_bank_changed'])
        self.assertEqual(out['D1_pilot'],'NOT_AUTHORIZED')

    def test_duplicate_or_missing_receipts_and_failed_raw_inclusion_fail_closed(self):
        card,rejected,fresh=fixture()
        for bad in ([],fresh+fresh,[dict(fresh[0],id='2')]):
            with self.assertRaises(ValueError):reconcile(card,rejected,bad)
        with self.assertRaises(ValueError):reconcile(card,[],fresh)

    def test_split_identity_codec_and_admission_cannot_be_relabelled(self):
        for target,key,value in [('card','split','DEV'),('fresh','source_full_audit_sha256','changed'),
                ('fresh','admitted_for_training',False),('fresh','task','T1')]:
            card,rejected,fresh=fixture()
            (card['attempt_rows'][0] if target=='card' else fresh[0])[key]=value
            with self.assertRaises((ValueError,PermissionError)):reconcile(card,rejected,fresh)
        card,rejected,fresh=fixture();fresh[0]['codec']['statistics_refit']=True
        with self.assertRaises(ValueError):reconcile(card,rejected,fresh)

    def test_numeric_gate_relaxation_nonfinite_or_passing_a_failure_is_rejected(self):
        for key,value in [('XYZ_m',.1),('XYZ_m',float('nan'))]:
            card,rejected,fresh=fixture();fresh[0]['codec']['maximum_error'][key]=value
            with self.assertRaises(ValueError):reconcile(card,rejected,fresh)
        card,rejected,fresh=fixture();fresh[0]['codec']['frozen_numeric_gates']['XYZ_m']=1.
        with self.assertRaises(ValueError):reconcile(card,rejected,fresh)


if __name__=='__main__':unittest.main()
