"""Bounded source-chain count is separate from total PBS authorization."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from . import alternate_accounting as a,a40_accounting,heartbeat_recovery as h,b_heartbeat_recovery as b
from .test_a40_accounting import prequeue


def rows(n,active=1):
    return [dict(native_platform='VANDA_A40',platform_training_version='CPU_FIXTURE',
                 status='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY' if i<active else 'NATIVE_TERMINAL',
                 reserved_optimizer_updates=0,reserved_neural_queries=0) for i in range(n)]


class SourceChainTests(unittest.TestCase):
    def checked(self,value):
        with patch.object(h,'validate_global') as old,patch.object(b,'validate_global') as exact,patch.object(a40_accounting,'validate_item'):
            out=a.validate_items(value)
            if b.FIELD in value:exact.assert_called_once_with(value)
            return out

    def test_exact_adopted_grant_accepts_ninth_and_tenth_retained_source_rows(self):
        for n in (9,10):
            v=dict(PBS_submissions=rows(n),reserved={},**{h.FIELD:{},b.FIELD:{}})
            self.assertIs(self.checked(v),v)

    def test_no_eleventh_source_or_second_active_chain(self):
        for n,active in ((11,1),(10,2)):
            with self.subTest(n=n,active=active),self.assertRaisesRegex(PermissionError,'BOUNDED_SOURCE_JOBS'):
                self.checked(dict(PBS_submissions=rows(n,active),reserved={},**{h.FIELD:{},b.FIELD:{}}))

    def test_legacy_seven_and_heartbeat_eight_limits_unchanged(self):
        for n,scope in ((8,{}),(9,{h.FIELD:{}})):
            with self.subTest(n=n),self.assertRaisesRegex(PermissionError,'BOUNDED_SOURCE_JOBS'):
                self.checked(dict(PBS_submissions=rows(n),reserved={},**scope))

    def test_invalid_new_marker_cannot_raise_count_limit(self):
        v=dict(PBS_submissions=rows(9),reserved={},**{b.FIELD:{}})
        with patch.object(a40_accounting,'validate_item'),self.assertRaises(PermissionError):a.validate_items(v)

    def test_invalid_complete_proposal_fails_before_return_and_inputs_unchanged(self):
        from . import dual_platform as dual
        import hashlib
        f=prequeue();before=deepcopy((f['ledger'],f['DEC1']))
        with patch.object(dual,'GRANT_SHA',hashlib.sha256(f['grant_bytes']).hexdigest()),patch.object(a,'validate_items',side_effect=PermissionError('CPU_FULL_PROPOSAL_REJECT')) as final:
            with self.assertRaisesRegex(PermissionError,'CPU_FULL_PROPOSAL_REJECT'):
                a40_accounting.reserve(f['ledger'],f['DEC1'],request_bytes=f['request_bytes'],budget_proof=f['budget_proof'],grant_bytes=f['grant_bytes'],campaign_view=f['campaign_view'],now_epoch=f['now'])
        final.assert_called_once();self.assertEqual(before,(f['ledger'],f['DEC1']))


if __name__=='__main__':unittest.main()
