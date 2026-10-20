import copy
import unittest
from .review import transitions,GATES,PASS


class ReviewTests(unittest.TestCase):
    def row(self,key,role,old_pass=True):
        result=dict(H40_windows_actually_checked=601,frozen_numeric_gates=GATES,
                    maximum_error={k:0. for k in GATES},status=PASS)
        old=copy.deepcopy(result)
        if not old_pass:old['status']='BLOCKED_FROZEN_CODEC_PHYSICAL_ERROR';old['maximum_error']['SO3_rad']=.001
        return dict(id=key,role=role,lineage=key,frozen_139_member=True,R0=old,R1=result,
                    transition='PASS->PASS' if old_pass else 'FAIL->PASS')

    def test_recompute_support_and_independent_group_count(self):
        result=transitions([self.row('fit','FIT'),self.row('diag','DIAGNOSTIC',False)])
        self.assertTrue(result['supported']);self.assertEqual(result['recovered_diagnostic_lineages'],1)

    def test_leakage_and_changed_status_rejected(self):
        rows=[self.row('fit','FIT'),self.row('diag','DIAGNOSTIC',False)]
        rows[1]['lineage']='fit'
        with self.assertRaises(ValueError):transitions(rows)
        rows=[self.row('fit','FIT')];rows[0]['R1']['maximum_error']['SO3_rad']=.01
        with self.assertRaises(ValueError):transitions(rows)
