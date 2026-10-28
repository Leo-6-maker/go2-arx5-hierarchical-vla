"""Finite control arithmetic only; no actual transport or task execution."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts.pi0294.pv1 import control_budget as control
from scripts.pi0294.relay_file_rpc import operation_budget
from scripts.pi0294.test_mp1_transport_fixture import put


class ControlBudget(unittest.TestCase):
    def test_full_finite_allocation_panel_fits_bound_without_action_or_watchdog_change(self):
        self.assertEqual(control.worst_case_calls(),124800)
        self.assertLess(control.worst_case_calls(),control.MAX_CALLS)
        for wall,cases in [(28801,24),(28800,25),(True,24),(3600,False)]:
            with self.assertRaises(PermissionError):control.worst_case_calls(wall_s=wall,conditions=cases)

    def test_new_scope_requires_both_presealed_profile_and_exact_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=dict(authority_release=control.AUTHORITY,control_budget_profile=control.PROFILE)
            m=dict(authority=dict(protocol_ref=control.AUTHORITY),execution=dict(control_budget_profile=control.PROFILE))
            registry=dict(control_budget_profile=control.PROFILE);put(root/'PV1_STAGE_BINDING.json',stage)
            maximum,value=operation_budget(root,m,registry,pv1=True)
            self.assertEqual(maximum,150000);self.assertEqual(value['poll_s'],.25)
            self.assertEqual(value['CPU_limit_s'],7200);self.assertTrue(value['one_shot_mutations'])
            for changed in [dict(registry,control_budget_profile='WRONG'),{}]:
                with self.assertRaises(PermissionError):operation_budget(root,m,changed,pv1=True)
            put(root/'PV1_STAGE_BINDING.json',dict(stage,control_budget_profile='WRONG'))
            with self.assertRaises(PermissionError):operation_budget(root,m,registry,pv1=True)

    def test_old_scopes_and_old_pv1_caps_remain_40000(self):
        self.assertEqual(operation_budget('/DOES_NOT_EXIST',{}, {},pv1=False),(40000,None))
        self.assertEqual(operation_budget('/DOES_NOT_EXIST',{}, {},pv1=True),(40000,None))
        m=dict(execution=dict(control_budget_profile=control.PROFILE))
        self.assertEqual(operation_budget('/DOES_NOT_EXIST',m,{},pv1=False),(40000,None))
