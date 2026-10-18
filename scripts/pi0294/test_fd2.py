"""CPU contracts only: no native engines, GPUs, models or remote access."""
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.pi0294.fd2 import Meter,adapt_teacher,geometry_row,pilot_plan,SCOPE


class FD2Tests(unittest.TestCase):
    def test_four_fresh_groups_eight_fixed_counterbalanced_cells(self):
        plan=pilot_plan([])
        self.assertEqual(len(plan['groups']),4)
        self.assertEqual([g['direction'] for g in plan['groups']],['+Y','+Y','-Y','-Y'])
        self.assertEqual([c['nominal_clearance_m'] for c in plan['cards']],
            [.004,.012,.012,.004,.004,.012,.012,.004])
        self.assertEqual(len({c['id'] for c in plan['cards']}),8)
        self.assertEqual(plan['replacements'],0)
        self.assertTrue(all(c['split']=='TRAIN' for c in plan['cards']))
        for i in range(4):
            a,b=copy.deepcopy(plan['cards'][i*2:i*2+2])
            for key in ['id','nominal_clearance_m']:a.pop(key);b.pop(key)
            self.assertEqual(a,b)

    def test_opened_geometry_even_new_appearance_is_rejected_without_search(self):
        old=pilot_plan([])['groups'][0]
        old=copy.deepcopy(old);old['group_id']='OLD_CONFIRM';old['split']='CONFIRM_CORE'
        old['visual']={'new_appearance':True}
        with self.assertRaises(ValueError):pilot_plan([old])

    def teacher(self):
        return SimpleNamespace(source_tcp=np.array([.45,-.08,.5]),goal_tcp=np.array([.45,.08,.5]),
            geometry=dict(tcp_height_m=.5),rotation=np.eye(3),stage='approach',other='FROZEN')

    def test_original_4mm_is_bitwise_identity(self):
        t=self.teacher();before=copy.deepcopy(t)
        self.assertIs(adapt_teacher(t,.004),t)
        self.assertEqual(t.source_tcp.tobytes(),before.source_tcp.tobytes())
        self.assertEqual(t.goal_tcp.tobytes(),before.goal_tcp.tobytes())
        self.assertEqual(t.geometry,before.geometry)

    def test_12mm_changes_only_two_reference_Z_values(self):
        t=self.teacher();before=copy.deepcopy(t);adapt_teacher(t,.012)
        np.testing.assert_array_equal(t.source_tcp[:2],before.source_tcp[:2])
        np.testing.assert_array_equal(t.goal_tcp[:2],before.goal_tcp[:2])
        self.assertEqual(t.source_tcp[2],before.source_tcp[2]+.008)
        self.assertEqual(t.goal_tcp[2],before.goal_tcp[2]+.008)
        np.testing.assert_array_equal(t.rotation,before.rotation)
        self.assertEqual((t.stage,t.other),(before.stage,before.other))

    def test_third_clearance_not_allowed(self):
        for x in [.008,0.,float('nan')]:
            with self.assertRaises(PermissionError):adapt_teacher(self.teacher(),x)

    def test_passive_geometry_never_calls_snapshot_forward_or_step(self):
        source=inspect.getsource(geometry_row)
        for call in ['.snapshot(','.step(','.mj_forward(','.mj_step(']:self.assertNotIn(call,source)

    def test_original_safety_exception_still_counts_actual_tick(self):
        meter=object.__new__(Meter);meter.guard=lambda:None
        meter.ticks=meter.live_ticks=meter.shadow_ticks=0
        engine=SimpleNamespace(steps=0)
        def failed(e,*args):e.steps+=1;raise RuntimeError('NATIVE_ORIGINAL_SAFETY')
        with self.assertRaisesRegex(RuntimeError,'NATIVE_ORIGINAL_SAFETY'):
            meter.step(engine,failed,None,None,True)
        self.assertEqual((meter.ticks,meter.live_ticks,meter.shadow_ticks),(1,1,0))

    def test_cap_is_checked_before_an_extra_tick_or_clone(self):
        meter=object.__new__(Meter);meter.guard=lambda:None
        meter.ticks=300000;meter.starts=700
        with self.assertRaises(PermissionError):meter.start()
        with self.assertRaises(PermissionError):meter.step(None,lambda *a:self.fail(),None,None,True)

    def test_whole_eight_can_fit_but_no_extra_scientific_run(self):
        card=json.loads((Path(__file__).resolve().parents[2]/SCOPE).read_text())
        self.assertEqual(card['teacher_episodes'],8)
        self.assertLessEqual(8*32000,card['limits']['all_physics_ticks'])
        self.assertLessEqual(8*(1+80+4),card['limits']['engine_starts_including_clones'])
        for key in ['GPU_s','PBS_submissions','neural_queries','optimizer_updates']:
            self.assertEqual(card['limits'][key],0)
        self.assertFalse(card['new_training_authorized'])


if __name__=='__main__':unittest.main()
