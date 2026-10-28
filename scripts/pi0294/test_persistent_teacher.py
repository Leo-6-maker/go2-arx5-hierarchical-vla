import unittest
from .teacher_qualification import prefix_boundary_pause
from .persistent_teacher import cards


class Contracts(unittest.TestCase):
    def test_observed_native_clock_float_crossing(self):
        from .task_clock import actual_source_age
        actual=10.200000000000076
        self.assertLess(actual_source_age(actual,actual),0.)
        source=round(actual*1e9)/1e9
        self.assertEqual(source,10.2)
        self.assertEqual(actual_source_age(actual,source),0.)
        self.assertLess(actual_source_age(actual,source+.001),0.)
        self.assertGreater(actual_source_age(actual,source-.026),.025)

    def test_fixed_budget(self):
        c=cards()
        self.assertEqual(len(c),6)
        self.assertEqual(sum(round((r['duration_s']+5)/.002)+3 for r in c),71018)
        self.assertEqual(sum(round(r['duration_s']/.05)+1 for r in c),2246)
        self.assertEqual(len({r['id'] for r in c}),6)

    def test_exact_paused_prefix_only(self):
        p=dict(running=False,step=2700);s=dict(step=2700,time_s=5.4)
        check=lambda **kw:prefix_boundary_pause(now=5.4,deadline=5.4,pause=p,packet=s,expected_step=2700,**kw)
        self.assertTrue(check())
        for change in (dict(running=True),dict(step=2699)):
            bad=dict(p,**change)
            self.assertFalse(prefix_boundary_pause(now=5.4,deadline=5.4,pause=bad,packet=s,expected_step=2700))
        for now in (5.398,5.402):
            self.assertFalse(prefix_boundary_pause(now=now,deadline=5.4,pause=p,packet=s,expected_step=2700))
        self.assertFalse(prefix_boundary_pause(now=5.4,deadline=5.4,pause=p,packet=dict(s,step=2701),expected_step=2700))


if __name__=='__main__':unittest.main()
