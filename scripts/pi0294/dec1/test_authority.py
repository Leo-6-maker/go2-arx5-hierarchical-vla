from copy import deepcopy
import unittest

from .authority import HARD_CLOSE, PARENT_AUTHORITY, available


def fixture():
    a=dict(authority_release=PARENT_AUTHORITY,run_uuid='original',T0_epoch=1792348196,
           hard_closeout_epoch=HARD_CLOSE)
    local=dict(run_uuid='original',T0_epoch=a['T0_epoch'],hard_closeout_epoch=HARD_CLOSE,
        unknown_effects=[],pending_updates=[],GPU_allocated_s=20473,GPU_reserved_s=28860,
        CPU_s=dict(CoreLab=136069,Hopper=6671,local_PC=25224),
        CPU_reserved_s=dict(CoreLab=43200,Hopper=0,local_PC=0),episode_starts=290,
        physics_ticks=dict(live=4577600,shadow=4577966,reconstruction=0),engine_starts=25380,
        final_reserve_inside_total=dict(H200_s=72000,neural_queries=9000,episode_starts=100,
                                       live_ticks=1600000,shadow_ticks=1600000))
    h=dict(authority_release=PARENT_AUTHORITY,campaign_run_uuid='original',CPU_s=6671,
        unknown_effects=[],pending_updates=[],used=dict(H200_s=20473,PBS_submissions=11,
        optimizer_updates=0,neural_queries=1054),reserved=dict(H200_s=28860),
        optimizer_updates=0,generation_queries=1054,
        PBS_submissions=[dict(status='NATIVE_TERMINAL',job_id=str(i)) for i in range(10)] +
            [dict(status='SUBMITTED_PENDING_NATIVE',job_id='679029.hopper-m-02',
                  reserved_H200_s=28800,accounting_pad_s=60)])
    return a,local,h


class AuthorityTests(unittest.TestCase):
    def test_budget_is_nonadditive_and_held_reservation_is_not_free(self):
        a,l,h=fixture(); before=deepcopy((a,l,h));v=available(a,l,h)
        self.assertEqual((a,l,h),before)
        self.assertEqual(v['DEC1_effective_caps']['GPU_s'],48*3600)
        self.assertEqual(v['original_remaining_after_spent_reservations_and_final']['GPU_s'],224267)
        self.assertTrue(v['no_new_GPU_while_held_may_recover'])

    def test_original_remaining_smaller_than_dec1_wins(self):
        a,l,h=fixture();h['used']['H200_s']=150000;l['GPU_allocated_s']=150000
        self.assertEqual(available(a,l,h)['DEC1_effective_caps']['GPU_s'],94740)

    def test_old_episode_tick_and_constructor_reservations_are_also_deducted(self):
        a,l,h=fixture();before=available(a,l,h)
        l['children']=[dict(episode_starts_reserved=34,physics_ticks_reserved=1088000,
                            engine_starts_reserved=5942)]
        after=available(a,l,h)
        for key,delta in (('episode_starts',34),('physics_ticks',1088000),('constructors',5942)):
            self.assertEqual(before['original_remaining_after_spent_reservations_and_final'][key]-delta,
                             after['original_remaining_after_spent_reservations_and_final'][key])
        self.assertEqual(before['DEC1_effective_caps'],after['DEC1_effective_caps'])

    def test_unknown_cost_bool_negative_clock_and_dropped_reservation_rejected(self):
        for mutate in (
            lambda a,l,h:h.update(unknown_effects=['UNKNOWN']),
            lambda a,l,h:a.update(hard_closeout_epoch=HARD_CLOSE+1),
            lambda a,l,h:h['reserved'].update(H200_s=0),
            lambda a,l,h:h['used'].update(H200_s=True),
            lambda a,l,h:h['used'].update(H200_s=-1),
            lambda a,l,h:h['used'].update(H200_s=float('nan')),
            lambda a,l,h:l['final_reserve_inside_total'].update(H200_s=0),
            lambda a,l,h:h['PBS_submissions'][-1].pop('job_id'),
        ):
            with self.subTest(mutate=mutate):
                a,l,h=fixture();mutate(a,l,h)
                with self.assertRaises(PermissionError):available(a,l,h)


if __name__ == '__main__':
    unittest.main()
