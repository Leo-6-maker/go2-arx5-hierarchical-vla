import unittest
from go2arx5_vla.world_model.next2_scope import validate_limits,PLAN_SHA,C08
class Next2Scope(unittest.TestCase):
    def test_new_scope_is_finite_and_not_an_old_lease_or_arbitrary_duration(self):
        b=dict(schema='WM02_NEXT2_FINITE_WINDOW_V1',plan_sha=PLAN_SHA,owner='EXISTING_LOCAL_PC',host='corelab-4080s',uid=2009,scope='WM02_NEXT2_SIM_ONLY',C08_native_binary_sha256=C08,max_native_attempts=96,max_native_ticks=600000,max_native_slots=2,max_task_atoms=400,max_main_updates=1600,max_restore_probes=4,max_neural_recipes=2,activation_epoch=1000,dispatch_cutoff_epoch=21400,deadline_epoch=22600,parent_campaign_deadline_epoch=100000)
        validate_limits(b,now=1001)
        for c in [dict(schema='WM02_NEXT1_FINITE_WINDOW_V1'),dict(max_task_atoms=640),dict(max_native_slots=4),dict(deadline_epoch=22601),dict(scope='WM02_AUTO1_SIM_ONLY')]:
            with self.assertRaises(PermissionError):validate_limits(dict(b,**c),now=1001)
        with self.assertRaises(PermissionError):validate_limits(b,now=21400)
