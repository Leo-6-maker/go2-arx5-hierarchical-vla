import unittest
from go2arx5_vla.world_model.next1_scope import validate_limits,PLAN_SHA,C08


class Next1Scope(unittest.TestCase):
    def test_old_closed_or_extended_window_and_changed_limits_rejected(self):
        b=dict(schema='WM02_NEXT1_FINITE_WINDOW_V1',plan_sha=PLAN_SHA,owner='EXISTING_LOCAL_PC',host='corelab-4080s',uid=2009,scope='WM02_NEXT1_SIM_ONLY',C08_native_binary_sha256=C08,activation_epoch=100,dispatch_cutoff_epoch=20500,deadline_epoch=21700,parent_campaign_deadline_epoch=50000,max_native_attempts=64,max_native_ticks=400000,max_native_slots=2,max_task_atoms=160,max_main_updates=600,max_restore_probes=3)
        validate_limits(b,now=200)
        for key,value in dict(plan_sha='old',deadline_epoch=21701,max_native_attempts=65,max_native_ticks=400001,max_native_slots=4,max_main_updates=601,scope='WM02_AUTO1_SIM_ONLY').items():
            with self.assertRaises(PermissionError):validate_limits(dict(b,**{key:value}),now=200)
        with self.assertRaises(PermissionError):validate_limits(b,now=20500)


if __name__=='__main__':unittest.main()
