import unittest
from go2arx5_vla.world_model.pilot1_scope import binding_contract, PLAN_SHA, C08


class PilotScope(unittest.TestCase):
    def test_new_scope_has_site_endpoints_but_no_cumulative_project_caps(self):
        value=dict(schema='WM02_PILOT1_SCOPE_V1',plan_sha=PLAN_SHA,scope='WM02_PILOT1_SIM_ONLY',
                   owner='EXISTING_LOCAL_PC',host='corelab-4080s',uid=2009,engineering_budget_caps=None,
                   max_native_slots=2,C08_native_binary_sha256=C08,activation_epoch=100,deadline_epoch=1000000,
                   parent_campaign_deadline_epoch=1000000)
        binding_contract(value,101)
        for changed in (dict(scope='WM02_NEXT2_SIM_ONLY'),dict(engineering_budget_caps=dict(attempts=96)),
                        dict(max_native_slots=4),dict(deadline_epoch=1000001),dict(C08_native_binary_sha256='changed')):
            with self.subTest(changed=changed),self.assertRaises(PermissionError):binding_contract(dict(value,**changed),101)
        with self.assertRaises(PermissionError):binding_contract(value,1000000)


if __name__=='__main__':unittest.main()
