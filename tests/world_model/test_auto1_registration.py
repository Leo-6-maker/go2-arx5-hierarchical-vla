import copy,unittest
from scripts.wm02.register_auto1 import validate

class Auto1Registration(unittest.TestCase):
    def setUp(self):
        self.profile=dict(name='4080-simulation',hostnames=['corelab-4080s'],python='/native/python')
        self.binding=dict(scope='WM02_AUTO1_SIM_ONLY',owner='EXISTING_LOCAL_PC',host='corelab-4080s',uid=2009,profile='4080-simulation',runtime_python='/native/python',authority_commit='930bb2ef0ff178771e48fb5f03ea3e37fa717582',activation_deadline_epoch=200.,campaign_hours=72,max_target_instances=1,max_init_candidates=6)
    def check(self,b):validate(b,host='corelab-4080s',uid=2009,now=100.,profile=self.profile)
    def test_exact_user_scope_still_requires_bound_existing_site_profile(self):self.check(self.binding)
    def test_changed_host_owner_expired_or_unbounded_scope_rejected(self):
        for key,value in dict(host='other',owner='IMPERSONATED',uid=1,profile='other',runtime_python='/other',authority_commit='a'*40,activation_deadline_epoch=99.,campaign_hours=73,max_target_instances=2,max_init_candidates=7).items():
            b=copy.deepcopy(self.binding);b[key]=value
            with self.assertRaises(PermissionError):self.check(b)

if __name__=='__main__':unittest.main()
