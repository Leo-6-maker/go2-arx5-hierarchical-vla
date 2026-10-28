import copy
import unittest

from scripts.pi0294.cr2 import scope as old
from scripts.pi0294.cr2.runtime import context
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.vap1 import scope
from .plan import AUTH, R1


class ScopeTests(unittest.TestCase):
    def adoption(self):
        t0 = 1792556800
        return dict(schema='PI0294_VAP1_SOURCE_DATA_ADOPTION_V1',owner='EXISTING_LOCAL_PC',
            protocol_release=AUTH,limits=copy.deepcopy(scope.LIMITS),R0_statistics_sha256=STATS,
            R1_statistics_sha256=R1,prior_closed_scopes_reused=False,PV1_numerics_changed=False,
            worker_cpu_pool=[18,19],PV1_reserved_CPUs=[5],T0_epoch=t0,
            stop_dispatch_epoch=t0+40*3600,stop_compute_epoch=t0+44*3600,hard_close_epoch=t0+48*3600)

    def test_new_authority_not_old_replay_or_reopened_CR2(self):
        a=self.adoption();scope.authority(a,a['T0_epoch']+1)
        with self.assertRaises(PermissionError): old.authority(a,a['T0_epoch']+1)
        for key,value in (('protocol_release','OLD_GO'),('PV1_numerics_changed',True),
                          ('worker_cpu_pool',[5,19]),('prior_closed_scopes_reused',True)):
            b=self.adoption();b[key]=value
            with self.assertRaises(PermissionError): scope.authority(b,b['T0_epoch']+1)

    def test_no_budget_or_deadline_inflation(self):
        for key in scope.LIMITS:
            b=self.adoption();b['limits'][key]+=1
            with self.assertRaises(PermissionError): scope.authority(b,b['T0_epoch']+1)
        b=self.adoption();b['hard_close_epoch']+=1
        with self.assertRaises(PermissionError): scope.authority(b,b['T0_epoch']+1)

    def test_only_explicit_new_profile_uses_new_guards(self):
        self.assertIs(context()[0],old.validate)
        self.assertIs(context('VAP1')[0],scope.validate)
        self.assertEqual(context()[4],12)
        self.assertEqual(context('VAP1')[4],24)
        with self.assertRaises(PermissionError): context('anything-larger')


if __name__ == '__main__': unittest.main()
