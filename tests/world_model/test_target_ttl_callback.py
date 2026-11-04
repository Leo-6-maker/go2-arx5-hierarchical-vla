import unittest
from types import SimpleNamespace
from scripts.pi0294.native_receiver import measured_observation,reference_expired

class TTLObservation(unittest.TestCase):
    def test_real_observation_at_deadline_expires_without_another_clock_tick(self):
        for t,expected in ((5.398,False),(5.4,True),(5.402,True)):
            m=SimpleNamespace(_r4_pending=None,_r4_deadline=5.4,observation=SimpleNamespace(time=5.398),mode='WBC')
            def original(m,msg):m.observation=msg
            def expiry(m):
                if reference_expired(m._r4_deadline,m.observation.time):m.mode='HOLD';m._r4_deadline=None
            measured_observation(m,SimpleNamespace(time=t),original,lambda *_:self.fail('No pending transaction'),expiry)
            self.assertEqual(m.mode=='HOLD',expected);self.assertEqual(m.observation.time,t)
if __name__=='__main__':unittest.main()
