import copy
import unittest
from .scope import authority
from .rules import AUTH


class ScopeTests(unittest.TestCase):
    def adoption(self):
        return dict(protocol_release=AUTH,owner='EXISTING_LOCAL_PC',
            approval_ref='HUMAN_REPLY_call_1c9b979dac7e4e92b967be93cc85083c_FULL_LIMITED_SCOPE',
            T0_epoch=100,stop_compute_epoch=100+22*3600,stop_new_batch_epoch=100+20*3600,hard_close_epoch=100+24*3600,
            limits=dict(CoreLab_CPU_s=43200,local_CPU_s=7200,max_workers=2,active_workers=1,cpus=[19],PV1_reserved_CPUs=[5],
                unique_parents=400,diagnostic_windows=32,output_bytes=8*1024**3,network_bytes=2*1024**3,
                constructors=0,episodes=0,physics_ticks=0,GPU=0,PBS=0,neural_forwards=0,queries=0,updates=0))

    def test_approved_scope(self): authority(self.adoption(),101)

    def test_expired_or_unauthorized_or_effects_rejected(self):
        for key,value in [('approval_ref','OLD_DATA_GO'),('stop_compute_epoch',100+23*3600)]:
            a=self.adoption();a[key]=value
            with self.assertRaises(PermissionError): authority(a,101)
        for key,value in [('GPU',1),('cpus',[5]),('unique_parents',401),('CoreLab_CPU_s',43201)]:
            a=copy.deepcopy(self.adoption());a['limits'][key]=value
            with self.assertRaises(PermissionError): authority(a,101)
        with self.assertRaises(PermissionError): authority(self.adoption(),100+22*3600)
