"""Contract tests only, not native closeout or mixed-batch acceptance."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from scripts.pi0294.data_cov1.closeout import authority,FAILED


class DataCov1Closeout(unittest.TestCase):
    def fixture(self):
        a=dict(T0_epoch=100,PV1_reserved_CPUs=[5])
        b=dict(status=FAILED,unknown_effects=[],episode_starts=4,CPU_s=5058,local_CPU_s=2700,
            CPU_reserved_s=0,local_CPU_reserved_s=0,constructors_reserved=0,episode_starts_reserved=0,ticks_reserved=0)
        stop=dict(reason='FIRST_FOUR_QUALIFICATION_GATE_NOT_MET')
        limits=dict(T0_epoch=100,deadline_epoch=1000,CPU_s=300,local_control_CPU_upper_s=300,
            episodes=0,constructors=0,physics_ticks=0,neural_queries=0,updates=0,PBS=0,
            output_bytes=64*1024**2,cpus=[19])
        return a,b,stop,limits

    @patch('scripts.pi0294.data_cov1.closeout.approval')
    def test_only_same_original_clock_and_remaining_cpu_are_admitted(self,approval):
        args=self.fixture();authority(*args,now=101);approval.assert_called_once_with(args[0])
        for key,value in [('T0_epoch',101),('deadline_epoch',100+24*3600+1),('CPU_s',301),
                ('local_control_CPU_upper_s',301),('episodes',1),('constructors',1),('physics_ticks',1),
                ('neural_queries',1),('updates',1),('PBS',1),('cpus',[5]),('cpus',[18,19])]:
            bad=deepcopy(args);bad[3][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(PermissionError):authority(*bad,now=101)

    @patch('scripts.pi0294.data_cov1.closeout.approval')
    def test_no_safety_stop_unknown_or_active_source_is_reopened(self,approval):
        args=self.fixture()
        for index,key,value in [(1,'status','ACTIVE_CONTROL_PLANE'),(1,'unknown_effects',['pending']),
                (1,'CPU_reserved_s',1),(1,'CPU_s',32*3600-200),(1,'local_CPU_s',4*3600-200),
                (2,'reason','NATIVE_SAFETY_FAULT')]:
            bad=deepcopy(args);bad[index][key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):authority(*bad,now=101)
        with self.assertRaises(PermissionError):authority(*args,now=1000)


if __name__=='__main__':unittest.main()
