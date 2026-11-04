import unittest,struct,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from go2arx5_vla.world_model.native_replay import packets,forbid_integration,apply_frame

def raw(step=1):return struct.pack('<8sQdIII',b'GRTSTATE',step,step*.002,34,32,20)+np.zeros(86,dtype='<f8').tobytes()+struct.pack('<I',0)
class NativeReplay(unittest.TestCase):
    def test_full_packet_identity_clock_truncation_and_gap_are_explicit(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'state';a=raw();b=raw(3);p.write_bytes(struct.pack('<I',len(a))+a+struct.pack('<I',len(b))+b);rows=packets(p);self.assertEqual([r['step']for r in rows],[1,3]);p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):packets(p)
    def test_all_integration_entrypoints_are_denied_and_restored(self):
        f=lambda *args:None;m=SimpleNamespace(mj_step=f,mj_step1=f,mj_step2=f)
        with forbid_integration(m)as calls:
            for name in('mj_step','mj_step1','mj_step2'):
                with self.assertRaises(RuntimeError):getattr(m,name)(None,None)
            self.assertEqual(len(calls),3)
        self.assertIs(m.mj_step,f)
    def test_forward_cannot_change_recorded_generalized_state(self):
        row=dict(qpos=np.zeros(34),qvel=np.zeros(32),ctrl=np.zeros(20),time_s=.002);d=SimpleNamespace(qpos=np.zeros(34),qvel=np.zeros(32),ctrl=np.zeros(20),time=0.)
        m=SimpleNamespace(mj_forward=lambda _,d:None);apply_frame(m,None,d,row);self.assertEqual(d.time,.002)
        m.mj_forward=lambda _,d:d.qpos.__setitem__(0,1)
        with self.assertRaises(RuntimeError):apply_frame(m,None,d,row)
if __name__=='__main__':unittest.main()
