import unittest,tempfile,struct
from pathlib import Path
import numpy as np
from go2arx5_vla.world_model.initialization_audit import changes,first_difference,load_packets


class InitializationAudit(unittest.TestCase):
    def test_actuation_hold_reports_tick_duration_and_first_divergence(self):
        a=[dict(step=i,time_s=i*.002,ctrl=np.zeros(20))for i in range(1,6)]
        b=[dict(x,ctrl=x['ctrl'].copy())for x in a]
        for row in b[2:]:row['ctrl'][5]=-23.7
        self.assertEqual(first_difference(a,b,'ctrl')['step'],3)
        self.assertEqual([(x['first_step'],x['last_step'])for x in changes(b,'ctrl')],[(1,2),(3,5)])
    def test_journal_reads_complete_original_abi_without_native_runtime(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'state.bin.journal';q=np.zeros(86);q[3]=1
            raw=struct.pack('<8sQdIII',b'GRTSTATE',1,.002,34,32,20)+q.astype('<f8').tobytes()+struct.pack('<I',0)
            p.write_bytes(struct.pack('<I',len(raw))+raw);self.assertEqual(load_packets(p)[0]['step'],1)
            p.write_bytes(struct.pack('<I',len(raw))+raw[:-1])
            with self.assertRaises((ValueError,struct.error)):load_packets(p)

if __name__=='__main__':unittest.main()
