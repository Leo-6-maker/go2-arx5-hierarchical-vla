import struct
import unittest
import numpy as np
from .task_state import parse_state


class NativeStateTests(unittest.TestCase):
    def test_exact_actual_state_and_force_frame_bytes(self):
        values=np.arange(86,dtype='<f8');contact=np.arange(18,dtype='<f8')
        raw=struct.pack('<8sQdIII',b'GRTSTATE',25,.05,34,32,20)+values.tobytes()+struct.pack('<IIId',1,4,7,-.001)+contact.tobytes()
        v=parse_state(raw)
        self.assertEqual(v['step'],25);self.assertEqual(v['time_s'],.05)
        np.testing.assert_array_equal(v['qpos'],values[:34])
        np.testing.assert_array_equal(v['qvel'],values[34:66])
        np.testing.assert_array_equal(v['ctrl'],values[66:])
        np.testing.assert_array_equal(v['contacts'][0]['wrench'],contact[:6])
        np.testing.assert_array_equal(v['contacts'][0]['frame'].ravel(),contact[6:15])
        self.assertAlmostEqual(v['contact_time_s'],.048)

    def test_partial_and_wrong_abi_rejected(self):
        raw=struct.pack('<8sQdIII',b'GRTSTATE',1,.002,34,32,20)+np.zeros(86,dtype='<f8').tobytes()+struct.pack('<I',0)
        for bad in (raw[:10],raw[:-1],b'WRONGABI'+raw[8:],raw+b'x'):
            with self.assertRaises((ValueError,struct.error)):parse_state(bad)


if __name__=='__main__':unittest.main()
