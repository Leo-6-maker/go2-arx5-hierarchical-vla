import unittest
from go2arx5_vla.world_model.runtime import cuda_uuid


class DeviceIdentityTests(unittest.TestCase):
    def test_actual_pytorch_bare_uuid_and_nvml_form_bind_same_device(self):
        actual='28534eb9-5324-45fe-3227-1f7a9cb3c5f2'
        self.assertEqual(cuda_uuid(actual),'GPU-'+actual)
        self.assertEqual(cuda_uuid('GPU-'+actual),'GPU-'+actual)

    def test_bad_device_identifier_is_not_a_fallback_to_gpu_zero(self):
        for value in ('0','',None,'GPU-garbage'):
            with self.assertRaises(ValueError):cuda_uuid(value)


if __name__=='__main__':unittest.main()
