import unittest
from go2arx5_vla.world_model.runtime import cuda_uuid, validate_core_smoke


class DeviceIdentityTests(unittest.TestCase):
    def test_actual_pytorch_bare_uuid_and_nvml_form_bind_same_device(self):
        actual='28534eb9-5324-45fe-3227-1f7a9cb3c5f2'
        self.assertEqual(cuda_uuid(actual),'GPU-'+actual)
        self.assertEqual(cuda_uuid('GPU-'+actual),'GPU-'+actual)

    def test_bad_device_identifier_is_not_a_fallback_to_gpu_zero(self):
        for value in ('0','',None,'GPU-garbage'):
            with self.assertRaises(ValueError):cuda_uuid(value)

    def test_core_sharing_is_only_the_authorized_bounded_gpu1_smoke(self):
        import copy
        spec = dict(host='corelab-4080s', uid=2009, profile='core-wm01-shared-smoke', physics=False,
            GPU_uuid='GPU-845348f1-1511-dc19-0877-cc97658e3edd',
            sharing=dict(direct_user_answer='允许GPU1共享', endpoint=200, allocator_limit_bytes=512*1024**2))
        observed = dict(uuid=spec['GPU_uuid'], name='NVIDIA GeForce RTX 4080 SUPER',
                        total_bytes=16376*1024**2, free_bytes=2439*1024**2)
        validate_core_smoke(spec, observed)
        for field, value in [('uuid', 'GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'),
                             ('free_bytes', 1024**3), ('name', 'NVIDIA A40')]:
            changed = dict(observed); changed[field] = value
            with self.assertRaises(PermissionError):validate_core_smoke(spec, changed)
        for field, value in [('endpoint', 5000), ('allocator_limit_bytes', 4*1024**3),
                             ('direct_user_answer', '')]:
            changed = copy.deepcopy(spec); changed['sharing'][field] = value
            with self.assertRaises(PermissionError):validate_core_smoke(changed, observed)


if __name__=='__main__':unittest.main()
