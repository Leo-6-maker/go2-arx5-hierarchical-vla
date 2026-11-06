import os
import unittest
from unittest.mock import patch

from go2arx5_vla.world_model.fusion_scope import (
    validate_structured_cpu, validate_target_capture_cpu, validate_corr_asset_cpu,
    validate_contact_timing_cpu)


class StructuredCPUAdmission(unittest.TestCase):
    def spec(self):
        return dict(execution_site='CORE_FUSION_OFFLINE_CPU',
                    profile='wm02-fusion-structured-offline-cpu',
                    kind='FUSION_STRUCTURED_RESPONSE', physics=False, device='cpu',
                    numeric_mode='NUMPY_FLOAT64', GR00T_updates=0,
                    arms=['ACTION_CONDITIONED', 'NO_FUTURE_ACTION'],
                    min_lopo_gain_mm=.08124448952310876)

    def test_offline_fit_still_checks_original_owner_and_scope(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), patch(
                'go2arx5_vla.world_model.fusion_scope.validate', return_value='original') as check:
            spec = self.spec()
            self.assertEqual(validate_structured_cpu(spec), 'original')
            check.assert_called_once_with(spec)
            check.side_effect = PermissionError('CLOSED_OWNER')
            with self.assertRaisesRegex(PermissionError, 'CLOSED_OWNER'):
                validate_structured_cpu(spec)

    def test_cuda_physics_changed_threshold_or_training_gr00t_never_admitted(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), patch(
                'go2arx5_vla.world_model.fusion_scope.validate') as check:
            for key, value in [('physics', True), ('device', 'cuda'),
                               ('min_lopo_gain_mm', 0.), ('GR00T_updates', 1),
                               ('kind', 'ACTOR_DISTILLATION')]:
                with self.subTest(key=key), self.assertRaises(PermissionError):
                    validate_structured_cpu(dict(self.spec(), **{key: value}))
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-occupied'):
                with self.assertRaises(PermissionError):
                    validate_structured_cpu(self.spec())
            check.assert_not_called()

    def test_saved_target_capture_needs_original_scope_and_software_rendering(self):
        spec = dict(execution_site='CORE_FUSION_TARGET_OBSERVATION_REPLAY_CPU',
                    profile='wm02-fusion-target-observation-replay-cpu',
                    kind='FUSION_TARGET_OBSERVATION_REPLAY', physics=False, device='cpu',
                    capture_steps=[3503, 3703, 4103], expected_native_ticks=4103,
                    GR00T_updates=0, WBC_changes=False)
        env = dict(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='egl',
                   PYOPENGL_PLATFORM='egl', LIBGL_ALWAYS_SOFTWARE='1',
                   __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
        with patch.dict(os.environ, env), patch(
                'go2arx5_vla.world_model.fusion_scope.validate', return_value='original') as check:
            self.assertEqual(validate_target_capture_cpu(spec), 'original')
            check.assert_called_once_with(spec)
            for key, value in [('MUJOCO_GL', 'osmesa'), ('CUDA_VISIBLE_DEVICES', 'GPU-0'),
                               ('PYOPENGL_PLATFORM', 'osmesa'), ('LIBGL_ALWAYS_SOFTWARE', '0'),
                               ('__EGL_VENDOR_LIBRARY_FILENAMES', 'nvidia.json'), ('MUJOCO_EGL_DEVICE_ID', '0')]:
                with self.subTest(key=key), patch.dict(os.environ, {key: value}):
                    with self.assertRaises(PermissionError):
                        validate_target_capture_cpu(spec)
            with self.assertRaises(PermissionError):
                validate_target_capture_cpu(dict(spec, capture_steps=[3503, 3703]))

    def test_fixed_source_asset_is_not_a_grant_to_load_a_model_or_use_cuda(self):
        spec = dict(execution_site='CORE_FUSION_ASSET_CPU',
                    profile='wm02-fusion-corr-asset-cpu', kind='FUSION_CORR_ASSET',
                    physics=False, device='cpu', GR00T_updates=0, WBC_changes=False,
                    original_checkpoint_manifest_sha256=
                    '94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38',
                    manifest_sha256='ec46067a1179405d6c75081cfa333f0daa994f8e1c54bd3658eac0b5fb4eff61',
                    data_revision='b37a3d959335b548abb45f87ad48a5186ec662d9')
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), patch(
                'go2arx5_vla.world_model.fusion_scope.validate', return_value='original') as check:
            self.assertEqual(validate_corr_asset_cpu(spec), 'original')
            check.assert_called_once_with(spec)
            for key, value in [('manifest_sha256', 'e1-other-model'), ('physics', True),
                               ('device', 'cuda'), ('GR00T_updates', 1),
                               ('data_revision', 'main')]:
                with self.subTest(key=key), self.assertRaises(PermissionError):
                    validate_corr_asset_cpu(dict(spec, **{key: value}))
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-0'):
                with self.assertRaises(PermissionError):
                    validate_corr_asset_cpu(spec)

    def test_contact_timing_offline_role_keeps_original_scope_without_native_or_GPU_grant(self):
        spec = dict(execution_site='CORE_FUSION_CONTACT_TIMING_CPU',
            profile='wm02-fusion-contact-timing-offline-cpu', kind='FUSION_CONTACT_TIMING_BASELINE',
            recipe='WM02_CONTACT_TIMING_MATCHED_TRAIN_FIR_V1', physics=False, device='cpu',
            numeric_mode='NUMPY_FLOAT64', ridge=1., arms=['ACTION_CONDITIONED', 'NO_FUTURE_PROGRAM'],
            information='PREDECLARED_PROGRAM_CONDITIONAL', GR00T_updates=0, WBC_changes=False)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), patch(
                'go2arx5_vla.world_model.fusion_scope.validate', return_value='original') as check:
            self.assertEqual(validate_contact_timing_cpu(spec), 'original')
            check.assert_called_once_with(spec)
            for key, value in [('physics', True), ('device', 'cuda'), ('GR00T_updates', 1),
                               ('ridge', .1), ('information', 'INSTALLED_FUTURE')]:
                with self.subTest(key=key), self.assertRaises(PermissionError):
                    validate_contact_timing_cpu(dict(spec, **{key:value}))


if __name__ == '__main__':
    unittest.main()
