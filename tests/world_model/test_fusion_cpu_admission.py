import os
import unittest
from unittest.mock import patch

from go2arx5_vla.world_model.fusion_scope import validate_structured_cpu, validate_target_capture_cpu


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
        env = dict(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='osmesa',
                   PYOPENGL_PLATFORM='osmesa', LIBGL_ALWAYS_SOFTWARE='1')
        with patch.dict(os.environ, env), patch(
                'go2arx5_vla.world_model.fusion_scope.validate', return_value='original') as check:
            self.assertEqual(validate_target_capture_cpu(spec), 'original')
            check.assert_called_once_with(spec)
            for key, value in [('MUJOCO_GL', 'egl'), ('CUDA_VISIBLE_DEVICES', 'GPU-0'),
                               ('PYOPENGL_PLATFORM', 'egl'), ('LIBGL_ALWAYS_SOFTWARE', '0')]:
                with self.subTest(key=key), patch.dict(os.environ, {key: value}):
                    with self.assertRaises(PermissionError):
                        validate_target_capture_cpu(spec)
            with self.assertRaises(PermissionError):
                validate_target_capture_cpu(dict(spec, capture_steps=[3503, 3703]))


if __name__ == '__main__':
    unittest.main()
