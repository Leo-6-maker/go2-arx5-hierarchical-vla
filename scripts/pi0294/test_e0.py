import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from . import e0_scope as e0


class E0Tests(unittest.TestCase):
    def test_native_pbs_time_is_fixed_site_time_not_local_timezone(self):
        self.assertEqual(e0.pbs_epoch('Fri Oct 16 20:01:16 2026'), 1792152076)
        self.assertEqual(e0.pbs_epoch('Fri Oct 16 20:04:24 2026'), 1792152264)
        self.assertEqual(e0.pbs_epoch('Thu Oct  1 00:00:00 2026'), 1790784000)
        for value in (1792152076, '1792152076'):
            self.assertEqual(e0.pbs_epoch(value), 1792152076)

    def test_native_pbs_time_fails_closed_on_ambiguous_or_invalid_input(self):
        for value in (None, True, False, -1, 1.5, '1792152076.0',
                      'Fri Oct 16 20:01:16 2026 UTC', '2026-10-16T20:01:16',
                      'Thu Oct 16 20:01:16 2026', 'Fri Oct 16 24:01:16 2026',
                      'Fri Oct 32 20:01:16 2026'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                e0.pbs_epoch(value)

    def manifest(self):
        return dict(schema='RUN_MANIFEST_V1',run_uuid=e0.RUN,owner='EXISTING_LOCAL_PC',pi='PI0294',
            authority=dict(protocol_ref=e0.AUTHORITY,approval='DIRECT_HUMAN_E0_ADOPTION'),limits=dict(e0.LIMITS),
            scope='M2048_V30_EEF10_SOURCE_D0_ORIGINAL_S1_INFERENCE_ONLY',
            data=dict(manifest_sha256=e0.DATA),interfaces=dict(statistics_sha256=e0.STATS))

    def test_new_window_not_old_allowance(self):
        value=self.manifest()
        e0.validate_authority(value,e0.T0)
        e0.validate_authority(value,e0.ADMISSION_END+1)
        for now in (e0.T0-1,e0.DEADLINE):
            with self.assertRaises(PermissionError):e0.validate_authority(value,now)
        value['limits']['training_updates']=1
        with self.assertRaises(PermissionError):e0.validate_authority(value,e0.T0)

    def test_no_dataset_or_guard_scope_substitution(self):
        for key in ('scope','owner','run_uuid'):
            value=self.manifest();value[key]='OTHER'
            with self.assertRaises(PermissionError):e0.validate_authority(value,e0.T0)
        value=self.manifest();value['interfaces']['statistics_sha256']='0'*64
        with self.assertRaises(PermissionError):e0.validate_authority(value,e0.T0)

    def test_inference_inventory_excludes_optimizer(self):
        names=['config.json','model.safetensors.index.json','statistics.json',
            'processor/embodiment_id.json','processor/processor_config.json','processor/statistics.json']
        names += [f'model-{i:05d}-of-00003.safetensors' for i in range(1,4)]
        identity=dict(global_step=2048,statistics_sha256=e0.STATS,files={n:'a'*64 for n in names})
        self.assertEqual(len(e0.inference_files(identity)),9)
        for key in ('optimizer.pt','scheduler.pt','SAMPLER_STATE.json'):
            altered=copy.deepcopy(identity);altered['files'][key]='a'*64
            with self.assertRaises(PermissionError):e0.inference_files(altered)

    def test_all_TRAIN_failures_are_not_DEV_gate(self):
        from .e0_core import classify
        result=dict(status='FAILED_PRESERVED',error='PermissionError: PRE_EXECUTION_SAFETY_REJECT')
        self.assertEqual(classify(result,dict(pre_execution_rejections=1,live_native_failures=0)),
            ('VALID_TASK_FAIL','S1_PRE_EXECUTION_REJECT',True))
        self.assertEqual(classify(dict(status='SAFETY_STOP_PRESERVED',error='JOINT_LIMIT'),{}),
            ('ACTUAL_SAFETY_FAIL','LIVE_NATIVE_SAFETY',False))
        broken=dict(status='FAILED_PRESERVED',error='S1_ACTUAL_LIVE_SHADOW_PARITY_MISMATCH')
        self.assertFalse(classify(broken,{})[2])

    def test_training_refused_before_allocation_or_torch(self):
        from .neural import verify_allocation
        with patch('scripts.pi0294.e0_scope.admit',side_effect=AssertionError('not reached')):
            with self.assertRaisesRegex(PermissionError,'INFERENCE_ONLY'):
                verify_allocation(Path('/unused')/e0.RUN,'main',Path('/unused'))

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(PermissionError):e0.inside(root/'..'/'bad',root)


if __name__=='__main__':unittest.main()
