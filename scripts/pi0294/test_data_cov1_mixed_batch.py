"""Official batch shape contracts with CPU reader mocks, not a native PASS."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.pi0294.data_cov1 import mixed_batch
from scripts.pi0294.fd1 import put,sha


class Reader:
    def __init__(self,root,*args,**kwargs):self.root=Path(root)
    def __len__(self):return 601
    def __getitem__(self,anchor):return dict(processed=(self.root.name,anchor))


class Processor:
    def __init__(self):self.calls=[]
    def collator(self,rows):
        self.calls.append(rows);n=len(rows)
        return dict(inputs=dict(action=torch.zeros(n,40,132),state=torch.zeros(n,1,132),
            action_mask=torch.tensor(np.broadcast_to(np.r_[np.ones(10),np.zeros(122)],(n,40,132)).copy())))


class MixedParentBatch(unittest.TestCase):
    def fixture(self,root,count=2):
        bindings=[]
        for i in range(count):
            folder=root/f'parent{i}';folder.mkdir();identifier=f'parent{i}'
            (folder/'numeric').mkdir();numeric=folder/'numeric'/f'{identifier}.npz';numeric.write_bytes(b'CPU numeric fixture')
            numeric.chmod(0o400);put(folder/'statistics.json',dict(CPU_fixture=True))
            row=dict(id=identifier,admitted_for_training=True,status='PASS_SAVED_TEACHER_EXPORT_AND_OFFICIAL_CHAIN',
                codec=dict(status='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'),split='TRAIN',nominal_clearance_m=.004,
                full_task_success=True,independent_raw_admitted=True,file=f'numeric/{identifier}.npz',sha256=sha(numeric),
                source_raw_manifest_sha256=str(i)*64)
            put(folder/'EXPORT_RECEIPT.json',row);put(folder/'DATASET_MANIFEST.json',dict(episodes=[row]))
            bindings.append(dict(id=identifier,export_root=str(folder),export_receipt_sha256=sha(folder/'EXPORT_RECEIPT.json')))
        return bindings,sha(root/'parent0/statistics.json')

    def test_mixes_all_six_anchors_from_distinct_parents_and_repeats_same_cpu_batch(self):
        with tempfile.TemporaryDirectory() as d:
            bindings,stats=self.fixture(Path(d));p=Processor()
            with patch.object(mixed_batch,'NativeDataset',Reader),patch.object(mixed_batch,'STATS',stats):
                result=mixed_batch.qualify(bindings,p,'CPU_SOURCE','CPU_COSMOS')
            self.assertEqual(result['parent_count'],2);self.assertEqual(len(result['actual_parent_anchor_rows']),12)
            self.assertEqual({r[0] for r in p.calls[0]},{'parent0','parent1'})
            self.assertEqual(p.calls[0],p.calls[1]);self.assertFalse(result['model_forward'])

    def test_one_repeated_parent_and_changed_receipt_fail_before_reader(self):
        with tempfile.TemporaryDirectory() as d:
            bindings,stats=self.fixture(Path(d))
            with patch.object(mixed_batch,'NativeDataset') as reader:
                for own in ([bindings[0]],[bindings[0],bindings[0]]):
                    with self.assertRaises(PermissionError):mixed_batch.qualify(own,Processor(),'x','y')
                reader.assert_not_called()
            bindings[0]['export_receipt_sha256']='changed'
            with self.assertRaises(ValueError):mixed_batch.qualify(bindings,Processor(),'x','y')

    def test_odd_population_uses_small_batches_and_covers_every_real_parent(self):
        with tempfile.TemporaryDirectory() as d:
            bindings,stats=self.fixture(Path(d),count=3);p=Processor()
            with patch.object(mixed_batch,'NativeDataset',Reader),patch.object(mixed_batch,'STATS',stats):
                result=mixed_batch.qualify_all(bindings,p,'CPU_SOURCE','CPU_COSMOS')
            self.assertEqual(result['distinct_parent_count'],3)
            self.assertEqual(len(result['batches']),2)
            self.assertTrue(all(len(rows)==12 for rows in p.calls))


if __name__=='__main__':unittest.main()
