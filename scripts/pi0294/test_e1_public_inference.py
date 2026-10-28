import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from .e1.public_inference import inference_members,tensor_header


class PublicInferenceTests(unittest.TestCase):
    def test_video_publication_requires_separate_bound_public_permission(self):
        from .e1 import video_delivery
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ledger=root/'RESOURCE_LEDGER.json';ledger.write_text(json.dumps({'reservations':{}}))
            with patch.object(video_delivery,'validate'),patch.object(video_delivery,'Ledger') as cls:
                cls.return_value.path=ledger
                with self.assertRaises(PermissionError):video_delivery.publish(root,'foreign/repo','unused',public=True)
                with self.assertRaises(FileNotFoundError):video_delivery.publish(root,'leoyuer/pi0294-public-artifacts','unused',public=True)
            self.assertFalse((root/'artifacts').exists())

    def test_only_inference_allowlist_not_native_optimizer_or_arguments(self):
        files={n:'x' for n in ['config.json','statistics.json','model.safetensors.index.json',
            'processor/embodiment_id.json','processor/processor_config.json','processor/statistics.json',
            'model-00001-of-00001.safetensors','optimizer.pt','rng_state.pth','training_args.bin']}
        selected=inference_members(files,{'weight_map':{'weight':'model-00001-of-00001.safetensors'}})
        self.assertEqual(len(selected),7)
        self.assertNotIn('training_args.bin',selected);self.assertNotIn('optimizer.pt',selected)
        with self.assertRaises(ValueError):inference_members(files,{'weight_map':{'x':'../secret'}})
        with self.assertRaises(ValueError):inference_members({}, {'weight_map':{'x':'model-00001-of-00001.safetensors'}})

    def test_weight_header_checked_without_tensor_or_pickle_loading(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'weights';raw=json.dumps({'__metadata__':{'source':'research'}}).encode()
            p.write_bytes(struct.pack('<Q',len(raw))+raw);tensor_header(p)
            raw=json.dumps({'__metadata__':{'path':'/scratch/private'}}).encode()
            p.write_bytes(struct.pack('<Q',len(raw))+raw)
            with self.assertRaises(ValueError):tensor_header(p)
            p.write_bytes(struct.pack('<Q',11*1024**2))
            with self.assertRaises(ValueError):tensor_header(p)
