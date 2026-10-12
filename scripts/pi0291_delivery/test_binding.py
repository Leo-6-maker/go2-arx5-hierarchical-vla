"""Path binding unit fixtures, not physics or dataset qualification."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('binding',Path(__file__).with_name('bind_consumer.py'))
binding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(binding)


class BindingTests(unittest.TestCase):
    def fixture(self, root):
        native, raw = root/'native', root/'raw'
        native.mkdir();(native/'episodes').mkdir();(raw/'T291000').mkdir(parents=True)
        folder = raw/'T291000'
        for name in ('samples.json','physics.npz'):
            (folder/name).write_bytes(b'unit fixture, not actual data')
        files = [dict(path=name,sha256=binding.sha(folder/name)) for name in ('samples.json','physics.npz')]
        (folder/'FILE_MANIFEST.json').write_text(json.dumps(files))
        for name in ('statistics.json','STATISTICS_PROVENANCE.json','modality.json','task_map.json'):
            (native/name).write_text('{}')
        (native/'episodes/T291000.npz').write_bytes(b'unit export fixture')
        row = dict(id='T291000',split='TRAIN',windows=601,raw_root='/producer/unavailable',
                   file='episodes/T291000.npz',sha256=binding.sha(native/'episodes/T291000.npz'),
                   samples_sha256=binding.sha(folder/'samples.json'),physics_sha256=binding.sha(folder/'physics.npz'),
                   raw_FILE_MANIFEST_sha256=binding.sha(folder/'FILE_MANIFEST.json'))
        (native/'DATASET_MANIFEST.json').write_text(json.dumps(dict(domain='FIXED_BASE_ARX5_TASK_SIM',
            statistics_sha256=binding.sha(native/'statistics.json'),episodes=[row])))
        return native,raw

    def test_new_path_keeps_hashes_and_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native,raw = self.fixture(root)
            old = binding.sha(native/'DATASET_MANIFEST.json')
            binding.bind(native,raw,root/'consumer',['T291000'])
            value = binding.read(root/'consumer/DATASET_MANIFEST.json')
            self.assertEqual(value['episodes'][0]['raw_root'],str((raw/'T291000').resolve()))
            self.assertEqual(value['producer_manifest_sha256'],old)
            self.assertEqual(binding.sha(native/'DATASET_MANIFEST.json'),old)
            self.assertEqual(binding.sha(root/'consumer/statistics.json'),binding.sha(native/'statistics.json'))

    def test_wrong_raw_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native,raw = self.fixture(root)
            (raw/'T291000/physics.npz').write_bytes(b'wrong bytes')
            with self.assertRaises(AssertionError):
                binding.bind(native,raw,root/'consumer',['T291000'])


if __name__ == '__main__':
    unittest.main()
