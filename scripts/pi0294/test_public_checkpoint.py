"""Public metadata tests: no model, Torch, pickle execution or remote calls."""
import pickle
from pathlib import Path
import tempfile
import unittest
import zipfile
from scripts.pi0294.public_checkpoint import public_config,relocate_pickle,relocate_args,reject_private


class PublicCheckpointTest(unittest.TestCase):
    def test_configs_only_rebind_model_name(self):
        value={'model_name':'/scratch/user/cosmos','action_horizon':40}
        out=public_config(value)
        self.assertEqual(out,dict(model_name='nvidia/Cosmos-Reason2-2B',action_horizon=40))
        self.assertEqual(value['model_name'],'/scratch/user/cosmos')
        out=public_config({'processor_kwargs':value},processor=True)
        self.assertEqual(out['processor_kwargs']['action_horizon'],40)

    def test_other_private_values_are_not_silently_removed(self):
        with self.assertRaises(ValueError):public_config({'model_name':'/scratch/x','other':'/home/user'})
        with self.assertRaises(ValueError):reject_private({'token':'hf_'+'a'*30})

    def test_args_rewrite_is_structural_and_retains_parameters(self):
        raw=pickle.dumps(dict(output_dir='/scratch/run/training/main',
            logging_dir='/scratch/run/logs',learning_rate=1e-4,bf16=True),protocol=2)
        out,n=relocate_pickle(raw,'/scratch/run')
        self.assertEqual(n,2)
        self.assertNotIn(b'/scratch/',out)
        self.assertIn(b'PUBLIC_RUN_ROOT/training/main',out)

    def test_unknown_paths_or_protocol_refused(self):
        for raw in (pickle.dumps({'output_dir':'/home/foreign'},protocol=2),
                    pickle.dumps({'output_dir':'/scratch/run/a'},protocol=4)):
            with self.assertRaises(ValueError):relocate_pickle(raw,'/scratch/run')

    def test_archive_rewrites_header_without_loading_or_mutating_original(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.bin';target=Path(directory)/'public.bin'
            raw=pickle.dumps(['/scratch/run/training/main','/scratch/run/logs'],protocol=2)
            with zipfile.ZipFile(source,'x') as z:
                z.writestr('training_args/data.pkl',raw);z.writestr('training_args/version','3\n')
            before=source.read_bytes()
            self.assertEqual(relocate_args(source,target,'/scratch/run'),2)
            self.assertEqual(source.read_bytes(),before)
            with zipfile.ZipFile(target) as z:self.assertEqual(z.read('training_args/version'),b'3\n')


if __name__=='__main__':unittest.main()
