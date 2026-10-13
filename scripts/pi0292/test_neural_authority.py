import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .durable import write
from .neural_train import RELEASE,recipe_for,verify_allocation


class TestNeuralAuthority(unittest.TestCase):
    def test_recipe_batch_and_fixed_main(self):
        self.assertEqual(recipe_for('main',2)['max_steps'],4096)
        for batch in (1,2):
            recipe=recipe_for('smoke-first',batch)
            self.assertEqual(recipe['per_device_train_batch_size']*recipe['gradient_accumulation_steps'],32)
            self.assertEqual(recipe,recipe_for('smoke-resume',batch))
            self.assertEqual(recipe,recipe_for('smoke-continuous',batch))
            self.assertEqual(recipe['max_steps'],4)
            self.assertFalse(recipe['skip_weight_loading'])
        with self.assertRaises(ValueError):recipe_for('main',4)

    def test_no_job_denied_before_any_torch_or_query(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{},clear=True),patch('subprocess.check_output') as query:
            root=Path(d)
            write(root/'ADOPTION.json',dict(release=RELEASE,new_work_cutoff_epoch=10**12))
            write(root/'NEURAL_AUTHORIZATION.json',dict(job_id='own',allowed_modes=['smoke-first']))
            with self.assertRaisesRegex(PermissionError,'OWN_PBS'):verify_allocation(root,'smoke-first')
            query.assert_not_called()


if __name__=='__main__':unittest.main()
