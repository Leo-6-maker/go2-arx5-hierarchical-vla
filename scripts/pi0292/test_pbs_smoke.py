import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .pbs_smoke import seconds,pbs_epoch,run


class TestPBSBoundary(unittest.TestCase):
    def test_walltime_and_epoch_parsing(self):
        self.assertEqual(seconds('00:30:00'),1800)
        self.assertEqual(seconds('04:30:00'),16200)
        self.assertEqual(pbs_epoch(1791860304),1791860304)
        self.assertEqual(pbs_epoch('1791860304'),1791860304)

    def test_login_entry_refused_before_model_or_files(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(PermissionError,'PBS_JOBID_REQUIRED'):
            run(SimpleNamespace(root='/nonexistent',request='/nonexistent'))


if __name__=='__main__':unittest.main()
