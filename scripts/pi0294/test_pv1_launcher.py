"""CPU bootstrap tests are not native scheduler or project-authorization proof."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import launcher
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.test_mp1_transport_fixture import put


class LauncherContracts(unittest.TestCase):
    def test_login_host_cannot_create_ready_or_launch_a_neural_process(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'PBS_JOBID':''}):
            root=Path(tmp)/'not-created'
            with self.assertRaisesRegex(PermissionError,'ONLY_IN_OWN_NATIVE'):launcher.run(root,root/'REQUEST.json')
            self.assertFalse(root.exists())

    def test_missing_owner_release_exits_within_original_180_seconds_without_model_or_login(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'pbs/alloc-01';directory.mkdir(parents=True)
            request=directory/'REQUEST.json';put(request,dict(stage='B0_DEV'),True)
            charges=[];ledger=SimpleNamespace(guard=lambda:None,cpu=charges.append)
            clock=iter((1000,1181,1182))
            with patch.object(launcher,'native_bootstrap',return_value=({},dict(stage='B0_DEV'),directory,'123','compute-fixture')),\
                    patch.object(launcher,'Ledger',return_value=ledger),patch('scripts.pi0294.pv1.neural.stage_work'),\
                    patch.object(launcher.time,'time',side_effect=lambda:next(clock)),patch.object(launcher.subprocess,'run') as child:
                with self.assertRaisesRegex(TimeoutError,'RELEASE_MISSING'):launcher.run(root,request)
                child.assert_not_called()
            self.assertGreaterEqual(charges[0],60)
            self.assertTrue((directory/'PBS_READY.json').exists());self.assertTrue((directory/'PBS_WRAPPER_FINAL.json').exists())

    def test_exact_native_release_invokes_only_the_pinned_pv1_module_never_amgr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'pbs/alloc-01';directory.mkdir(parents=True)
            request=directory/'REQUEST.json';r=dict(stage='B0_DEV',submission_nonce='fixture',python='/CPU-fixture/python')
            put(request,r,True);attest=directory/'RUNNING_PROJECT_ATTESTATION.json';put(attest,dict(fixture=True),True)
            put(directory/'START_NEURAL.json',dict(run_uuid=root.name,job_id='123',request_sha256=sha(request),
                submission_nonce='fixture',project_attestation_sha256=sha(attest),observed_epoch=1000),True)
            ledger=SimpleNamespace(guard=lambda:None,cpu=lambda value:None)
            with patch.object(launcher,'native_bootstrap',return_value=({},r,directory,'123','compute-fixture')),\
                    patch.object(launcher,'Ledger',return_value=ledger),patch('scripts.pi0294.pv1.neural.stage_work'),\
                    patch.object(launcher.time,'time',return_value=1000),\
                    patch.object(launcher.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as child:
                self.assertEqual(launcher.run(root,request),0)
                self.assertEqual(child.call_args.args[0],['/CPU-fixture/python','-m','scripts.pi0294.pv1.neural',
                    '--root',str(root),'--request',str(request)])
