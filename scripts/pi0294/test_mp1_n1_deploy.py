import hashlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from .mp1 import n1_boundary as boundary, n1_deploy as deploy, n1_limits as n1
from .native_dataset import sha
from . import test_mp1_n1_boundary as fixture


class ColdDeployment(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.N1BoundaryTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def inputs(self):
        f=self.fixture
        contents={name: b'fixed-learning-'+name.encode() for name in f.old['code']['unchanged_learning_files']}
        contents.update({name: b'fixed-numerical-'+name.encode() for name in f.old['code']['unchanged_numerical_files']})
        return dict(prior_ledger=f.past, native_terminals=f.native_rows, outcomes=f.rows,
            effects=f.effects, processes=f.processes,
            authority_files={name:(self.root/name).read_bytes() for name in n1.FILES},
            fixture_plan={'scope':'CPU_TEST_ONLY_SYNTHETIC_RECORDED_REFERENCE'}),contents

    def test_stage_preserves_pointer_and_activate_exact_manifest(self):
        f=self.fixture;args,contents=self.inputs()
        # This synthetic capsule checks staging mechanics; boundary identity gates are tested independently.
        f.old['code']['unchanged_learning_files']={n:hashlib.sha256(v).hexdigest() for n,v in contents.items()}
        f.old['code']['unchanged_numerical_files']={}
        f.put(self.root/'RUN_MANIFEST.json',f.old)
        for name in boundary.paths(6):
            (self.root/name).unlink()
        old=(self.root/'RUN_MANIFEST.json').read_bytes()
        with patch.object(boundary,'validate',return_value=f.q):
            result=deploy.stage(self.root,6,'a'*40,contents,**args)
            self.assertEqual((self.root/'RUN_MANIFEST.json').read_bytes(),old)
            result2=deploy.stage(self.root,6,'a'*40,contents,**args)
            self.assertEqual(result,result2)
            value=deploy.activate(self.root,result)
        self.assertEqual(value['status'],'ACTUAL_COLD_N1_MANIFEST_ADVANCED')
        self.assertEqual(sha(self.root/'RUN_MANIFEST.json'),result['proposed_manifest_sha256'])
        with self.assertRaises(PermissionError):deploy.activate(self.root,result)

    def test_live_native_consumer_is_not_fenced_by_metadata(self):
        record=self.root/'aux_worker_processes/live.json'
        self.fixture.put(record,dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21]))
        with self.assertRaisesRegex(PermissionError,'STILL_LIVE'):deploy.assert_no_consumers(self.root)

    def test_pending_resources_stop_and_unknown_artifact_rejected(self):
        self.fixture.put(self.root/'RESOURCE_LEDGER.json',dict(reservations={'unknown':{}},CPU_reserved_s=0))
        with self.assertRaisesRegex(PermissionError,'RESERVATION'):deploy.assert_no_consumers(self.root)
        path=self.root/'metadata.json'
        deploy.put(path,b'one')
        deploy.put(path,b'one')
        with self.assertRaises(PermissionError):deploy.put(path,b'two')
        path.chmod(0o600)
        with self.assertRaises(PermissionError):deploy.put(path,b'one')

    def test_original_learning_mismatch_and_member_escape_fail_before_activation(self):
        args,contents=self.inputs()
        with self.assertRaisesRegex(PermissionError,'ORIGINAL_NUMERICAL'):
            deploy.stage(self.root,6,'a'*40,contents,**args)
        for name in ['/absolute','../escape','x/../escape']:
            with self.subTest(name=name),self.assertRaises(PermissionError):
                deploy.stage(self.root,6,'b'*40,{name:b''},**args)


if __name__=='__main__':unittest.main()
