from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import patch

from .mp1.relay_qualification import filesystem_probe


class FilesystemQualificationTests(unittest.TestCase):
    def test_request_to_result_fixture_checks_real_bound_paths_and_lease(self):
        from .mp1 import relay_qualification as q
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'plans').mkdir();(root/'evidence').mkdir()
            spec=dict(run_uuid=root.name,role='hopper',purpose='SYNTHETIC_FILESYSTEM_TIMING_NO_MODEL_OR_PHYSICS',source_sha='fixture',maximum_CPU_s=600,maximum_wall_s=180,cycles=96,attempt=2)
            (root/'plans/FILE_RPC_CPU_QUALIFICATION_hopper_02.json').write_text(json.dumps(spec))
            (root/'plans/FILE_RPC_CPU_QUALIFICATION_hopper_01.json').write_text('original-failed-request-preserved')
            (root/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json').write_text(json.dumps({'hard_closeout_epoch':time.time()+300}))
            (root/'OUTPUT_BINDING.private.json').write_text(json.dumps({'lease':str(root/'LEASE.json')}))
            (root/'LEASE.json').write_text(json.dumps({'run_uuid':root.name,'status':'ACTIVE_CONTROL_PLANE_ONLY'}))
            with patch.object(q,'validate',return_value={'code':{'vla_commit':'fixture'}}),patch.object(q,'charge_cpu') as charge:
                value=q.run(root,'hopper',2)
                self.assertEqual(value['cycles'],96);self.assertEqual(value['status'],'PASS_SAME_HOST_FILESYSTEM_PROBE_NOT_CLOSED_LOOP')
                self.assertTrue((root/'evidence/file-rpc-cpu-qualification-hopper-02/RESULT.json').is_file())
                self.assertEqual((root/'plans/FILE_RPC_CPU_QUALIFICATION_hopper_01.json').read_text(),'original-failed-request-preserved')
                charge.assert_called_once()

    def test_real_fixture_durability_stays_synthetic_and_bounded(self):
        calls=[]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture';v=filesystem_probe(p,lambda:calls.append(1),cycles=3)
            self.assertEqual(v['cycles'],3);self.assertEqual(len(calls),3)
            self.assertEqual(v['new_model_queries'],v['new_physical_integrations'],0)
            self.assertFalse(v['full_32s_closed_loop_qualification'])
            self.assertFalse(v['cross_compute_node_NFS_visibility_tested'])
            self.assertEqual(len(v['summary']),6)
            self.assertTrue(all(x['samples']==3 for x in v['summary'].values()))
            with self.assertRaises(FileExistsError):filesystem_probe(p,lambda:None,cycles=1)

    def test_bound_and_guard_failure_are_not_silent_success(self):
        with tempfile.TemporaryDirectory() as d:
            for cycles in [0,97,True]:
                with self.assertRaises(ValueError):filesystem_probe(Path(d)/'bad',lambda:None,cycles=cycles)
            def stopped():raise PermissionError('STOP')
            with self.assertRaises(PermissionError):filesystem_probe(Path(d)/'stop',stopped,cycles=1)
