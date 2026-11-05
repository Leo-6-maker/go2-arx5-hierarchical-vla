import json
import tempfile
import unittest
from pathlib import Path
from scripts.wm02.audit_target_population import select
from scripts.wm02.audit_recorded_target import quaternion_error
from types import SimpleNamespace
import numpy as np


class PopulationSelection(unittest.TestCase):
    def test_quaternion_validation_handles_free_ball_and_hinge_layout(self):
        model=SimpleNamespace(njnt=3,jnt_type=[0,1,3],jnt_qposadr=[0,7,11])
        q=np.r_[0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,0.]
        self.assertEqual(quaternion_error(model,[dict(qpos=q)]),0.)
        q[7]=.5
        with self.assertRaisesRegex(ValueError,'RECORDED_QUATERNION_NOT_UNIT'):
            quaternion_error(model,[dict(qpos=q)])

    def test_passed_child_of_failed_pair_and_partial_worker_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Path(tmp)
            batch = c / 'pair'
            (batch / 'batch-01').mkdir(parents=True)
            (batch / 'batch-01/RESULT.json').write_text(json.dumps(dict(status='FAILED')))
            for name, status, lease, peer in [('good', 'NATIVE_STAGE_PASSED', 'CLOSED', False), ('failed', 'NATIVE_STAGE_FAILED_PRESERVED', 'CLOSED', False), ('peer', 'NATIVE_STAGE_PASSED', 'CLOSED', True), ('live', 'NATIVE_STAGE_PASSED', 'ACTIVE', False)]:
                r = c / 'runs' / ('night1-' + name)
                (r / 'control').mkdir(parents=True)
                (r / 'native-01').mkdir()
                files = {'EPISODE_MANIFEST.private.json':dict(purpose='PRODUCTION'), 'EXECUTION_SPEC.json':dict(parallel_batch_spec=str(batch/'SPEC.json')) if peer else {}, 'native-01/RESULT.json':dict(status=status), 'control/LEASE.json':dict(status=lease)}
                for path, value in files.items():
                    (r / path).write_text(json.dumps(value))
            included, excluded = select(c)
            self.assertEqual([r[0].name for r in included], ['night1-good'])
            self.assertEqual({r['reason'] for r in excluded}, {'NATIVE_FAILED', 'WHOLE_PEER_BATCH_NOT_PASSED', 'WORKER_NOT_SEALED'})


if __name__ == '__main__':
    unittest.main()
