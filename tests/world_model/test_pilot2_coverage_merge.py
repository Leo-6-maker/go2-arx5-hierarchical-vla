import json,tempfile,unittest
from pathlib import Path
import numpy as np
from tests.world_model.test_pilot1_feedback import saved
from scripts.wm02.build_pilot2_coverage import merge
from go2arx5_vla.world_model.target_replay import load


class CoverageMerge(unittest.TestCase):
    def incoming(self,path,splits,**meta_changes):
        saved(path,splits,'new',True)
        with np.load(path,allow_pickle=False)as a:raw={k:a[k].copy()for k in a.files}
        meta=json.loads(raw['metadata_utf8'].tobytes());meta.update(manifest_schema='WM02_PILOT2_COVERAGE_REPLAY_MANIFEST_V1',**meta_changes)
        raw['metadata_utf8']=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8);np.savez_compressed(path,**raw)

    def test_train_and_dev_remain_distinct_and_old_dev_is_unchanged(self):
        with tempfile.TemporaryDirectory()as tmp:
            r=Path(tmp);saved(r/'base.npz',['TRAIN','DEV'],'old');self.incoming(r/'new.npz',['TRAIN','DEV'])
            result=merge(r/'base.npz',r/'new.npz',r/'all.npz');data,meta=load(r/'all.npz')
            self.assertEqual(result['new_TRAIN_windows'],1);self.assertEqual(result['new_DEV_windows'],1)
            self.assertEqual([e['split']for e in meta['episodes']],['TRAIN','DEV','TRAIN','DEV'])
            np.testing.assert_array_equal(data['state'][1],1.);np.testing.assert_array_equal(data['state'][3],2.)

    def test_test_or_executor_migration_rejected(self):
        for splits,change in [(['TEST'],{}),(['TRAIN'],dict(robot_fingerprint='DIFFERENT_NATIVE'))]:
            with self.subTest(splits=splits,change=change),tempfile.TemporaryDirectory()as tmp:
                r=Path(tmp);saved(r/'base.npz',['TRAIN','DEV'],'old');self.incoming(r/'new.npz',splits,**change)
                with self.assertRaises(ValueError):merge(r/'base.npz',r/'new.npz',r/'all.npz')


if __name__=='__main__':unittest.main()
