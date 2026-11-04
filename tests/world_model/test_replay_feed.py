import unittest,tempfile,threading,json
from pathlib import Path
import numpy as np
from go2arx5_vla.world_model.replay_feed import publish_replay,read_snapshot
class ReplayFeed(unittest.TestCase):
    def test_private_file_feed_has_no_actions_and_refuses_fake_live_identity(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'snapshot.json';cfg={'original_native_source_sha':'s','scene':{'sha256':'m'},'journal':{'sha256':'j'}};row=dict(step=1,time_s=.002,qpos=np.zeros(34),qvel=np.zeros(32),ctrl=np.zeros(20),contacts=[]);publish_replay([row],cfg,p,threading.Event());v=read_snapshot(p,cfg);self.assertEqual(v['step'],1);self.assertIsNone(read_snapshot(p,cfg,1));raw=json.loads(p.read_text());self.assertNotIn('action',raw);self.assertIsNone(raw['recorded_monotonic_s']);raw['label']='LIVE_TARGET_AUTHORIZED';p.write_text(json.dumps(raw))
            with self.assertRaises(PermissionError):read_snapshot(p,cfg)
            raw['label']='TARGET_FAILED_REPLAY';raw['qpos']=[0.];p.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):read_snapshot(p,cfg)
if __name__=='__main__':unittest.main()
