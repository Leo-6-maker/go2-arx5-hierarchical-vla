import base64
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from scripts.pi0292.relay_protocol import sealed
from scripts.pi0294.native_dataset import sha
from .mp1.anchor_restore import load_owned_snapshot
from .mp1.failure_config import ATTEMPTS


def fixture(root):
    identifier=next(iter(ATTEMPTS));folder=Path(root)/'evidence/TRAIN_STATES'/identifier;folder.mkdir(parents=True)
    np.savez(folder/'integration.npz',live=np.zeros(3),ik=np.zeros(3),observed=np.zeros(3))
    (folder/'history.private.pkl').write_bytes(pickle.dumps(dict(engine_python={},teacher={},python_rng=None,numpy_rng=None)))
    raw=b'trusted-original-camera-fixture';checksum=hashlib.sha256(raw).hexdigest()
    packet=sealed(dict(sequence=3,tick=600,episode_uuid='fixed',payload=dict(images=[dict(camera=c,png_base64=base64.b64encode(raw).decode(),sha256=checksum) for c in ['external_rgb','wrist_rgb']])))
    (folder/'observation.json').write_text(json.dumps(packet));(folder/'ORIGINAL_ACTION_BUNDLE.private.json').write_text('{}')
    return dict(original_attempt_id=identifier,sequence=3,tick=600,split='TRAIN',physically_safety_valid=True,
        snapshot_relative=identifier,full_integration_state_sha256=sha(folder/'integration.npz'),history_sha256=sha(folder/'history.private.pkl'),original_action_bundle_sha256=sha(folder/'ORIGINAL_ACTION_BUNDLE.private.json'),observation_sha256=packet['sha256'],original_episode_uuid='fixed'),folder


class AnchorRestoreTests(unittest.TestCase):
    def test_owned_sealed_snapshot_keeps_complete_live_IK_and_original_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            row,_=fixture(tmp);arrays,history,body=load_owned_snapshot(tmp,row)
            self.assertEqual(set(arrays),{'live','ik','observed'});self.assertEqual(body['tick'],600)

    def test_changed_pickle_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            row,folder=fixture(tmp);(folder/'history.private.pkl').write_bytes(b'changed-not-a-pickle')
            with self.assertRaisesRegex(ValueError,'HASH_BEFORE_DESERIALIZATION'):load_owned_snapshot(tmp,row)

    def test_initial_DEV_and_path_escape_are_not_qualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            row,_=fixture(tmp)
            for change in [dict(sequence=0,tick=0),dict(split='DEV'),dict(physically_safety_valid=False),dict(snapshot_relative='../foreign')]:
                with self.subTest(change=change),self.assertRaises(PermissionError):load_owned_snapshot(tmp,dict(row,**change))

    def test_missing_history_or_runtime_object_injection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            row,folder=fixture(tmp);p=folder/'history.private.pkl'
            for value in [dict(engine_python={}),dict(engine_python=dict(model='foreign'),teacher={},python_rng=None,numpy_rng=None)]:
                p.write_bytes(pickle.dumps(value));row['history_sha256']=sha(p)
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,'HISTORY_NO_MODEL'):load_owned_snapshot(tmp,row)


if __name__=='__main__':unittest.main()
