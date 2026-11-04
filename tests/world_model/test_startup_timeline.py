import json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
from go2arx5_vla.world_model.startup_timeline import run
from go2arx5_vla.world_model.storage import digest
class StartupTimeline(unittest.TestCase):
    def test_unrecorded_policy_pd_and_untriggered_STOP_remain_unknown(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);episodes={}
            for name in('S1_RECOVERY01_T1','S1_RECOVERY01_T2'):
                d=root/name;d.mkdir();state=np.zeros(86,dtype='<f8');raw=struct.pack('<8sQdIII',b'GRTSTATE',1,.002,34,32,20)+state.tobytes()+struct.pack('<I',0);(d/'state.bin.journal').write_bytes(struct.pack('<I',len(raw))+raw)
                values={'initialization.jsonl':dict(qpos=[0.]*34,qvel=[0.]*32,ctrl=[0.]*20),'BEFORE_FIRST_INTEGRATION.json':dict(control_calls=1),'physics.jsonl':dict(kind='model',joints=[])}
                for p,v in values.items():(d/p).write_text(json.dumps(v)+'\n')
                (d/'control.jsonl').write_text('');episodes[name]={p.name:digest(p)for p in d.iterdir()}
            v=run(dict(root=str(root),source_sha='s',episodes=episodes,joint_mapping=list(range(18))),root/'result.json');e=v['episodes']['S1_RECOVERY01_T2'];self.assertIsNone(e['extra_integrations_after_first_STOP']);self.assertEqual(e['timeline'][1]['actual_PD_calls_at_this_tick'],'UNKNOWN_SPARSE_CONTROL_TRACE');self.assertEqual(e['timeline'][1]['policy_observation_history'],'UNKNOWN_NOT_RECORDED');self.assertEqual(v['new_native_integrations'],0)
if __name__=='__main__':unittest.main()
