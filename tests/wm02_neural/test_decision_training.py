import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from go2arx5_vla.world_model.decision_training import read,quality
from go2arx5_vla.world_model.replay import SCHEMA,SLICES

class DecisionTraining(unittest.TestCase):
    def test_explicit_schema_whole_parent_partition_and_no_old_purpose_alias(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'data.npz';arrays=dict(purpose=np.array('WM02_DECISION_PAIRED_REPLAY_V5'),schema_json=np.array(json.dumps(SCHEMA)),
                scene=np.arange(40),anchor=np.array(['preclose']*40),candidate=np.array(['hold']*40),partition=np.array(['FIT']*24+['DIAGNOSTIC_DEV']*8+['ONCE_HOLDOUT']*8))
            for k,shape in dict(state=(40,25,50),reference=(40,24,40,10),events=(40,24,4),valid=(40,24),done=(40,24),timeout=(40,24),clock=(40,24,2),context=(40,5),raw=(40,8,7),initial_event=(40,4)).items():arrays[k]=np.zeros(shape)
            np.savez_compressed(p,**arrays);data,meta=read(p,'cpu');self.assertEqual(len(data['state']),40)
            arrays['partition'][24]='FIT';np.savez_compressed(p,**arrays)
            with self.assertRaises(ValueError):read(p,'cpu')
            arrays['purpose']=np.array('WM02_S_CONTACT_PAIRED_REPLAY_V1');np.savez_compressed(p,**arrays)
            with self.assertRaises(ValueError):read(p,'cpu')

    def test_release_requires_support_and_transport_prefers_goal_not_height(self):
        state=torch.zeros(2,50);events=torch.zeros(2,4);context=torch.zeros(2,5)
        events[:,3]=torch.tensor([0.,1.]);self.assertEqual(quality(state,events,context,'supported_release').tolist(),[0.,1.])
        events[:,2]=1;state[:,SLICES['object_position']]=torch.tensor([[.1,0,1.],[.01,0,.5]])
        self.assertGreater(float(quality(state,events,context,'transport')[1]),float(quality(state,events,context,'transport')[0]))

if __name__=='__main__':unittest.main()
