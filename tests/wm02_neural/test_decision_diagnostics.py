import hashlib,tempfile,unittest
from pathlib import Path
import numpy as np
try:import torch
except ModuleNotFoundError as e:
    if e.name!='torch':raise
    raise unittest.SkipTest('Torch environment required')
from go2arx5_vla.world_model.decision_diagnostics import choices,rule_scores,frozen_model
from go2arx5_vla.world_model.neural import WorldModel
from go2arx5_vla.world_model.replay import GROUPS,SLICES
from go2arx5_vla.world_model.storage import digest


class DecisionDiagnostics(unittest.TestCase):
    def test_regret_is_independent_of_nominal_improvement_and_candidate_position(self):
        names=np.array(['hold','nominal','delay']);truth=np.array([0,1,0]);score=np.array([.1,.9,.2]);rule=score.copy()
        a=choices(score,truth,names,rule,16)
        self.assertEqual(a['WM']['regret'],0);self.assertEqual(a['nominal_Q'],1)
        ix=[2,0,1];b=choices(score[ix],truth[ix],names[ix],rule[ix],16)
        self.assertEqual(a['WM'],b['WM'])
    def test_rule_uses_only_same_current_observation_and_proposed_commands(self):
        s=np.zeros(50);s[SLICES['tcp_position']]=[.4,0,.5];s[SLICES['object_position']]=[.4,0,.46]
        raw=np.zeros((3,8,7));raw[:,:,6]=.05;raw[0,:,6]=.95;raw[2,:,0]=.0005
        self.assertEqual(int(np.argmax(rule_scores(s,raw))),1)
    def test_frozen_loader_verifies_hash_and_provenance_without_optimizer_or_cuda(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'model.pt';model=WorldModel(hidden=(16,),history_hidden=8,groups=GROUPS,context_dim=5)
            torch.save(dict(schema='WM02_ENGINEERING_CHECKPOINT_V1',provenance={'unit':'not native'},step=200,
                model_config=dict(hidden=[16],history_hidden=8,groups=GROUPS,context_dim=5),model=model.state_dict()),p)
            binding=dict(path=str(p),sha256=digest(p),provenance={'unit':'not native'});loaded=frozen_model(binding)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            self.assertTrue(all(torch.equal(x,y)for x,y in zip(model.state_dict().values(),loaded.state_dict().values())))
            binding['sha256']='0'*64
            with self.assertRaises(ValueError):frozen_model(binding)

if __name__=='__main__':unittest.main()
