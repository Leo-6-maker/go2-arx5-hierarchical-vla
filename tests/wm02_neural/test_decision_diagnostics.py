import hashlib,tempfile,unittest
from pathlib import Path
import numpy as np
try:import torch
except ModuleNotFoundError as e:
    if e.name!='torch':raise
    raise unittest.SkipTest('Torch environment required')
from go2arx5_vla.world_model.decision_diagnostics import choices,rule_scores,frozen_model,stable_choice
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
            torch.save(dict(schema='WM02_ENGINEERING_CHECKPOINT_V1',provenance={'unit':('not native',)},step=200,
                model_config=dict(hidden=[16],history_hidden=8,groups=GROUPS,context_dim=5),model=model.state_dict()),p)
            binding=dict(path=str(p),sha256=digest(p),provenance={'unit':['not native']});loaded=frozen_model(binding)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            self.assertTrue(all(torch.equal(x,y)for x,y in zip(model.state_dict().values(),loaded.state_dict().values())))
            binding['provenance']['unit']=['wrong source']
            with self.assertRaises(ValueError):frozen_model(binding)
            binding['sha256']='0'*64
            with self.assertRaises(ValueError):frozen_model(binding)

    def test_prompt_closure_beats_equal_endpoint_delay(self):
        s=np.zeros(50);raw=np.zeros((2,8,7));raw[:,:,6]=.05;raw[1,:4,6]=.95
        self.assertEqual(stable_choice(rule_scores(s,raw),raw),0)

    def test_numeric_ties_and_exact_blind_scores_ignore_candidate_order(self):
        raw=np.zeros((5,8,7));raw[:,:,6]=np.arange(5)[:,None]/5
        scores=np.ones(5)*.9;scores[1]+=1e-8
        chosen=raw[stable_choice(scores,raw)].copy();ix=[4,2,0,3,1]
        self.assertTrue(np.array_equal(chosen,raw[ix][stable_choice(scores[ix],raw[ix])]))
        self.assertEqual(stable_choice(scores,raw),stable_choice(np.ones(5)*.9,raw))

    def test_blind_prediction_is_exactly_identical_for_all_future_commands(self):
        from go2arx5_vla.world_model.online_selector import choose
        from unittest.mock import patch
        seen=[]
        def predict(*args,**kwargs):
            seen.append((args[1].shape[0],kwargs['blind']))
            return None,torch.full((1,16,4),.7),None
        history=[dict(state_before={},reference_compiled=np.zeros((40,10)),t0=0.,native_progress={'point':0})]*8
        with patch('go2arx5_vla.world_model.online_selector.tensor_state',return_value=np.zeros(50)),patch('go2arx5_vla.world_model.contact_training.predict',side_effect=predict):
            scores=choose(None,history,{'time_s':0.},np.zeros(5),np.zeros((5,8,7)),True)
        self.assertEqual(seen,[(1,True)]);self.assertTrue(np.array_equal(scores,np.repeat(scores[0],5)))

if __name__=='__main__':unittest.main()
