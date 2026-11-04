import unittest
import numpy as np
import torch
from go2arx5_vla.world_model.response_diagnostics import EVENTS,groups

class ResponseDiagnostics(unittest.TestCase):
    def test_named_event_semantics_precede_model_comparison(self):
        self.assertEqual(EVENTS['native_grasp'],2);self.assertEqual(EVENTS['illegal_contact'],3)
        self.assertNotIn('support',EVENTS)

    def test_common_start_requires_history_context_and_clock_identity(self):
        meta=dict(scene=np.zeros(5,dtype=int),partition=np.array(['FIT']*5),anchor=np.array(['transport']*5),candidate=np.array(['hold','toward_goal','away','left','right']))
        data=dict(state=torch.zeros(5,25,50),reference=torch.zeros(5,24,40,10),context=torch.zeros(5,5),clock=torch.zeros(5,24,2))
        self.assertEqual(len(groups(data,meta)),1)
        for key,index in [('state',(1,0,0)),('reference',(1,0,0,0)),('clock',(1,0,0)),('context',(1,0))]:
            data[key][index]=1
            with self.assertRaises(ValueError):groups(data,meta)
            data[key][index]=0

if __name__=='__main__':unittest.main()
